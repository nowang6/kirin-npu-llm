import os
import sys
import copy
import json
import onnx
import numpy as np
import gc
from onnxsim import simplify
import copy

NODE_INDICES = {}

SIZE_1MB = 1024 * 1024

COMPRESS_NODE_TYPES = ["Conv", "Gemm", "MatMul"]
CONST_OF_SHAPE_VALUE = 0.01

DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
}


def set_onnx_input_shape(onnx_model, shape_cfg):
    if not shape_cfg:
        return onnx_model
    if isinstance(shape_cfg, str):
        shape_cfg = json.loads(shape_cfg)

    graph = onnx_model.graph
    for _input in graph.input:
        if _input.name not in shape_cfg:
            continue
        tensor_shape_proto = _input.type.tensor_type.shape

        new_shape = shape_cfg[_input.name]
        # delete old shape
        elem_num = len(tensor_shape_proto.dim)
        for i in reversed(range(elem_num)):
            del tensor_shape_proto.dim[i]

        for i, d in enumerate(new_shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                d = -1
            if d < -1:
                d = f"unk_{-d}"
            if isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(f"invalid shape: {new_shape}")
    return onnx_model


def del_onnx_nodes(graph, nodes, del_node_init=False):
    unused_init_names = []
    if del_node_init:
        init_names = [init.name for init in graph.initializer]
        for node in nodes:
            for in_name in node.input:
                if in_name in init_names:
                    unused_init_names.append(in_name)

    indices = []
    for idx, node in enumerate(graph.node):
        if node in nodes:
            indices.append(idx)
    indices = sorted(indices, reverse=True)
    for idx in indices:
        del graph.node[idx]

    if del_node_init:
        del_onnx_initializers(graph, unused_init_names)


def add_onnx_inits(graph, new_inits):
    del_init_names = [init.name for init in new_inits]
    del_onnx_initializers(graph, del_init_names)
    graph.initializer.extend(new_inits)


def del_onnx_initializers(graph, del_init_names):
    indices = []
    for idx, tensor_proto in enumerate(graph.initializer):
        if tensor_proto.name in del_init_names:
            indices.append(idx)

    indices = sorted(indices, reverse=True)
    for idx in indices:
        del graph.initializer[idx]


def insert_onnx_nodes(graph, idx, new_nodes):
    new_nodes = reversed(new_nodes)
    for node in new_nodes:
        graph.node.insert(idx, node)


def create_node_name(node_type):
    global NODE_INDICES
    if node_type not in NODE_INDICES:
        NODE_INDICES[node_type] = 0
    node_id = NODE_INDICES[node_type]
    NODE_INDICES[node_type] += 1

    name = f"{node_type}_{node_id}"
    return name


def create_const_of_shape(shape, dtype=onnx.TensorProto.FLOAT, value=0.0, output_name=None, node_name=None):
    if node_name is None:
        node_name = create_node_name("ConstantOfShape")
    if not output_name:
        output_name = node_name + "_output0"
    const_shape_name = node_name + "_shape"

    shape_dim = [len(shape)]
    shape_initializer = onnx.helper.make_tensor(name=const_shape_name, data_type=onnx.TensorProto.INT64,dims=shape_dim,
                                                vals=shape, raw=False)
    tensor_value_attr = onnx.helper.make_tensor("value", dtype, dims=[1], vals=[value])

    node = onnx.helper.make_node(op_type="ConstantOfShape", inputs=[const_shape_name], outputs=[output_name],
                                 value=tensor_value_attr)
    return node, shape_initializer


def get_onnx_tensor_proto_shape(onnx_tensor_proto):
    shape = [elem for elem in onnx_tensor_proto.dims]
    return shape


def get_onnx_tensor_proto_dtype(onnx_tensor_proto):
    return onnx_tensor_proto.data_type


def shape_elem_num(shape):
    elem_num = 1
    for elem in shape:
        elem_num *= elem
    return elem_num


def compress_onnx_model(onnx_model, size_th_bytes=SIZE_1MB):
    graph = onnx_model.graph
    initializer = graph.initializer

    name_2_init_map = {}
    for init in initializer:
        name_2_init_map[init.name] = init

    new_nodes = []
    new_inits = []
    removed_inits = []

    for node in graph.node:
        if node.op_type not in COMPRESS_NODE_TYPES:
            continue
        init_name = node.input[1]
        if init_name not in name_2_init_map:
            continue

        init = name_2_init_map[init_name]
        dtype = get_onnx_tensor_proto_dtype(init)
        shape = get_onnx_tensor_proto_shape(init)

        if dtype not in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
            continue

        dtype_bytes = DTYPE_BYTES[dtype]
        shape_elem = shape_elem_num(shape)
        if shape_elem * dtype_bytes <= size_th_bytes:
            continue

        global CONST_OF_SHAPE_VALUE
        node, shape_init = create_const_of_shape(
            shape=shape, dtype=onnx.TensorProto.FLOAT, value=CONST_OF_SHAPE_VALUE, output_name=init.name)
        CONST_OF_SHAPE_VALUE += 0.003

        removed_inits.append(init)
        new_nodes.append(node)
        new_inits.append(shape_init)

    replaced_tensor_names = [init.name for init in removed_inits]
    print(f"replaced_tensor_name:{replaced_tensor_names}")
    del_onnx_initializers(graph, replaced_tensor_names)
    insert_onnx_nodes(graph, 0, new_nodes)
    add_onnx_inits(graph, new_inits)
    return onnx_model, removed_inits


def uncompress_onnx_model(onnx_model, removed_inits):
    onnx_model.graph.initializer.extend(removed_inits)
    replaced_tensor_names = [init.name for init in removed_inits]

    del_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        if node.output[0] in replaced_tensor_names:
            del_nodes.append(node)

    recover_replaced_tensors = [_node.output[0] for _node in del_nodes]
    print(f"recover_replaced_tensors: {recover_replaced_tensors}")

    del_onnx_nodes(onnx_model.graph, del_nodes, del_node_init=False)
    return onnx_model


def process_onnx(onnx_file_name, lora_export = False, model_prefix="model"):
    onnx_model = onnx.load(onnx_file_name)

    print("save onnx model")
    weight_file_name = os.path.basename(onnx_file_name).split(".onnx")[0] + ".pb"
    os.unlink(onnx_file_name)
    dirname = os.path.dirname(onnx_file_name)
    if os.path.exists(os.path.join(dirname, weight_file_name)):os.unlink(os.path.join(dirname, weight_file_name))

    size_th_kb = 1024
    size_th_bytes = size_th_kb * 1024
    onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
    print("compress model success")

    print("sim onnx model")
    tensor_size_threshold = f"{size_th_kb}KB"
    skipped_optimizers = ["fuse_matmul_add_bias_into_gemm", "fuse_qkv", "eliminate_duplicate_initializer"]
    onnx_model, check = simplify(onnx_model, tensor_size_threshold=tensor_size_threshold,
                                 skipped_optimizers=skipped_optimizers)
    print(f"sim check stats: {check}")

    onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
    print("uncompress model success")

    print(f"Modify onnx input node tensor type.")
    for node in onnx_model.graph.input:
        if node.name == "input_ids" or node.name == "position_ids" or node.name == "new_kv_cache_pos":
            node.type.tensor_type.elem_type = onnx.TensorProto.INT32

    print(f"onnx_file_name: {onnx_file_name}, weight_file_name: {weight_file_name}")
    onnx.save(onnx_model,
              onnx_file_name,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=weight_file_name,
              convert_attribute=False)

    del onnx_model
    gc.collect()
    fixed_onnx_path = fix_onnx_model_name(onnx_file_name, model_prefix=model_prefix)
    fixed_onnx_path = split_shared_weights(fixed_onnx_path)
    if lora_export:
        from do_opt import generate_lora_config; generate_lora_config(fixed_onnx_path)


def fix_onnx_model_name(model_dir, model_prefix = "model"):
    import onnx
    onnx_model = onnx.load(model_dir)
    used_name = set()
    for node in onnx_model.graph.input:
        if node.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            node.type.tensor_type.elem_type = onnx.TensorProto.INT32
            print(f"modift onnx input node {node.name} tensor type.")

    def generate_unique_name(new_name, cache):
        idx = 0
        init_name = new_name
        while new_name in cache:new_name = init_name + "_" +str(idx); idx += 1
        cache.add(new_name)
        return new_name
    for node in onnx_model.graph.node:
        if node.op_type in {"Gather"}:
            name = node.name
            if name == "":
                print("skip empty name")
                continue
            if node.input[0].endswith(".weight"): node.name = node.input[0][:-7].replace("model.model", "model")
            print(f"changing {name} to {node.name}")
            continue

        if node.op_type in {"MatMul", "Gemm", "Mul", "Cast"}:
            name = node.name
            if name == "":
                print("skip empty name")
                continue
            if name.startswith("/"): name = name[1:]
            name_list = name.split("/")
            name_list = name_list[:-1]
            new_name = ".".join(name_list)
            new_name = new_name.replace("attn_linear.attn_linear", "attn_linear")

            new_name = generate_unique_name(new_name, used_name)
            if model_prefix and new_name != "lm_head": new_name = "model." + new_name
            node.name = new_name
            print(f"changing {name} to {node.name}")

        if node.op_type in {"Transpose"}:
            name = node.name
            if name == "":
                print("skip empty name")
                continue
            if name.startswith("/"): name = name[1:]
            name_list = name.split("/")
            name_list = name_list[:-1]
            if len(name_list) == 0: continue

            new_name = ".".join(name_list)
            new_name = generate_unique_name(new_name, used_name)
            if model_prefix: node.name = model_prefix + "." + new_name
            print(f"changing {name} to {node.name}")

    onnx_file_name = os.path.realpath(model_dir)[:-5] + ".onnx"
    weight_file_name = os.path.basename(onnx_file_name)[:-5] + ".pb"
    os.unlink(model_dir)
    if os.path.exists(weight_file_name): os.remove(weight_file_name)

    onnx.save(
        onnx_model,
        onnx_file_name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weight_file_name,
        convert_attribute=False
    )
    return onnx_file_name


def split_shared_weights(model_dir):
    onnx_model = onnx.load(model_dir)

    graph = onnx_model.graph
    node_name = {}
    for node in graph.node:
        if node.op_type in {"MatMul"}:
            w_name = node.input[1]
            if w_name not in node_name:
                node_name[w_name] = 1
                continue
            for initializer in graph.initializer:
                if initializer.name == w_name:
                    node_name[w_name] = node_name[w_name] + 1
                    new_name = w_name + "_" +str(node_name[w_name])
                    node.input[1] = new_name

                    new_initailzier = copy.deepcopy(initializer)
                    new_initailzier.name = new_name
                    graph.initializer.append(new_initailzier)
    for i in node_name: print(f"name:{i},    num:{node_name[i]}")

    graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    info_model = onnx.helper.make_model(graph)
    info_model.opset_import[0].version = 12

    onnx_file_name = os.path.realpath(model_dir)[:-5] + ".onnx"
    weight_file_name = os.path.basename(onnx_file_name)[:-5] + ".pb"
    os.unlink(model_dir)
    if os.path.exists(weight_file_name): os.remove(weight_file_name)

    onnx.save(info_model,
        onnx_file_name,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weight_file_name,
        convert_attribute=False
    )
    return onnx_file_name




