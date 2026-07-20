import os
import numpy as np
import json
import copy
import torch
import torch.nn.functional as F

try:
    from onnx_utils import compress_onnx_model, uncompress_onnx_model
    from onnx_utils import process_onnx
    from onnx_utils import fix_onnx_model_name
except Exception:
    print("not support onnx export utils")


def process_embedding_weights(ckpt, mul_twice=False, onnx_output_dir=None, omc_name="", fp16=True):
    weight = ckpt["model.embed_tokens.weight"].to(torch.float32)
    if "model.embed_tokens.s" in ckpt:
        sx = ckpt["model.embed_tokens.s"]
    elif "model.embed_tokens.quant_op.weight_quantizer.s" in ckpt:
        sx = ckpt["model.embed_tokens.quant_op.weight_quantizer.s"]
    else:
        ma = torch.max(weight, dim=1, keepdim=True)[0]
        mi = torch.min(weight, dim=1, keepdim=True)[0]
        sa = ma / 127
        si = -mi / 128
        sx = torch.where(sa > si, sa, si)
        sx = torch.max(sx, torch.tensor(1.e-10))

    if sx.dim() != weight.dim():
        sx = sx.reshape(-1, *([1] * (weight.dim() - 1)))

    qw = weight / sx
    fqw = torch.clamp(torch.round(qw), -128, 127)
    if mul_twice: sx = torch.sqrt(sx).to(torch.float16).to(torch.float32)
    if onnx_output_dir is not None:
        df = qw - fqw
        print(f"quant_diff max is {df.max()}, min {df.min()}")
        qw = fqw.to(torch.int8)
        qw.cpu().numpy().tofile(f"{onnx_output_dir}/{omc_name}.embedding_weights")
        sx.cpu().numpy().tofile(f"{onnx_output_dir}/{omc_name}.embedding_dequant_scale")

    if fp16:
        sx = sx.to(torch.float16)
        fqw = fqw.to(torch.float16)
    fqw = fqw * sx
    if mul_twice: fqw = fqw * sx

    ckpt["model.embed_tokens.weight"] = fqw.to(torch.float32)
    if sx.dim() == 2:
        assert sx.shape[1] == 1
        sx.reshape(-1)
    ckpt["model.embed_tokens.quant_op.weight_quantizer.s"] = sx.to(torch.float32)

    return ckpt


class FLinearMatmul(torch.nn.Module):
    def __init__(
            self, m: torch.nn.Linear,
            algin_dim=1,
            squeeze_batch=False,
            unsqueeze_before_bias=False,
    ):
        super().__init__()

        self.in_features = m.in_features
        self.out_features = m.out_features
        self.squeeze_batch = squeeze_batch
        self.unsqueeze_before_bias = unsqueeze_before_bias

        if m.out_features % algin_dim != 0:
            self.out_features = (m.out_features + algin_dim - 1) // algin_dim * algin_dim
            weight = torch.zeros((self.out_features, self.in_features))
            weight[:m.out_features, :] = m.weight.data
            self.weight = weight
            if self.bias is not None:
                bias = torch.zeros(self.out_features); bias[:m.out_features] = m.bias.data
                self.bias = bias
        else:
            self.weight = m.weight
            self.bias = m.bias

    def __repr__(self):
        return self.__class__.__name__ + f"(in_features={self.in_features}, out_features={self.in_features}, bias={self.bias is not None})"

    def add_bias(self):
        if self.bias is None:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features), requires_grad=True)

    def forward(self, x):
        if self.squeeze_batch:
            x = x.reshape(x.shape[1], x.shape[2])

        out = F.linear(
            x,
            self.weight,
        )

        if self.squeeze_batch and self.unsqueeze_before_bias:
            out = out.reshape(1, out.shape[0], out.shape[1])

        if self.bias is not None:
            out = out + self.bias

        if self.squeeze_batch and (not self.unsqueeze_before_bias):
            out = out.reshape(1, out.shape[0], out.shape[1])

        return out


def replace_module_by_names(model, modules_to_replace):
    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            is_replaced = False
            for full_name, m in model.named_modules():
                if full_name not in modules_to_replace:
                    continue
                if c is m:
                    child.add_module(n, modules_to_replace.pop(full_name))
                    is_replaced = True
                    break
            if not is_replaced:
                helper(c)

    helper(model)
    return model


def optimize_model_add_lora_layers(model, config_file=None):
    if config_file is None: return model

    with open(config_file, "r") as f: config = json.load(f)
    lora_config = config.get("lora_strategy", dict())

    replace_modules = dict()
    for name, m in model.named_modules():
        if name in lora_config:
            replace_modules[name] = FLinearLora(m, **lora_config[name])
            print(f"updating {name} to lora fusion")
    if replace_modules:
        model = replace_module_by_names(model, replace_modules)
    return model


def load_state_dict(model, ckpt, load_args=None):
    if load_args is None: load_args = dict()
    if load_args.get("refresh_input_quantize", False):
        ckpt_keys = list(ckpt.keys())
        for k in ckpt_keys:
            if "input_quantizer" in k:
                ckpt.pop(k)

    if load_args.get("refresh_weight_quantize", False):
        ckpt_keys = list(ckpt.keys())
        for k in ckpt_keys:
            if "weight_quantizer" in k:
                ckpt.pop(k)
    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception as ext:
        model.load_state_dict(ckpt, strict=False)
    return model


def optimize_model_gemm2matmul(model, algin_dim=1, squeeze_batch=False, unsqueeze_before_bias=False):
    replace_modules = dict()
    for name, m in model.named_modules():
        if type(m) == torch.nn.Linear:
            replace_modules[name] = FLinearMatmul(m, algin_dim, squeeze_batch, unsqueeze_before_bias)
            print(f"updating {name} to matmul + bias")
    if replace_modules:
        model = replace_module_by_names(model, replace_modules)
    return model


def generate_lora_config(model_dir):
    import onnx
    onnx_model = onnx.load(model_dir)
    lora_config = dict()

    def set_lora_param(layer_name, lora_op, name, lora_config):
        if layer_name not in lora_config:
            lora_config[layer_name] = dict()
        lora_config[layer_name][lora_op] = name
    tensor2adds = dict()
    for node in onnx_model.graph.node:
        if node.op_type == "Add":
            tensor2adds[node.input[0]] = node.name
            tensor2adds[node.input[1]] = node.name

    for node in onnx_model.graph.node:
        layer_name = ".".join(node.name.split(".")[:-1])
        if node.name.endswith(".lora_A"): set_lora_param(layer_name, "MatMulA", node.name, lora_config)
        if node.name.endswith(".lora_B"):
            set_lora_param(layer_name, "MatMulB", node.name, lora_config)
            set_lora_param(layer_name, "Add",tensor2adds[node.output[0]], lora_config)

    config_to_save = list()
    for k in lora_config:
        config_to_save.append(lora_config[k])

    lora_config = {
        "modelType": "loraModel",
        "loraLayers": config_to_save,
    }
    with open(model_dir[:-5] + "_lora_config.json", "w") as f:
        json.dump(lora_config, f, indent=4)
