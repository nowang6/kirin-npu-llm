

# Env
```sh
uv venv .venv
source .venv/bin/activate
uv pip install -e .
export PYTHONPATH=tools/tools_dopt/dopt_pytorch_py3:.
```

# Tools
Download tools and unzip to tools folder
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations


```sh
rm -rf onnx_embedding_out_no_output_pos dopt_out out

# 量化
source .venv/bin/activate
python quant_ptq.py --quant-stage stage1
python quant_ptq.py --quant-stage stage2
python quant_ptq.py --quant-stage stage3

# 导出ONNX
python export_qwen2.py

```


```sh

# 编译OMC
./tools/tools_omg/omg --model onnx_embedding_out_no_output_pos/model.onnx --framework 5 \
--output omc_out \
--input_shape="input_embed:1,-1,1536;attention_mask:1,1,-1,2048;position_ids:1,-1;past_key_in0:2048,2,1,128;past_value_in0:2048,2,1,128;past_key_in1:2048,2,1,128;past_value_in1:2048,2,1,128;past_key_in2:2048,2,1,128;past_value_in2:2048,2,1,128;past_key_in3:2048,2,1,128;past_value_in3:2048,2,1,128;past_key_in4:2048,2,1,128;past_value_in4:2048,2,1,128;past_key_in5:2048,2,1,128;past_value_in5:2048,2,1,128;past_key_in6:2048,2,1,128;past_value_in6:2048,2,1,128;past_key_in7:2048,2,1,128;past_value_in7:2048,2,1,128;past_key_in8:2048,2,1,128;past_value_in8:2048,2,1,128;past_key_in9:2048,2,1,128;past_value_in9:2048,2,1,128;past_key_in10:2048,2,1,128;past_value_in10:2048,2,1,128;past_key_in11:2048,2,1,128;past_value_in11:2048,2,1,128;past_key_in12:2048,2,1,128;past_value_in12:2048,2,1,128;past_key_in13:2048,2,1,128;past_value_in13:2048,2,1,128;past_key_in14:2048,2,1,128;past_value_in14:2048,2,1,128;past_key_in15:2048,2,1,128;past_value_in15:2048,2,1,128;past_key_in16:2048,2,1,128;past_value_in16:2048,2,1,128;past_key_in17:2048,2,1,128;past_value_in17:2048,2,1,128;past_key_in18:2048,2,1,128;past_value_in18:2048,2,1,128;past_key_in19:2048,2,1,128;past_value_in19:2048,2,1,128;past_key_in20:2048,2,1,128;past_value_in20:2048,2,1,128;past_key_in21:2048,2,1,128;past_value_in21:2048,2,1,128;past_key_in22:2048,2,1,128;past_value_in22:2048,2,1,128;past_key_in23:2048,2,1,128;past_value_in23:2048,2,1,128;past_key_in24:2048,2,1,128;past_value_in24:2048,2,1,128;past_key_in25:2048,2,1,128;past_value_in25:2048,2,1,128;past_key_in26:2048,2,1,128;past_value_in26:2048,2,1,128;past_key_in27:2048,2,1,128;past_value_in27:2048,2,1,128;new_kv_cache_pos:-1;embed_scales:1,-1,1" \
--dynamic_dims="1,1,1,1,1;64,64,64,64,64" \
--input_type="past_key_in0:FP16;past_value_in0:FP16;past_key_in1:FP16;past_value_in1:FP16;past_key_in2:FP16;past_value_in2:FP16;past_key_in3:FP16;past_value_in3:FP16;past_key_in4:FP16;past_value_in4:FP16;past_key_in5:FP16;past_value_in5:FP16;past_key_in6:FP16;past_value_in6:FP16;past_key_in7:FP16;past_value_in7:FP16;past_key_in8:FP16;past_value_in8:FP16;past_key_in9:FP16;past_value_in9:FP16;past_key_in10:FP16;past_value_in10:FP16;past_key_in11:FP16;past_value_in11:FP16;past_key_in12:FP16;past_value_in12:FP16;past_key_in13:FP16;past_value_in13:FP16;past_key_in14:FP16;past_value_in14:FP16;past_key_in15:FP16;past_value_in15:FP16;past_key_in16:FP16;past_value_in16:FP16;past_key_in17:FP16;past_value_in17:FP16;past_key_in18:FP16;past_value_in18:FP16;past_key_in19:FP16;past_value_in19:FP16;past_key_in20:FP16;past_value_in20:FP16;past_key_in21:FP16;past_value_in21:FP16;past_key_in22:FP16;past_value_in22:FP16;past_key_in23:FP16;past_value_in23:FP16;past_key_in24:FP16;past_value_in24:FP16;past_key_in25:FP16;past_value_in25:FP16;past_key_in26:FP16;past_value_in26:FP16;past_key_in27:FP16;past_value_in27:FP16" \
--output_type="lm_logits:FP32;past_key0:FP16;past_value0:FP16;past_key1:FP16;past_value1:FP16;past_key2:FP16;past_value2:FP16;past_key3:FP16;past_value3:FP16;past_key4:FP16;past_value4:FP16;past_key5:FP16;past_value5:FP16;past_key6:FP16;past_value6:FP16;past_key7:FP16;past_value7:FP16;past_key8:FP16;past_value8:FP16;past_key9:FP16;past_value9:FP16;past_key10:FP16;past_value10:FP16;past_key11:FP16;past_value11:FP16;past_key12:FP16;past_value12:FP16;past_key13:FP16;past_value13:FP16;past_key14:FP16;past_value14:FP16;past_key15:FP16;past_value15:FP16;past_key16:FP16;past_value16:FP16;past_key17:FP16;past_value17:FP16;past_key18:FP16;past_value18:FP16;past_key19:FP16;past_value19:FP16;past_key20:FP16;past_value20:FP16;past_key21:FP16;past_value21:FP16;past_key22:FP16;past_value22:FP16;past_key23:FP16;past_value23:FP16;past_key24:FP16;past_value24:FP16;past_key25:FP16;past_value25:FP16;past_key26:FP16;past_value26:FP16;past_key27:FP16;past_value27:FP16"  \
--compress_conf dopt_out/quant_params_file  \
--save_weights_as_external_data=true \
--platform=kirin9020 \
--target=omc

# 端侧部署
mkdir out
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_dequant_scale out/
cp onnx_embedding_out_no_output_pos/model_64_2048.embedding_weights out/
cp omc_out/* out/
cp app_conf/* out/
cp Qwen2.5-1.5B-Instruct/tokenizer.json out/
```
