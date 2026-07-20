# Kirin NPU LLM — 鸿蒙命令行大模型推理与评测

在 HarmonyOS / OpenHarmony 设备上，通过命令行运行 Qwen2.5 等大模型推理与工具调用准确率评测。基于 **HiAI LLM Engine**（Kirin NPU）。

| 可执行文件 | 用途 |
|------------|------|
| `llm_bin` | 交互对话、批处理推理与 benchmark |
| `llm_eval` | 工具调用（tool calling）准确率评测 |

## 功能特性

**llm_bin**

- 交互式多轮对话（流式输出 token）
- 批处理：从 `prompt.txt` 按行推理并输出 benchmark 统计
- 支持 `/exit`、`/reset` 命令
- 可通过命令行参数覆盖最大生成长度 `max_gen_tokens`

**llm_eval**

- 读取 JSONL 评测集，按 Qwen ChatML 模板构造带工具定义的 system prompt
- 解析模型输出中的 `<tool_call>...</tool_call>`，与标注答案比对
- 输出「工具正确」（名称 + 数量）与「全部正确」（名称 + 参数 + 数量）及汇总统计
- 评测逻辑与 `llm_eval.py` 对齐，可在设备端直接跑 NPU 推理评测

**通用**

- 交叉编译为 `aarch64-linux-ohos` 可执行文件

## 环境要求

| 项目 | 说明 |
|------|------|
| 主机 | Linux，已安装 CMake ≥ 3.16 |
| OHOS NDK | OpenHarmony / HarmonyOS SDK 的 `native` 目录 |
| 设备 | 支持 HiAI NPU 的 Kirin 鸿蒙真机 |
| 工具 | `hdc`（推送与 shell 调试） |

## 编译

```bash
cd kirin-npu-llm/engine_ohos

export OHOS_NDK=/path/to/openharmony/native

cmake -S . -B build-ohos \
  -DCMAKE_TOOLCHAIN_FILE=ohos-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-ohos -j
```

编译成功后：

```bash
file build-ohos/llm_bin build-ohos/llm_eval
# 期望: ELF 64-bit ... ARM aarch64 ... ld-musl-aarch64
```


## llm_bin 用法

```text
llm_bin <model_dir> [prompt.txt] [max_tokens] [--json-out FILE]
```

| 模式 | 命令 | 说明 |
|------|------|------|
| 交互对话 | `./llm_bin qwen25` | 多轮 stdin 输入，流式打印回复 |
| 批处理 | `./llm_bin qwen25 prompts.txt` | 每行一个 prompt，终端打印 prompt/answer |
| JSON 批处理 | `./llm_bin qwen25 prompts.txt --json-out results.json` | 结果写入 JSON，终端仅进度与统计 |
| 限制长度 | `./llm_bin qwen25 prompts.txt 100` | 覆盖 `max_gen_tokens=100` |
| JSON + 限制长度 | `./llm_bin qwen25 prompts.txt 100 -o results.json` | 同上，`-o` 为 `--json-out` 简写 |

### 交互命令

| 命令 | 作用 |
|------|------|
| `/exit` | 退出程序 |
| `/reset` | 清空对话历史并重建推理 Context |

### prompt 文件格式

- 每行一个 prompt
- 空行自动跳过
- 行首 `#` 为注释

示例 `prompts.txt`：

```text
# 测试用例
你好，请介绍一下自己
1+1等于几？
```

### JSON 输出格式（`--json-out`）

每条 prompt 对应 `results` 数组中的一项，包含 `prompt`、`answer`、token 数与耗时；末尾 `summary` 为汇总统计。示例：

```json
{
  "model_dir": "qwen25",
  "prompt_file": "prompts.txt",
  "results": [
    {
      "index": 0,
      "prompt": "1+1等于几？",
      "answer": "1+1等于2。",
      "input_tokens": 42,
      "output_tokens": 8,
      "prefill_ms": 120.5,
      "decode_ms": 230.1,
      "decode_ms_per_token": 28.76
    }
  ],
  "summary": {
    "total_input_tokens": 42,
    "total_output_tokens": 8,
    "total_prefill_ms": 120.5,
    "total_decode_ms": 230.1,
    "prefill_speed_tok_s": 0.35,
    "decode_speed_tok_s": 34.8
  }
}
```

### 工作目录

**必须在包含 `qwen25/` 的目录下运行**（一般为仓库根目录）。`executor.json` 内模型、tokenizer 路径均为相对路径，例如：

- `qwen25/tokenizer.json`
- `qwen25/omc_out.omc`
- `qwen25/files`



### 2. 在设备上运行

```bash
hdc shell

cd /data/local/tmp/llm
chmod +x llm_bin llm_eval
export LD_LIBRARY_PATH=$PWD/lib64:$LD_LIBRARY_PATH

# 交互模式
./llm_bin qwen25

# 批处理
echo "你好" > /data/local/tmp/llm/t.txt
./llm_bin qwen25 t.txt


```
