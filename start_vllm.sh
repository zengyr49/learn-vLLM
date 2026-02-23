#!/usr/bin/env bash
# 在 CPU（含 Apple Silicon）上启动 vLLM Qwen 模型，供 test_client.py 调用
# 使用方式: ./start_vllm.sh  或  bash start_vllm.sh

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# 必须：ARM/POWER/S390X CPU 下 max_num_batched_tokens 默认 2048，小于 max_model_len 32768 会报错
# 必须：多进程通信用 127.0.0.1，否则可能解析到错误网卡 IP（如 26.26.26.1）导致 TCPStore 连接失败
export VLLM_HOST_IP=127.0.0.1

python -m vllm.entrypoints.openai.api_server \
  --model ./qwen_model/qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name qwen2.5-1.5b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float32 \
  --trust-remote-code \
  --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml   # Qwen 风格工具调用更匹配 XML parser
