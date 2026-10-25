#!/usr/bin/env bash
# Start the vLLM server that regex_agent.py talks to (OpenAI-compatible API on 127.0.0.1:8000).
# Stop it with:  pkill -f "[v]llm serve"      Logs: vllm_server.log
#
# Defaults: 8-bit FP8 Qwen3.5-9B (11.9 GiB weights) with 85% of the GPU (~19 GiB total) - better regex quality.
# The 4-bit build (9.5 GiB weights) saves only ~4 GB and made lower-quality regexes (extra junk fields):
#   MODEL=RedHatAI/Qwen3.5-9B-quantized.w4a16 GPU_MEM=0.62 ./start_vllm.sh
# The bigger Qwen3.8-27B (4-bit, 18 GiB of weights) fits the 24 GB card only with a short context and few parallel requests:
#   MODEL=RedHatAI/Qwen3.8-27B-INT4 SERVED=qwen3.8-27b TOOL_PARSER=qwen3_xml GPU_MEM=0.95 \
#   EXTRA_ARGS='--language-model-only --max-num-seqs 4 --max-model-len 12288 --kv-cache-dtype bfloat16 --attention-backend FLASH_ATTN' ./start_vllm.sh
# (loads in ~100 s, 20.8 GB. The checkpoint stores an fp8 attention cache, which an RTX 3090 cannot use: hence bfloat16. Without an explicit
#  backend vLLM picks FlashInfer, whose kernel does not compile on this server.)
# Then run the agent with:  LLM_REASONING_EFFORT=medium python pipeline.py regex ... --model qwen3.8-27b --workers 3
# (LLM_REASONING_EFFORT=medium makes it think less than its default)
# Another model family needs its own parser names and served name, e.g. Gemma 4 E4B:
#   MODEL=google/gemma-4-E4B-it SERVED=gemma-4-e4b TOOL_PARSER=gemma4 REASONING_PARSER=gemma4 \
#   EXTRA_ARGS='--limit-mm-per-prompt {"image":0,"audio":0}' ./start_vllm.sh
# (then run the agent with --model gemma-4-e4b)
cd "$(dirname "$0")"
MODEL=${MODEL:-RedHatAI/Qwen3.5-9B-FP8-dynamic}
SERVED=${SERVED:-qwen3.5-9b}      # name the API serves; regex_agent.py --model must match (default qwen3.5-9b)
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
EXTRA_ARGS=${EXTRA_ARGS---language-model-only}
GPU_MEM=${GPU_MEM:-0.85}          # fraction of the 24 GB GPU vLLM reserves (weights + working memory + KV cache)
# No system CUDA toolkit (nvcc) on this server -> skip the FlashInfer sampler, whose warmup compiles a kernel.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Runtime-compiled kernels (NVRTC) need the CUDA libs/nvcc that pip installed in the venv.
CUDA_PIP=~/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CUDA_PIP
export LD_LIBRARY_PATH=$CUDA_PIP/lib:$LD_LIBRARY_PATH
# ninja (installed in the venv) is needed when vLLM compiles an attention kernel on first use (Qwen3.8-27B); the venv is not activated here
export PATH=$HOME/venvs/vllm/bin:$CUDA_PIP/bin:$PATH
nohup ~/venvs/vllm/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization "$GPU_MEM" --max-num-seqs 16 \
  --enable-prefix-caching $EXTRA_ARGS \
  --reasoning-parser "$REASONING_PARSER" --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  > vllm_server.log 2>&1 &
echo "vLLM ($MODEL, gpu-mem $GPU_MEM) starting (pid $!); wait for 'Application startup complete' in vllm_server.log"
