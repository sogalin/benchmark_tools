#!/bin/bash

#./SERVER.sh GROK1 aiter
#./SERVER.sh LLAMA3.1-70B aiter
#./SERVER.sh LLAMA3.1-8B aiter

# Environment variables
export HF_HOME=/data/models/huggingface/
export SGLANG_TORCH_PROFILER_DIR=/sglang/python/profile/

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="sglang_server_log_$TIMESTAMP.json"

# Enable profiling if the third argument is set to "profile"
ENABLE_PROFILING=${3:-"off"}

# Select model and attention backend
MODEL=${1:-"GROK1"}  # Default to GROK1
ATTN_BACKEND=${2:-"aiter"}  # Default to aiter

# Set model-specific parameters
case "$MODEL" in
  "GROK1")
    export RCCL_MSCCL_ENABLE=0
    export CK_MOE=1
    export USE_INT4_WEIGHT=1
    MODEL_PATH="/data/models/huggingface/amd--grok-1-W4A8KV8/"
    TOKENIZER_PATH="Xenova/grok-1-tokenizer"
    TP=8
    QUANT="fp8"
    EXTRA_ARGS=""
    ;;
  "LLAMA3.1-70B")
    export RCCL_MSCCL_ENABLE=1
    MODEL_PATH="amd/Llama-3.1-70B-Instruct-FP8-KV"
    TP=8
    QUANT="fp8"
    EXTRA_ARGS="--cuda-graph-max-bs 1024 --mem-fraction-static 0.6"
    ;;
  "LLAMA3.1-8B")
    export RCCL_MSCCL_ENABLE=1
    MODEL_PATH="amd/Llama-3.1-8B-Instruct-FP8-KV"
    TP=1
    QUANT="fp8"
    EXTRA_ARGS="--cuda-graph-max-bs 1024 --mem-fraction-static 0.6"
    ;;
  *)
    echo "Unknown model name: $MODEL"
    exit 1
    ;;
esac

# Construct the command
CMD="python3 -m sglang.launch_server \
  --model $MODEL_PATH \
  ${TOKENIZER_PATH:+--tokenizer-path $TOKENIZER_PATH} \
  --tp $TP \
  --quantization $QUANT \
  --trust-remote-code \
  --attention-backend $ATTN_BACKEND \
  $EXTRA_ARGS"
# Run with or without profiling
if [ "$ENABLE_PROFILING" == "profile" ]; then
  CMD="./loadTracer.sh $CMD"
fi
echo "CMD=$CMD"
eval $CMD 2>&1 | tee "$LOGFILE"
