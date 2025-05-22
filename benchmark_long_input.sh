#!/bin/bash  
  
# Ensure the script exits on the first error encountered  
set -e  
  
# Define environment variables  
export RCCL_MSCCL_ENABLE=0  
export SGLANG_AITER_MOE=1  
export SGLANG_INT4_WEIGHT=1  
export MOE_PADDING=0  
  
# Define common parameters  
BATCH_SIZE=1  
OUTPUT=10  
MODEL_PATH="/data/models/huggingface/hub/models--amd--grok-1-W4A8KV8/snapshots/f47a2b93f0215b8bb156e817a2a08fc93fffdbaa/"  
TOKENIZER_PATH="Xenova/grok-1-tokenizer"  
TP=8  
QUANTIZATION="fp8"  
ATTENTION_BACKEND="aiter"  
  
# Define an array of input sizes  
INPUT_SIZES=(1024 2048 4096 8192 16384 32768 65536 131072)
  
# Loop through each input size and run the benchmark  
for INPUT_SIZE in "${INPUT_SIZES[@]}"  
do  
  # Define the benchmark command  
  CMD="python -m sglang.bench_one_batch --batch-size $BATCH_SIZE --input $INPUT_SIZE --output $OUTPUT --model $MODEL_PATH --tokenizer-path $TOKENIZER_PATH --tp $TP --quantization $QUANTIZATION --trust-remote-code --attention-backend $ATTENTION_BACKEND --mem-fraction-static 0.2 --enable-prefill-prof"
  
  # Print the command to be executed  
  echo "Executing: $CMD"  
    
  # Execute the benchmark command  
  eval $CMD  
  
  # Move the trace file to a new name based on input size  
  mv trace.rpd trace-$INPUT_SIZE.rpd  
  
  # Define the sqlite command  
  SQLITE_CMD="sqlite3 trace-$INPUT_SIZE.rpd '.mode csv' '.header on' '.output trace-$INPUT_SIZE.csv' 'select * from top;' '.output stdout'"  
  
  # Print the sqlite command to be executed  
  echo "Executing: sqlite3 processing for trace-$INPUT_SIZE.rpd"  
  
  # Execute the sqlite command  
  eval "$SQLITE_CMD"  
  
done  

