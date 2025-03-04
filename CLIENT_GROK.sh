#!/bin/bash

# Start profiling via API
# curl http://localhost:30000/start_profile -H "Content-Type: application/json"

# Define the log file with a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# List of request rates
REQUEST_RATES=(1 2 4 8 16)

# Loop through each request rate
for RATE in "${REQUEST_RATES[@]}"; do
    for i in {1..3}; do  # Run each rate 3 times
        LOGFILE="sglang_client_log_grok1_${RATE}_run${i}_$TIMESTAMP.log"
        echo "Running benchmark with request rate: $RATE (Run $i)" | tee -a "$LOGFILE"
        
        NUM_PROMPTS=$(( 300 * RATE ))
        if [ "$NUM_PROMPTS" -gt 2400 ]; then
            NUM_PROMPTS=2400
        fi
        
        CMD="python3 -m sglang.bench_serving \
            --backend sglang \
            --tokenizer Xenova/grok-1-tokenizer \
            --dataset-name random \
            --random-input 1024 \
            --random-output 1024 \
            --num-prompts $NUM_PROMPTS \
            --request-rate $RATE \
            --output-file online.jsonl"
        
        echo "Executing: $CMD" | tee -a "$LOGFILE"
        eval "$CMD" 2>&1 | tee -a "$LOGFILE"
    done
done

# Stop profiling via API
# curl http://localhost:30000/stop_profile -H "Content-Type: application/json"

# Convert tracing file to csv & json
# sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
# python3 /sgl-workspace/rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json

