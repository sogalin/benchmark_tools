#!/bin/bash  
  
# Start profiling via API  
# curl http://localhost:30000/start_profile -H "Content-Type: application/json"  
  
# Define the log file with a timestamp  
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  
  
# List of request rates  
REQUEST_RATES=(1 2 4 8 16 32 64)  
REQUEST_RATES=(64)  
  
# Possible maximum number of prompts  
MAX_NUM_PROMPTS_OPTIONS=(2400 4800 9600)  
  
# File to track completed combinations  
COMPLETED_FILE="completed_combinations.log"  
  
# Ensure the file exists  
touch "$COMPLETED_FILE"  
  
# Loop through each request rate  
for RATE in "${REQUEST_RATES[@]}"; do  
    for MAX_NUM_PROMPTS in "${MAX_NUM_PROMPTS_OPTIONS[@]}"; do  
        for i in {1..3}; do  # Run each rate with each max prompts 3 times  
            NUM_PROMPTS=$(( 300 * RATE ))  
            if [ "$NUM_PROMPTS" -gt "$MAX_NUM_PROMPTS" ]; then  
                NUM_PROMPTS="$MAX_NUM_PROMPTS"  
            fi  
  
            # Check if this combination has been run  
            COMBINATION="${RATE}_${NUM_PROMPTS}_run${i}"  
            if grep -q "$COMBINATION" "$COMPLETED_FILE"; then  
                echo "Skipping already completed combination: $COMBINATION"  
                continue  
            fi  
  
            LOGFILE="sglang_client_log_grok1_${RATE}_max${MAX_NUM_PROMPTS}_run${i}_$TIMESTAMP.log"  
            echo "Running benchmark with request rate: $RATE, max num_prompts: $MAX_NUM_PROMPTS, actual num_prompts: $NUM_PROMPTS (Run $i)" | tee -a "$LOGFILE"  
  
            CMD="python3 -m sglang.bench_serving --backend sglang --tokenizer Xenova/grok-1-tokenizer --dataset-name random --random-input 1024 --random-output 1024 --num-prompts $NUM_PROMPTS --request-rate $RATE --output-file online.jsonl"  
  
            echo "Executing: $CMD" | tee -a "$LOGFILE"  
            eval "$CMD" 2>&1 | tee -a "$LOGFILE"  
  
            # Record the completed combination  
            echo "$COMBINATION" >> "$COMPLETED_FILE"  
  
            # Wait for 3 seconds between tests  
            sleep 3  
        done  
    done  
done  
  
# Stop profiling via API  
# curl http://localhost:30000/stop_profile -H "Content-Type: application/json"  
  
# Convert tracing file to csv & json  
# sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"  
# python3 /sgl-workspace/rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json  

