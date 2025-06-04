import os
import subprocess
import pandas as pd
import argparse

# Ensure the script exits on the first error encountered
def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {command}")

# Define environment variable parameters
RCCL_MSCCL_ENABLE = '0'
SGLANG_AITER_MOE = '1'
SGLANG_INT4_WEIGHT = '1'
MOE_PADDING = '0'

# Define common parameters
BATCH_SIZE = 1
OUTPUT = 10
MODEL_PATH = "/data/grok-1-W4A8KV8/"
TOKENIZER_PATH = "Xenova/grok-1-tokenizer"
TP = 8
QUANTIZATION = "fp8"
ATTENTION_BACKEND = "aiter"

# Define an array of default input sizes
DEFAULT_INPUT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

def run_benchmark(input_sizes, num_runs):
    # Dictionary to collect all Ave_us values per input size
    results = {size: [] for size in input_sizes}

    for INPUT_SIZE in input_sizes:
        for run in range(num_runs):
            # Define the benchmark command with environment variables
            CMD = (
                f"RCCL_MSCCL_ENABLE={RCCL_MSCCL_ENABLE} "
                f"SGLANG_AITER_MOE={SGLANG_AITER_MOE} "
                f"SGLANG_INT4_WEIGHT={SGLANG_INT4_WEIGHT} "
                f"MOE_PADDING={MOE_PADDING} "
                f"python -m sglang.bench_one_batch --batch-size {BATCH_SIZE} --input {INPUT_SIZE} "
                f"--output {OUTPUT} --model {MODEL_PATH} --tokenizer-path {TOKENIZER_PATH} --tp {TP} "
                f"--quantization {QUANTIZATION} --trust-remote-code --attention-backend {ATTENTION_BACKEND} "
                f"--mem-fraction-static 0.2 --enable-prefill-prof"
            )

            # Print the command to be executed
            print(f"Running test case {run+1}/{num_runs} for input size {INPUT_SIZE}: {CMD}")

            # Execute the benchmark command
            run_command(CMD)

            # Move the trace file to a new name based on input size and run number
            trace_file = f"trace-{INPUT_SIZE}-run-{run+1}.rpd"
            os.rename("trace.rpd", trace_file)

            # Define the sqlite command
            SQLITE_CMD = (
                f"sqlite3 {trace_file} '.mode csv' '.header on' '.output trace-{INPUT_SIZE}-run-{run+1}.csv' "
                f"'select * from top;' '.output stdout'"
            )

            # Print the sqlite command to be executed
            print(f"Executing: sqlite3 processing for {trace_file}")

            # Execute the sqlite command
            run_command(SQLITE_CMD)

            # Load the generated CSV file to filter and extract the required data
            csv_file = f"trace-{INPUT_SIZE}-run-{run+1}.csv"
            df = pd.read_csv(csv_file)

            # Filter Name column containing keyword
            filtered_df = df[df['Name'].str.contains('FmhaBatchPrefillWithPagedKVCacheKernel', na=False)]

            # Append Ave_us column data to the list for this input size
            ave_us_values = filtered_df['Ave_us'].tolist()
            results[INPUT_SIZE].extend(ave_us_values)

            # Print the Ave_us values for each run
            print(f"Ave_us values for input size {INPUT_SIZE}, run {run+1}: {ave_us_values}")

    # Print all collected Ave_us values and their averages
    for size, values in results.items():
        average_ave_us = sum(values) / len(values) if values else 0
        print(f"Input size {size}: Ave_us values = {values}, Average = {average_ave_us}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark with specified input size.")
    parser.add_argument("--input", type=int, help="Specify a specific input token length.")
    parser.add_argument("--runs", type=int, default=10, help="Number of times to run each test case.")

    args = parser.parse_args()

    # Use the specified input size if provided, otherwise use the default sizes
    input_sizes = [args.input] if args.input else DEFAULT_INPUT_SIZES
    run_benchmark(input_sizes, args.runs)

