import os
import datetime
from itertools import product

# Mapping table for model names and their corresponding paths
MODEL_PATHS = {
    "Meta-Llama-3.1-405B-Instruct-FP8-KV": "/data/huggingface/hub/models--amd--Meta-Llama-3.1-405B-Instruct-FP8-KV/snapshots/2ff6250f60a75b879fc6c05830c67e37171a22ee/",
    # Add more models here if needed
}

# Parameterized settings
model_name = "Meta-Llama-3.1-405B-Instruct-FP8-KV"  # Model name
random_inputs = [128, 1024, 2048]  # Values for --random-input
random_outputs = [128, 1024, 2048]  # Values for --random-output
num_prompts_list = [1000, 2000, 3000]  # Values for --num-prompts
random_range_ratios = [0, 1]  # Values for --random-range-ratio
tp = 8  # Value for --tp
quant = "fp8"  # Value for --quant
context_len = 4096  # Value for --context-len
chunked_prefill_sizes = [4096, 8192]  # Values for --chunked-prefill-size

# Verify if the model name exists in the mapping table
if model_name not in MODEL_PATHS:
    raise ValueError(f"Model name '{model_name}' not found in MODEL_PATHS mapping table.")

# Retrieve the corresponding model path
model_path = MODEL_PATHS[model_name]

# Generate all combinations of parameters
param_combinations = list(product(
    random_inputs, random_outputs, num_prompts_list, chunked_prefill_sizes, random_range_ratios
))

# Run tests and display progress
total_tests = len(param_combinations)
for idx, (random_input, random_output, num_prompts, chunked_prefill_size, random_range_ratio) in enumerate(param_combinations, start=1):
    # Construct the command
    command = (
        f"python -m sglang.bench_offline_throughput "
        f"--model-path {model_path} "
        f"--dataset-name random "
        f"--random-input {random_input} "
        f"--random-output {random_output} "
        f"--quant {quant} "
        f"--tp {tp} "
        f"--num-prompts {num_prompts} "
        f"--context-len {context_len} "
        f"--chunked-prefill-size {chunked_prefill_size} "
        f"--random-range-ratio {random_range_ratio}"
    )

    # Generate the log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = (
        f"benchmark_{model_name}_tp{tp}_input{random_input}_output{random_output}_prompts{num_prompts}_"
        f"context{context_len}_chunk{chunked_prefill_size}_range{random_range_ratio}_{timestamp}.log"
    )

    # Display progress
    print(f"[{idx}/{total_tests}] Running test: random_input={random_input}, random_output={random_output}, "
          f"num_prompts={num_prompts}, chunked_prefill_size={chunked_prefill_size}, random_range_ratio={random_range_ratio}")
    print(f"Executing command:\n{command}")

    # Execute the command and redirect output to the log file
    os.system(f"{command} > {log_file_name} 2>&1")
    print(f"Log written to: {log_file_name}")

print("All tests completed.")

