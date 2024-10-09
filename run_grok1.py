import os
import torch

# Settings
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
inputs = [128, 2048]
outputs = [128, 2048]
tp_values = [8, 4]  # Default TP values

# Define the model path
model_path = "/sglang/dummy_grok1/"
# Extract model name from the model path
model_name = model_path.split('/')[-2]

# Get the GPU name
gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')

for tp in tp_values:
    # Create directory for each tp value
    tp_directory = f"tp_{tp}_results"
    if not os.path.exists(tp_directory):
        os.makedirs(tp_directory)

    for batch_size in batch_sizes:
        for input_size in inputs:
            for output_size in outputs:
                # Include GPU name, model name, and tp value in the output file
                output_file = f"{tp_directory}/{gpu_name}_{model_name}_tp{tp}_bs{batch_size}_in{input_size}_out{output_size}.txt"

                # Construct the command to execute
                command = (
                    f"HIP_FORCE_DEV_KERNARG=1 python -m sglang.bench_latency "
                    f"--model {model_path} "
                    f"--load-format dummy "
                    f"--tokenizer-path Xenova/grok-1-tokenizer "
                    f"--tp {tp} "
                    f"--batch-size {batch_size} "
                    f"--input {input_size} "
                    f"--output {output_size} "
                    f"--quant fp8 "
                )

                # Command to write both the executed command and all output (stdout and stderr) to the output_file
                full_command = f'echo "{command}" | tee -a {output_file} && {command} 2>&1 | tee -a {output_file}'

                # Execute the command
                os.system(full_command)
                print(f"Executed and logged: {full_command}")

