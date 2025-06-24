import os
import torch

# Settings
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#inputs = [128, 2048]
#outputs = [128, 2048]
inputs = [256]
outputs = [2048]
tp_values = [8]  # Default TP values

# Define the model path
#model_path = "/data/models/huggingface/amd--grok-1-W4A8KV8"
model_path = "/data/grok-1-W4A8KV8"
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
                    f"RCCL_MSCCL_ENABLE=0 SGLANG_AITER_MOE=1 SGLANG_INT4_WEIGHT=1 "
                    f"python -m sglang.bench_one_batch "
                    f"--batch-size {batch_size} "
                    f"--input {input_size} "
                    f"--output {output_size} "
                    f"--model {model_path} "
                    f"--tokenizer-path Xenova/grok-1-tokenizer "
                    f"--tp {tp} "
                    f"--quantization fp8 "
                    f"--trust-remote-code "
                    f"--attention-backend aiter"
                )

                # Command to write both the executed command and all output (stdout and stderr) to the output_file
                full_command = f'echo "{command}" | tee -a {output_file} && {command} 2>&1 | tee -a {output_file}'

                # Execute the command
                os.system(full_command)
                print(f"Executed and logged: {full_command}")

