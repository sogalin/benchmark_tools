import os
import torch

# Settings
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
inputs = [128, 2048]
outputs = [128, 2048]

# Define the model paths
model_paths = [
    "amd/Llama-3.1-8B-Instruct-FP8-KV",
    "amd/Llama-3.1-70B-Instruct-FP8-KV",
]

# Flag to enable or disable --attention-backend aiter
enable_attention_backend_aiter = True  # Set to False to disable the option

# Set environment variables
os.environ["RCCL_MSCCL_ENABLE"] = "0"
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
os.environ["NCCL_MIN_NCHANNELS"] = "112"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["MOE_PADDING"] = "1"
os.environ["VLLM_FP8_PADDING"] = "1"
os.environ["VLLM_FP8_ACT_PADDING"] = "1"
os.environ["VLLM_FP8_WEIGHT_PADDING"] = "1"
os.environ["VLLM_FP8_REDUCE_CONV"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"] = "1"
os.environ["SGLANG_SET_CPU_AFFINITY"] = "1"

# Get the GPU name and format it for file naming
gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')

for model_path in model_paths:
    # Extract the model name from the model path
    model_name = model_path.split('/')[-1]  # Get the last part of the path as the model name

    # Set TP values based on the model type
    tp_values = [1] if model_name == "Llama-3.1-8B-Instruct-FP8-KV" else [1, 8]

    for tp in tp_values:
        for batch_size in batch_sizes:
            for input_size in inputs:
                for output_size in outputs:
                    # Generate the output filename with GPU name, model name, and parameters
                    output_file = f"{gpu_name}_{model_name}_tp{tp}_result_bs{batch_size}_in{input_size}_out{output_size}.txt"

                    # Construct the base command
                    command = (
                        f"python -m sglang.bench_one_batch "
                        f"--model {model_path} "
                        f"--tp {tp} "
                        f"--batch-size {batch_size} "
                        f"--input {input_size} "
                        f"--output {output_size} "
                        f"--quantization fp8 "
                    )

                    # Append --attention-backend aiter if the flag is enabled
                    if enable_attention_backend_aiter:
                        command += "--attention-backend aiter --mem-fraction-static 0.6"

                    # Write the command and its output to the file
                    full_command = f'echo "{command}" | tee -a {output_file} && {command} | tee -a {output_file}'

                    # Execute the command
                    os.system(full_command)
                    print(f"Executed and logged: {full_command}")

