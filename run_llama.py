import os
import torch

# Settings
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
inputs = [128, 2048]
outputs = [128, 2048]

#batch_sizes = [1]
#inputs = [128]
#outputs = [128]

# Define the model paths
model_paths = [
#    "amd/Meta-Llama-3.1-8B-Instruct-FP8-KV",
    "amd/Llama-3.1-70B-Instruct-FP8-KV",
#    "amd/Meta-Llama-3.1-405B-Instruct-FP8-KV"
]

#os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
#os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
#os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "/tunableop-config.csv"
#os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
#os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "1"
#os.environ["PYTORCH_TUNABLEOP_ROCBLAS_ENABLED"] = "1"

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

# Get the GPU name
gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')

for model_path in model_paths:
    # Extract model name from the model path
    model_name = model_path.split('/')[-1]  # Get the full model name from the path

    # Determine the TP values based on the model
#    tp_values = [8] if model_name == "Meta-Llama-3.1-405B-Instruct-FP8-KV" else [1]
    tp_values = [1]

    for tp in tp_values:
        for batch_size in batch_sizes:
            for input_size in inputs:
                for output_size in outputs:
                    # Include GPU name and model name in output file
                    output_file = f"{gpu_name}_{model_name}_tp{tp}_result_bs{batch_size}_in{input_size}_out{output_size}.txt"

                    # Construct the command to execute
                    command = (
                        f"python -m sglang.bench_one_batch "
                        f"--model {model_path} "
                        f"--tp {tp} "
                        f"--batch-size {batch_size} "
                        f"--input {input_size} "
                        f"--output {output_size} "
                        f"--quantization fp8 "
#                        f"--enable-torch-compile "
#                        f"--disable-cuda-graph "
#                        f"--enable-decode-prof "
                    )

                    # Command to write both the executed command and the output to the output_file
                    full_command = f'echo "{command}" | tee -a {output_file} && {command} | tee -a {output_file}'

                    # Execute the command
                    os.system(full_command)
                    print(f"Executed and logged: {full_command}")
