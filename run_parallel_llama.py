import os
import torch
import multiprocessing
from queue import Queue

# Settings
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
inputs = [128, 2048]
outputs = [128, 2048]

# Define the model paths
model_paths = [
    "amd/Llama-3.1-70B-Instruct-FP8-KV",
]

# Get available GPUs
gpu_count = torch.cuda.device_count()
if gpu_count < 8:
    print(f"Warning: Less than 8 GPUs detected ({gpu_count} found). Adjusting instance count accordingly.")

# Generate test cases
test_queue = Queue()
for model_path in model_paths:
    tp_values = [1]  # Can be modified if needed
    for tp in tp_values:
        for batch_size in batch_sizes:
            for input_size in inputs:
                for output_size in outputs:
                    test_queue.put((model_path, tp, batch_size, input_size, output_size))

# Function to run a single test instance
def run_test(gpu_id):
    while not test_queue.empty():
        model_path, tp, batch_size, input_size, output_size = test_queue.get()
        model_name = model_path.split('/')[-1]
        gpu_name = torch.cuda.get_device_name(gpu_id).replace(' ', '_')
        output_file = f"{gpu_name}_GPU{gpu_id}_{model_name}_tp{tp}_result_bs{batch_size}_in{input_size}_out{output_size}.txt"
        
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python -m sglang.bench_one_batch "
            f"--model {model_path} "
            f"--tp {tp} "
            f"--batch-size {batch_size} "
            f"--input {input_size} "
            f"--output {output_size} "
            f"--quantization fp8 "
        )
        
        full_command = f'echo "{command}" | tee -a {output_file} && {command} | tee -a {output_file}'
        os.system(full_command)
        print(f"Executed and logged on GPU {gpu_id}: {full_command}")

# Run tests in parallel
processes = []
for i in range(min(8, gpu_count)):
    p = multiprocessing.Process(target=run_test, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

