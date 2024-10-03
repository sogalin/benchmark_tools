import os

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
inputs = [128, 2048]
outputs = [128, 2048]

#batch_sizes = [1]
#inputs = [1024]
#outputs = [8]

for batch_size in batch_sizes:
    for input_size in inputs:
        for output_size in outputs:
            output_file = f"llama3.1-8b-result_bs{batch_size}_in{input_size}_out{output_size}.txt"

            command = (
                f"HIP_FORCE_DEV_KERNARG=1 python -m sglang.bench_latency "
           #     f"--model amd/Meta-Llama-3.1-8B-Instruct-FP8-KV "
                f"--model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 "
                f"--tp 1 "
                f"--batch-size {batch_size} "
                f"--input {input_size} "
                f"--output {output_size} "
           #     f"--quant fp8 "
                f"--disable-cuda-graph "
           #     f"--enable-prefill-prof "
                f"> {output_file}"
            )

            os.system(command)
            print(f"Executed: {command}")


