import subprocess
import os
import torch
def get_least_used_gpu():
    # Query GPU memory usage
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    # Parse the result
    memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    # Get the index of the GPU with the least memory used
    return int(min(range(len(memory_used)), key=lambda i: memory_used[i]))


if __name__ == "__main__":
# Set the most unused GPU as default
    gpu_id = get_least_used_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Now torch and transformers will use the selected GPU by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {gpu_id}")