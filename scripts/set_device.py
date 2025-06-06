import subprocess
import os
def set_device():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        )
        memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        gpu_id = int(min(range(len(memory_used)), key=lambda i: memory_used[i]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    except Exception as e:
        print("Could not set CUDA_VISIBLE_DEVICES, defaulting to CPU. Reason:", e)
        gpu_id = None
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = set_device()
    print(f"Device set to: {device}")
    
if __name__ == "__main__":
# Set the most unused GPU as default
    main()