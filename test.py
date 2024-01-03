import torch

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()

# Display information about available GPUs
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch will run on CPU.")

