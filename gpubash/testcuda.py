import torch

def check_cuda_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Built with CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Device capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
    else:
        print("❌ CUDA is NOT available.")

if __name__ == "__main__":
    check_cuda_info()
