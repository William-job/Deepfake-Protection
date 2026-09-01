import torch
import sys


def main():
    print("=" * 60)
    print("CAR Deepfake Detection - GPU & PyTorch Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print("")

    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print("")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   [GPU {i}] {props.name}")
            print(f"         Compute capability: {props.major}.{props.minor}")
            print(f"         Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"         Multi-processor count: {props.multi_processor_count}")
            print("")

        # Perform a quick GPU test
        print("Performing quick GPU test...")
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print(f"✅ GPU operation successful: {z.sum().item():.4f}")
        print("")

        print("Everything looks good! You're ready to train with GPU.")
    else:
        print("❌ CUDA is NOT available.")
        print("")
        print("Possible reasons:")
        print("  1. PyTorch was installed without CUDA support")
        print("  2. NVIDIA GPU drivers are not installed or outdated")
        print("  3. CUDA toolkit is not installed")
        print("")
        print("To install PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  (or cu118 depending on your CUDA version)")
        print("")
        print("Or, visit https://pytorch.org/get-started/locally/")
        sys.exit(1)


if __name__ == "__main__":
    main()
