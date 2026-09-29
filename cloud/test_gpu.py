from beam import Image, function

image = Image(
    python_version="python3.10",
    python_packages=["torch"]
)

@function(
    gpu="RTX4090",
    image=image,
    memory="16Gi",
    cpu=4,
)
def ping_rtx4090():
    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
    vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_ok else 0.0
    return {
        "status": "success" if cuda_ok else "failed",
        "device": device_name,
        "vram_gb": vram,
        "cuda_version": torch.version.cuda
    }

if __name__ == "__main__":
    print("Testing RTX 4090 on Beam Cloud...")
    res = ping_rtx4090.remote()
    print("Result:", res)
