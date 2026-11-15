"""Check Python / PyTorch / CUDA / required libraries. Read-only."""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "numpy", "pandas", "yaml", "tqdm", "nibabel", "SimpleITK", "cv2", "sklearn",
    "matplotlib", "timm", "ultralytics", "pytorch_grad_cam", "streamlit", "pytest",
]


def main() -> int:
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform    : {platform.platform()}")

    ok = True
    try:
        import torch
        import torchvision

        print(f"torch       : {torch.__version__} (CUDA build: {torch.version.cuda})")
        print(f"torchvision : {torchvision.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda_available: {cuda}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU         : {props.name}, VRAM {props.total_memory / 1024**3:.2f} GB, "
                  f"capability {props.major}.{props.minor}")
            print(f"cuDNN       : {torch.backends.cudnn.version()}")
            x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
            _ = (x @ x).sum().item()
            print("CUDA fp16 matmul: OK")
        else:
            ok = False
    except ImportError as e:
        print(f"torch       : MISSING ({e})")
        ok = False

    print("\nLibraries:")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]      {name:18s} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001 - report any import failure
            print(f"  [MISSING] {name:18s} {type(e).__name__}: {e}")
            ok = False

    print("\nRESULT:", "READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
