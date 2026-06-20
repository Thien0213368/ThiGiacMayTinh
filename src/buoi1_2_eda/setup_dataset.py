"""
setup_dataset.py — Buổi 1-2: Cài đặt môi trường & giải nén dataset
BƯỚC 0: Cài thư viện
BƯỚC 1: Import thư viện
BƯỚC 2: Giải nén dataset
"""

import os
import zipfile
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_DIR


def install_dependencies():
    """BƯỚC 0: Cài đặt các thư viện cần thiết."""
    print("📦 Cài đặt thư viện...")
    print("   Chạy: pip install -r requirements.txt")
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        os.system(f'pip install -r "{req_file}" -q')
        print("✅ Cài đặt thư viện thành công!")
    else:
        print("⚠️  Không tìm thấy requirements.txt — hãy cài thủ công:")
        print("   pip install ultralytics pyyaml opencv-python matplotlib")


def import_libraries():
    """BƯỚC 1: Kiểm tra import thư viện."""
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from glob import glob
        from collections import Counter
        import yaml
        print("✅ Import thư viện thành công!")
        return True
    except ImportError as e:
        print(f"❌ Thiếu thư viện: {e}")
        print("   Chạy: pip install -r requirements.txt")
        return False


def setup_dataset(zip_file: str = None):
    """
    BƯỚC 2: Giải nén dataset.

    Args:
        zip_file: Đường dẫn tới file .zip dataset.
                  Mặc định tìm trong PROJECT_ROOT.
    """
    # Tự động tìm file zip nếu không truyền vào
    if zip_file is None:
        candidates = list(PROJECT_ROOT.glob("*.zip")) + list(PROJECT_ROOT.glob("**/*.zip"))
        candidates = [f for f in candidates if "yolov8-obb" in f.name or "ThiGiac" in f.name]
        if candidates:
            zip_file = str(candidates[0])
        else:
            print("ℹ️  Không tìm thấy file .zip dataset")
            print(f"   Dataset hiện tại: {DATASET_DIR}")

    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"📂 Dataset đã tồn tại tại: {DATASET_DIR}")
        return True

    if zip_file and Path(zip_file).exists():
        print(f"📦 Giải nén: {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(str(DATASET_DIR))
        print(f"✅ Giải nén xong vào: {DATASET_DIR}")
        return True
    else:
        print(f"❌ Không tìm thấy file zip. Dataset cần ở: {DATASET_DIR}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 0-2: Setup môi trường & Dataset")
    print("=" * 50)

    install_dependencies()
    print()
    import_libraries()
    print()
    setup_dataset()
