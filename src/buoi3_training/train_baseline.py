"""
train_baseline.py — Buổi 3: Train mô hình Baseline YOLOv8n-OBB
BƯỚC 8: Train Baseline — YOLOv8n-OBB (Nano)

Chiến lược:
  - Dùng mô hình nhỏ nhất (nano) để train nhanh
  - Xác nhận pipeline hoạt động đúng
  - Ghi nhận loss/mAP ban đầu làm mốc so sánh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, TRAIN_DEFAULTS, EXPERIMENTS


def train_baseline(yaml_path: str = None, epochs: int = None,
                   device: str = None) -> object:
    """
    BƯỚC 8: Train Baseline YOLOv8n-OBB.

    Args:
        yaml_path: Đường dẫn tới data_obb.yaml
        epochs: Số epochs (mặc định 100 theo notebook)
        device: '0' cho GPU, 'cpu' cho CPU

    Returns:
        results: Kết quả training từ YOLO
    """
    from ultralytics import YOLO

    if yaml_path is None:
        yaml_path = str(DATASET_YAML)
    if epochs is None:
        epochs = EXPERIMENTS["baseline_nano"]["epochs"]
    if device is None:
        device = TRAIN_DEFAULTS["device"]

    cfg = EXPERIMENTS["baseline_nano"]

    print("🚀 Bắt đầu train Baseline (YOLOv8n-OBB)...")
    print(f"   Thời gian ước tính: ~5-10 phút (20 epochs trên GPU)")
    print(f"   Model  : {cfg['model']}")
    print(f"   Epochs : {epochs}")
    print(f"   Device : {device}")
    print("-" * 50)

    # Tải model pretrained (đã học từ COCO dataset)
    model = YOLO(cfg["model"])

    # Train
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=TRAIN_DEFAULTS["batch"],
        device=device,
        name=cfg["name"],
        verbose=TRAIN_DEFAULTS["verbose"],
    )

    print("\n✅ Train Baseline xong!")
    weights_path = PROJECT_ROOT / "runs" / "obb" / cfg["name"] / "weights" / "best.pt"
    print(f"   Weights: {weights_path}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 8: Train Baseline — YOLOv8n-OBB")
    print("=" * 50)
    train_baseline()
