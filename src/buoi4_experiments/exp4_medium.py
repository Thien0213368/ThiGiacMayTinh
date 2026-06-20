"""
exp4_medium.py — Buổi 4: Thực nghiệm 4 — YOLOv8m-OBB (Medium) — BEST
BƯỚC 14: Model lớn + Augmentation + Train lâu → kết quả tốt nhất

Tham số: Medium=26.4M | Thời gian: ~30-60 phút
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, TRAIN_DEFAULTS, EXPERIMENTS


def train_exp4_medium(yaml_path: str = None, epochs: int = None,
                       device: str = None) -> object:
    """
    BƯỚC 14: Thực nghiệm 4 — YOLOv8m-OBB (Medium) — BEST.

    Kết hợp model lớn + augmentation → best of both worlds.
    - 26.4M params → học được đặc trưng chi tiết hơn
    - Augmentation: Mosaic + Mixup + Rotation + HSV
    - Train 30 epochs với early stopping patience=20

    Args:
        yaml_path: Đường dẫn data_obb.yaml
        epochs: Số epochs (mặc định 30)
        device: '0' GPU hoặc 'cpu'

    Returns:
        results: Kết quả training
    """
    from ultralytics import YOLO

    if yaml_path is None:
        yaml_path = str(DATASET_YAML)
    if epochs is None:
        epochs = EXPERIMENTS["exp4_medium_best"]["epochs"]
    if device is None:
        device = TRAIN_DEFAULTS["device"]

    cfg = EXPERIMENTS["exp4_medium_best"]

    print("🚀 Thực nghiệm 4: YOLOv8m-OBB (Medium) — Cố gắng đạt mục tiêu")
    print(f"   Tham số: Medium=26.4M | Thời gian: ~30-60 phút")
    print(f"   Model  : {cfg['model']}")
    print(f"   Epochs : {epochs}")
    print("-" * 50)

    model = YOLO(cfg["model"])

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=TRAIN_DEFAULTS["batch"],
        lr0=0.01,
        lrf=0.001,
        patience=20,
        device=device,
        name=cfg["name"],
        # Augmentation (giống Exp3 nhưng áp dụng cho model lớn hơn)
        mosaic=1.0,
        mixup=0.2,
        degrees=15.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("\n✅ Thực nghiệm 4 xong!")
    weights_path = PROJECT_ROOT / "runs" / "obb" / cfg["name"] / "weights" / "best.pt"
    print(f"   Weights: {weights_path}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 14: Thực nghiệm 4 — YOLOv8m-OBB (Medium)")
    print("=" * 50)
    train_exp4_medium()
