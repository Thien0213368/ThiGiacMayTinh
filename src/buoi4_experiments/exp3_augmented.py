"""
exp3_augmented.py — Buổi 4: Thực nghiệm 3 — YOLOv8n + Augmentation mạnh
BƯỚC 13: Dữ liệu ít → augmentation giúp model robust hơn

Augmentation được thêm vào:
  - Mosaic: ghép 4 ảnh → model thấy đối tượng trong nhiều bối cảnh
  - Mixup: trộn 2 ảnh → giảm overconfidence
  - Rotation ±15°: phù hợp với OBB (đối tượng nằm nghiêng)
  - Flip trái/phải, trên/dưới
  - HSV color jitter
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, TRAIN_DEFAULTS, EXPERIMENTS


def train_exp3_augmented(yaml_path: str = None, epochs: int = None,
                          device: str = None) -> object:
    """
    BƯỚC 13: Thực nghiệm 3 — YOLOv8n + Augmentation mạnh.

    Ý tưởng: Cùng model nhỏ nhưng augmentation mạnh để giảm overfitting.

    Args:
        yaml_path: Đường dẫn data_obb.yaml
        epochs: Số epochs (mặc định 50)
        device: '0' GPU hoặc 'cpu'

    Returns:
        results: Kết quả training
    """
    from ultralytics import YOLO

    if yaml_path is None:
        yaml_path = str(DATASET_YAML)
    if epochs is None:
        epochs = EXPERIMENTS["exp3_augmented"]["epochs"]
    if device is None:
        device = TRAIN_DEFAULTS["device"]

    cfg = EXPERIMENTS["exp3_augmented"]

    print("🚀 Thực nghiệm 3: YOLOv8n + Augmentation")
    print("   Augmentation: Mosaic + Mixup + Rotation + Flip + HSV")
    print("-" * 50)

    model = YOLO(cfg["model"])

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=TRAIN_DEFAULTS["batch"],
        device=device,
        name=cfg["name"],
        # === CÁC AUGMENTATION THÊM VÀO ===
        mosaic=1.0,      # ghép 4 ảnh thành 1 → học object trong nhiều bối cảnh
        mixup=0.2,       # trộn 2 ảnh lại → giảm overconfidence
        degrees=15.0,    # xoay ảnh ngẫu nhiên ±15°
        fliplr=0.5,      # lật trái/phải với xác suất 50%
        flipud=0.1,      # lật trên/dưới với xác suất 10%
        hsv_h=0.015,     # biến đổi màu Hue
        hsv_s=0.7,       # biến đổi Saturation
        hsv_v=0.4,       # biến đổi Value/Brightness
        translate=0.1,   # dịch chuyển ảnh ±10%
        scale=0.5,       # zoom in/out ngẫu nhiên
    )

    print("\n✅ Thực nghiệm 3 xong!")
    weights_path = PROJECT_ROOT / "runs" / "obb" / cfg["name"] / "weights" / "best.pt"
    print(f"   Weights: {weights_path}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 13: Thực nghiệm 3 — YOLOv8n + Augmentation")
    print("=" * 50)
    train_exp3_augmented()
