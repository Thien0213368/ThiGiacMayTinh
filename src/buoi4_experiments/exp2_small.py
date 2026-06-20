"""
exp2_small.py — Buổi 4: Thực nghiệm 2 — YOLOv8s-OBB (Small)
BƯỚC 12: Model lớn hơn → nhiều tham số hơn → hiểu sâu hơn

So sánh tham số: Nano=3.1M | Small=11.4M
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, TRAIN_DEFAULTS, EXPERIMENTS


def train_exp2_small(yaml_path: str = None, epochs: int = None,
                     device: str = None) -> object:
    """
    BƯỚC 12: Thực nghiệm 2 — YOLOv8s-OBB (Small).

    Ý tưởng: Model lớn hơn → nhiều tham số hơn → hiểu sâu hơn về đặc trưng.
    Thay đổi so với baseline: model yolov8s thay vì yolov8n.

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
        epochs = EXPERIMENTS["exp2_small"]["epochs"]
    if device is None:
        device = TRAIN_DEFAULTS["device"]

    cfg = EXPERIMENTS["exp2_small"]

    print("🚀 Thực nghiệm 2: YOLOv8s-OBB (Small)")
    print("   So sánh tham số: Nano=3.1M | Small=11.4M")
    print(f"   Model  : {cfg['model']}")
    print(f"   Epochs : {epochs}")
    print("-" * 50)

    model = YOLO(cfg["model"])

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=TRAIN_DEFAULTS["imgsz"],
        batch=TRAIN_DEFAULTS["batch"],
        lr0=0.01,        # learning rate ban đầu
        lrf=0.01,        # learning rate cuối (lr0 * lrf)
        patience=15,     # dừng sớm nếu 15 epoch không cải thiện
        device=device,
        name=cfg["name"],
    )

    print("\n✅ Thực nghiệm 2 xong!")
    weights_path = PROJECT_ROOT / "runs" / "obb" / cfg["name"] / "weights" / "best.pt"
    print(f"   Weights: {weights_path}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 12: Thực nghiệm 2 — YOLOv8s-OBB (Small)")
    print("=" * 50)
    train_exp2_small()
