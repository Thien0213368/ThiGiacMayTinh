"""
confusion_matrix.py — Buổi 5: Trực quan hóa Confusion Matrix
BƯỚC 18: Hiển thị và phân tích ma trận nhầm lẫn

Mục tiêu: Hiểu lớp nào khó nhận diện nhất
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, OUTPUTS_DIR


def show_confusion_matrix(run_name: str = None, split: str = "val",
                           save_plot: bool = True):
    """
    BƯỚC 18: Hiển thị confusion matrix của model tốt nhất.

    Args:
        run_name: Tên run (vd: 'exp4_medium_best'). Nếu None thì tự tìm.
        split: 'val' hoặc 'test'
        save_plot: Có lưu ảnh không
    """
    # Tự động tìm confusion matrix đã có
    candidates = [
        PROJECT_ROOT / "runs" / "obb" / "val-10" / "confusion_matrix_normalized.png",
        PROJECT_ROOT / "runs" / "obb" / "val_cm"  / "confusion_matrix_normalized.png",
        PROJECT_ROOT / "runs" / "obb" / "test_cm" / "confusion_matrix_normalized.png",
    ]

    if run_name:
        candidates.insert(0,
            PROJECT_ROOT / "runs" / "obb" / run_name / "confusion_matrix_normalized.png"
        )

    cm_path = None
    for c in candidates:
        if c.exists():
            cm_path = str(c)
            break

    if cm_path:
        plt.figure(figsize=(10, 8))
        img = mpimg.imread(cm_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Normalized Confusion Matrix — Exp4 (YOLOv8m)', fontsize=14)
        plt.show()
        print(f"📊 Confusion matrix từ: {cm_path}")
    else:
        print("ℹ️  Chưa tìm thấy confusion matrix. Tạo mới...")
        generate_confusion_matrix(split=split)


def generate_confusion_matrix(model_path: str = None, split: str = "val",
                               name: str = "val_cm"):
    """
    Chạy validation để sinh ra confusion matrix plots.

    Args:
        model_path: Đường dẫn best.pt. Nếu None thì tự tìm.
        split: 'val' hoặc 'test'
        name: Tên thư mục lưu kết quả
    """
    from ultralytics import YOLO

    if model_path is None:
        # Tìm model tốt nhất theo thứ tự ưu tiên
        candidates = [
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp2_small"        / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if model_path is None:
        print("❌ Không tìm thấy model. Hãy train ít nhất 1 thực nghiệm trước!")
        return

    print(f"📊 Tạo confusion matrix từ: {model_path}")
    print(f"   Split: {split}, Lưu vào: runs/obb/{name}/")

    model = YOLO(model_path)
    results = model.val(
        data=str(DATASET_YAML),
        split=split,
        name=name,
        plots=True
    )

    # Hiển thị confusion matrix vừa tạo
    cm_path = PROJECT_ROOT / "runs" / "obb" / name / "confusion_matrix_normalized.png"
    if cm_path.exists():
        plt.figure(figsize=(12, 10))
        img = mpimg.imread(str(cm_path))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Normalized Confusion Matrix ({split.upper()} SET)', fontsize=16)
        plt.show()
        print(f"✅ Confusion matrix đã được lưu tại: {cm_path}")
    else:
        print(f"⚠️  Không tìm thấy file ma trận nhầm lẫn. Kiểm tra thư mục: runs/obb/{name}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hiển thị Confusion Matrix")
    parser.add_argument("--run", default=None, help="Tên run folder")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Tập dữ liệu đánh giá")
    parser.add_argument("--generate", action="store_true",
                        help="Tạo mới confusion matrix (chạy validation)")
    args = parser.parse_args()

    print("=" * 50)
    print("BƯỚC 18: Confusion Matrix")
    print("=" * 50)

    if args.generate:
        generate_confusion_matrix(split=args.split)
    else:
        show_confusion_matrix(run_name=args.run, split=args.split)
