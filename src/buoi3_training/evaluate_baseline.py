"""
evaluate_baseline.py — Buổi 3: Đánh giá mô hình Baseline
BƯỚC 9:  Đánh giá Baseline trên tập Validation
BƯỚC 10: Phân tích lỗi sơ bộ — xem mô hình dự đoán gì trên tập Test
"""

import os
import sys
from pathlib import Path
from glob import glob

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_DIR, DATASET_YAML, CONF_THRESHOLD, OUTPUTS_DIR


def evaluate_baseline(model_path: str = None, yaml_path: str = None) -> dict:
    """
    BƯỚC 9: Đánh giá Baseline trên tập Validation.

    Args:
        model_path: Đường dẫn tới best.pt của baseline
        yaml_path: Đường dẫn tới data_obb.yaml

    Returns:
        dict: {model, mAP50, mAP50_95, Precision, Recall}
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = str(PROJECT_ROOT / "runs" / "obb" / "baseline_nano" / "weights" / "best.pt")
    if yaml_path is None:
        yaml_path = str(DATASET_YAML)

    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        print("   Hãy chạy train_baseline.py trước!")
        return {}

    print(f"📊 Đánh giá Baseline: {model_path}")
    best_baseline = YOLO(model_path)
    metrics_baseline = best_baseline.val(data=yaml_path)

    results = {
        'model':     'YOLOv8n-OBB (Baseline)',
        'mAP50':     metrics_baseline.box.map50,
        'mAP50_95':  metrics_baseline.box.map,
        'Precision': metrics_baseline.box.mp,
        'Recall':    metrics_baseline.box.mr,
    }

    print("\n📋 KẾT QUẢ BASELINE:")
    print("-" * 40)
    for k, v in results.items():
        if k != 'model':
            print(f"  {k:12s}: {v:.4f}")
    print("-" * 40)
    print("  🎯 Đây là mốc chuẩn để so sánh với các model cải tiến")

    return results


def analyze_predictions(model_path: str = None, dataset_dir: str = None,
                         n_images: int = 4, save_path: str = None):
    """
    BƯỚC 10: Phân tích lỗi sơ bộ — xem mô hình dự đoán gì trên tập Test.

    Args:
        model_path: Đường dẫn tới best.pt
        dataset_dir: Đường dẫn dataset
        n_images: Số ảnh test hiển thị
        save_path: Đường dẫn lưu ảnh output
    """
    from ultralytics import YOLO

    if model_path is None:
        model_path = str(PROJECT_ROOT / "runs" / "obb" / "baseline_nano" / "weights" / "best.pt")
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return

    best_baseline = YOLO(model_path)
    test_images = glob(f"{dataset_dir}/test/images/*.jpg")[:n_images]

    if not test_images:
        print(f"❌ Không tìm thấy ảnh test trong {dataset_dir}/test/images/")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Baseline: Kết quả dự đoán trên tập Test',
                 fontsize=14, fontweight='bold')

    for ax, img_path in zip(axes.flatten(), test_images):
        results = best_baseline.predict(source=img_path,
                                        conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated_rgb)
        ax.set_title(
            f"{os.path.basename(img_path)}\n"
            f"Số object phát hiện: {len(results[0].obb)}",
            fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "baseline_predictions.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 9-10: Đánh giá Baseline & Phân tích lỗi")
    print("=" * 50)

    results = evaluate_baseline()
    if results:
        print()
        analyze_predictions()
