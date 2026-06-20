"""
predict_test.py — Buổi 5: Dự đoán thử nghiệm trên các ảnh khó ở tập Test
BƯỚC 19: Chạy inference với model tốt nhất, hiển thị kết quả

Bám sát notebook BƯỚC 19:
  model_best.predict(source=img_path, conf=0.3, verbose=False)
  res_plotted = results[0].plot()
  res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
"""

import os
import sys
from pathlib import Path
from glob import glob

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_DIR, CONF_THRESHOLD, OUTPUTS_DIR


def predict_on_test_images(model_path: str = None, dataset_dir: str = None,
                            n_images: int = 8, conf: float = None,
                            save_path: str = None):
    """
    BƯỚC 19: Dự đoán OBB với model tốt nhất trên tập Test.

    Args:
        model_path: Đường dẫn best.pt
        dataset_dir: Đường dẫn dataset
        n_images: Số ảnh hiển thị (default 8)
        conf: Confidence threshold (default 0.3)
        save_path: Đường dẫn lưu ảnh output
    """
    from ultralytics import YOLO

    if conf is None:
        conf = CONF_THRESHOLD
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    # Tìm model tốt nhất
    if model_path is None:
        candidates = [
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp2_small"        / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp3_augmented"    / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "baseline_nano"     / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if model_path is None:
        print("❌ Không tìm thấy model. Hãy train ít nhất 1 thực nghiệm trước!")
        return

    # Lấy danh sách ảnh test
    test_images = glob(f"{dataset_dir}/test/images/*.jpg")[:n_images]
    if not test_images:
        print(f"❌ Không tìm thấy ảnh test trong {dataset_dir}/test/images/")
        return

    print(f"🔍 Dự đoán {len(test_images)} ảnh từ tập Test")
    print(f"   Model: {model_path}")
    print(f"   Conf : {conf}")

    model_best = YOLO(model_path)

    cols = 2
    rows = (len(test_images) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    fig.suptitle('Dự đoán OBB với mô hình tốt nhất (Exp4)',
                 fontsize=16, fontweight='bold')

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, img_path in zip(axes_flat, test_images):
        # Dự đoán với threshold thấp để xem các vùng nghi ngờ
        results = model_best.predict(source=img_path, conf=conf, verbose=False)

        # Vẽ kết quả (bám sát notebook BƯỚC 19)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        ax.imshow(res_rgb)
        ax.set_title(f"File: {os.path.basename(img_path)}", fontsize=10)
        ax.axis('off')

    # Ẩn axes thừa
    for j in range(len(test_images), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "test_predictions.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu: {save_path}")


def predict_single(image_path: str, model_path: str = None,
                   conf: float = None, show: bool = True):
    """
    Dự đoán OBB trên 1 ảnh đơn lẻ.

    Args:
        image_path: Đường dẫn ảnh đầu vào
        model_path: Đường dẫn model weights
        conf: Confidence threshold
        show: Có hiển thị không

    Returns:
        results: YOLO results object
    """
    from ultralytics import YOLO

    if conf is None:
        conf = CONF_THRESHOLD
    if model_path is None:
        candidates = [
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if model_path is None:
        print("❌ Không tìm thấy model!")
        return None

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    if show and results:
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_rgb)
        plt.title(f"Kết quả: {len(results[0].obb)} objects phát hiện")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"✅ Phát hiện {len(results[0].obb)} objects (conf ≥ {conf})")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dự đoán OBB trên tập Test")
    parser.add_argument("--image", default=None, help="Đường dẫn ảnh đơn lẻ")
    parser.add_argument("--n", type=int, default=8, help="Số ảnh test hiển thị")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()

    print("=" * 50)
    print("BƯỚC 19: Dự đoán trên tập Test")
    print("=" * 50)

    if args.image:
        predict_single(args.image, conf=args.conf)
    else:
        predict_on_test_images(n_images=args.n, conf=args.conf)
