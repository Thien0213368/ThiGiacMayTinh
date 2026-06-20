"""
final_evaluation.py — Buổi 5: Đánh giá cuối cùng trên tập Test
BƯỚC 31: Đánh giá khách quan trên tập Test (chưa từng thấy trong train/val)

Mục tiêu: Đảm bảo tính khách quan — đánh giá trên tập dữ liệu
hoàn toàn mới mà mô hình chưa từng thấy trong quá trình huấn luyện
hay tối ưu hyperparameter.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, OUTPUTS_DIR


def evaluate_on_test(model_path: str, split: str = "test",
                     yaml_path: str = None) -> dict:
    """
    Đánh giá model trên tập dữ liệu chỉ định.

    Args:
        model_path: Đường dẫn tới best.pt
        split: 'test', 'val', hoặc 'train'
        yaml_path: Đường dẫn data_obb.yaml

    Returns:
        dict: {Precision, Recall, mAP50, mAP50-95}
    """
    from ultralytics import YOLO

    if yaml_path is None:
        yaml_path = str(DATASET_YAML)

    if not Path(model_path).exists():
        print(f"❌ Không tìm thấy model: {model_path}")
        return {}

    model = YOLO(model_path)
    metrics = model.val(data=yaml_path, split=split, verbose=False)

    return {
        "Precision": metrics.box.mp,
        "Recall":    metrics.box.mr,
        "mAP50":     metrics.box.map50,
        "mAP50-95":  metrics.box.map,
    }


def run_final_evaluation(yaml_path: str = None):
    """
    Đánh giá và so sánh Baseline vs Best model trên tập Test.
    Đây là kết quả cuối cùng để báo cáo.
    """
    # So sánh 2 model tiềm năng nhất trên tập Test
    models_to_compare = {
        "YOLOv8n (Baseline)": str(PROJECT_ROOT / "runs" / "obb" / "baseline_nano" / "weights" / "best.pt"),
        "YOLOv8m (Best)":     str(PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt"),
    }

    # Thêm model từ thư mục models/ nếu có
    best_model_path = str(PROJECT_ROOT / "models" / "best.pt")
    if Path(best_model_path).exists():
        models_to_compare["Custom (models/best.pt)"] = best_model_path

    test_results = {}
    print("\n🔬 KẾT QUẢ TRÊN TẬP TEST:")
    print("-" * 50)

    for name, path in models_to_compare.items():
        res = evaluate_on_test(path, yaml_path=yaml_path)
        if res:
            test_results[name] = res
            print(f"\n{name}:")
            print(f"  - Precision : {res['Precision']:.4f}")
            print(f"  - Recall    : {res['Recall']:.4f}")
            print(f"  - mAP50     : {res['mAP50']:.4f}")
            print(f"  - mAP50-95  : {res['mAP50-95']:.4f}")
            if res["Precision"] >= 0.85 and res["Recall"] >= 0.85:
                print("  ✅ ĐẠT MỤC TIÊU")
            else:
                print("  ❌ CHƯA ĐẠT MỤC TIÊU")

    return test_results


def generate_test_confusion_matrix(model_path: str = None, yaml_path: str = None):
    """
    Tạo Confusion Matrix trên tập Test và hiển thị.
    (Bổ sung: Ma trận nhầm lẫn trên tập Test)
    """
    from ultralytics import YOLO

    if yaml_path is None:
        yaml_path = str(DATASET_YAML)

    if model_path is None:
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
        print("❌ Không tìm thấy model!")
        return

    print(f"📊 Tạo Confusion Matrix trên tập TEST")
    print(f"   Model: {model_path}")

    model = YOLO(model_path)
    model.val(data=yaml_path, split="test", name="test_cm", plots=True)

    cm_test_path = PROJECT_ROOT / "runs" / "obb" / "test_cm" / "confusion_matrix_normalized.png"

    if cm_test_path.exists():
        plt.figure(figsize=(12, 10))
        img = mpimg.imread(str(cm_test_path))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Normalized Confusion Matrix (TEST SET)', fontsize=16)
        plt.show()
        print(f"✅ Đã lưu: {cm_test_path}")
    else:
        print(f"⚠️  Không tìm thấy file. Kiểm tra thư mục: runs/obb/test_cm/")


def download_best_weights():
    """
    Liệt kê các file trọng số đã train và thông báo đường dẫn.
    (Thay thế google.colab files.download() bằng bản local)
    """
    paths = {
        "Baseline (Nano)":     PROJECT_ROOT / "runs" / "obb" / "baseline_nano"     / "weights" / "best.pt",
        "Exp2 (Small)":        PROJECT_ROOT / "runs" / "obb" / "exp2_small"         / "weights" / "best.pt",
        "Exp3 (Augmented)":    PROJECT_ROOT / "runs" / "obb" / "exp3_augmented"    / "weights" / "best.pt",
        "Exp4 (Medium - Best)": PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt",
        "models/best.pt":      PROJECT_ROOT / "models" / "best.pt",
    }

    print("🔍 Kiểm tra các file trọng số:")
    found_any = False
    best_path = None
    for name, path in paths.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name:25s}: Tồn tại ({size_mb:.2f} MB)")
            print(f"   📁 {path}")
            found_any = True
            if best_path is None or "exp4" in str(path) or "models/best" in str(path):
                best_path = path
        else:
            print(f"❌ {name:25s}: Không thấy")

    if best_path:
        print(f"\n⭐ Trọng số tốt nhất để dùng: {best_path}")
    else:
        print("\n⚠️  Chưa có file weights nào. Hãy train trước!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Đánh giá cuối cùng trên tập Test")
    parser.add_argument("--list-weights", action="store_true",
                        help="Liệt kê các file weights đã train")
    parser.add_argument("--confusion-matrix", action="store_true",
                        help="Tạo confusion matrix trên tập Test")
    args = parser.parse_args()

    print("=" * 50)
    print("BUỔI 5: Đánh giá cuối cùng trên tập Test")
    print("=" * 50)

    if args.list_weights:
        download_best_weights()
    elif args.confusion_matrix:
        generate_test_confusion_matrix()
    else:
        run_final_evaluation()
        print()
        download_best_weights()
