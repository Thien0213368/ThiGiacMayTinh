"""
run_pipeline.py — Pipeline chính chạy toàn bộ dự án theo đúng thứ tự notebook
=================================================================================

Tương đương chạy tất cả cells trong Do_an_TGMT.ipynb theo thứ tự.

Chuỗi dữ liệu (giống notebook):
  STEP 0-2  setup_dataset   → tạo dataset/
  STEP 3-5  eda_statistics  → đọc dataset/, xuất biểu đồ
  STEP 6    visualize_obb   → đọc dataset/, xuất ảnh mẫu
  STEP 7    create_yaml     → đọc train_counts từ eda_statistics → tạo data_obb.yaml
  STEP 8    train_baseline  → đọc data_obb.yaml → tạo runs/obb/baseline_nano/
  STEP 9-10 evaluate_baseline→ đọc runs/obb/baseline_nano/ + dataset/ → in metrics
  STEP 11   plot_curves     → đọc runs/obb/baseline_nano/results.csv → xuất biểu đồ
  STEP 12   exp2_small      → đọc data_obb.yaml → tạo runs/obb/exp2_small/
  STEP 13   exp3_augmented  → đọc data_obb.yaml → tạo runs/obb/exp3_augmented/
  STEP 14   exp4_medium     → đọc data_obb.yaml → tạo runs/obb/exp4_medium_best/
  STEP 15-17 compare        → đọc tất cả runs/obb/*/weights/best.pt → so sánh
  STEP 18   confusion_matrix→ đọc best model → tạo confusion matrix
  STEP 19   predict_test    → đọc best model + dataset/test/ → dự đoán
  Grad-CAM  gradcam         → đọc best model + dataset/test/ → heatmap
  STEP 31   final_evaluation→ đọc best model + data_obb.yaml → kết quả cuối

Cách chạy:
  python run_pipeline.py                  # chạy toàn bộ
  python run_pipeline.py --step eda       # chỉ chạy bước EDA
  python run_pipeline.py --from train     # chạy từ bước train trở đi
  python run_pipeline.py --skip gradcam   # bỏ qua bước gradcam
"""

import sys
import argparse
from pathlib import Path

# Thêm thư mục gốc vào sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import (
    PROJECT_ROOT, DATASET_DIR, DATASET_YAML,
    OUTPUTS_DIR, MODELS_DIR
)


# ============================================================
# Helpers
# ============================================================

def section(title: str):
    """In tiêu đề phần."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def skip_if_done(path, label: str) -> bool:
    """Kiểm tra xem output đã tồn tại chưa (tránh chạy lại)."""
    if Path(path).exists():
        print(f"  ⏩ Bỏ qua {label} — đã có: {path}")
        return True
    return False


# ============================================================
# STEP 1: EDA & Chuẩn bị dữ liệu (Buổi 1-2)
# ============================================================

def step_setup():
    """BƯỚC 0-2: Setup dataset."""
    section("BƯỚC 0-2: Setup & Kiểm tra Dataset")
    from src.buoi1_2_eda.setup_dataset import import_libraries, setup_dataset
    import_libraries()
    setup_dataset()


def step_eda():
    """BƯỚC 3-5: EDA — thống kê và biểu đồ phân bố lớp."""
    section("BƯỚC 3-5: EDA — Thống kê & Phân bố lớp")
    from src.buoi1_2_eda.eda_statistics import (
        count_images, count_classes, print_class_distribution, plot_class_distribution
    )

    count_images()
    print()

    # count_classes kết quả được dùng lại ở BƯỚC 7
    train_counts, valid_counts = print_class_distribution()
    print()
    plot_class_distribution(train_counts, valid_counts)

    # Trả về train_counts để BƯỚC 7 dùng (giống notebook cell variable)
    return train_counts, valid_counts


def step_visualize():
    """BƯỚC 6: Trực quan hóa ảnh + OBB."""
    section("BƯỚC 6: Visualize OBB Labels")
    from src.buoi1_2_eda.visualize_obb import show_sample_images
    show_sample_images(split="train", n_images=6)


def step_create_yaml(train_counts=None):
    """BƯỚC 7: Tạo data_obb.yaml — nhận train_counts từ BƯỚC 3-5."""
    section("BƯỚC 7: Tạo data_obb.yaml")

    if skip_if_done(DATASET_YAML, "data_obb.yaml"):
        return

    from src.buoi1_2_eda.create_yaml import create_data_yaml, auto_detect_classes

    # Nếu có train_counts từ step_eda() → dùng luôn (như notebook)
    # Nếu không → tự detect từ dataset
    if train_counts is not None and len(train_counts) > 0:
        num_classes = max(train_counts.keys()) + 1
        class_names = [f"defect_{i}" for i in range(num_classes)]
        print(f"  📊 Dùng kết quả từ BƯỚC 3-5: {num_classes} lớp")
    else:
        num_classes, class_names = auto_detect_classes()
        print(f"  📊 Phát hiện tự động: {num_classes} lớp")

    create_data_yaml(num_classes=num_classes, class_names=class_names)


# ============================================================
# STEP 2: Training Baseline (Buổi 3)
# ============================================================

def step_train_baseline():
    """BƯỚC 8: Train YOLOv8n-OBB Baseline."""
    section("BƯỚC 8: Train Baseline — YOLOv8n-OBB")

    baseline_weights = PROJECT_ROOT / "runs" / "obb" / "baseline_nano" / "weights" / "best.pt"
    if skip_if_done(baseline_weights, "Baseline weights"):
        return

    from src.buoi3_training.train_baseline import train_baseline
    train_baseline()


def step_evaluate_baseline():
    """BƯỚC 9-10: Đánh giá Baseline."""
    section("BƯỚC 9-10: Đánh giá Baseline & Phân tích lỗi")

    from src.buoi3_training.evaluate_baseline import evaluate_baseline, analyze_predictions
    results = evaluate_baseline()
    if results:
        print()
        analyze_predictions()
    return results  # Truyền sang step_compare nếu cần


def step_plot_baseline_curves():
    """BƯỚC 11: Vẽ đồ thị Loss/mAP của Baseline."""
    section("BƯỚC 11: Vẽ Training Curves — Baseline")
    from src.buoi3_training.plot_training_curves import plot_training_curves
    plot_training_curves(run_name="baseline_nano")


# ============================================================
# STEP 3: Thực nghiệm (Buổi 4)
# ============================================================

def step_exp2():
    """BƯỚC 12: Thực nghiệm 2 — YOLOv8s."""
    section("BƯỚC 12: Thực nghiệm 2 — YOLOv8s (Small)")

    weights = PROJECT_ROOT / "runs" / "obb" / "exp2_small" / "weights" / "best.pt"
    if skip_if_done(weights, "Exp2 weights"):
        return

    from src.buoi4_experiments.exp2_small import train_exp2_small
    train_exp2_small()


def step_exp3():
    """BƯỚC 13: Thực nghiệm 3 — YOLOv8n + Augmentation."""
    section("BƯỚC 13: Thực nghiệm 3 — YOLOv8n + Augmentation")

    weights = PROJECT_ROOT / "runs" / "obb" / "exp3_augmented" / "weights" / "best.pt"
    if skip_if_done(weights, "Exp3 weights"):
        return

    from src.buoi4_experiments.exp3_augmented import train_exp3_augmented
    train_exp3_augmented()


def step_exp4():
    """BƯỚC 14: Thực nghiệm 4 — YOLOv8m (Best)."""
    section("BƯỚC 14: Thực nghiệm 4 — YOLOv8m (Medium) — BEST")

    weights = PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt"
    if skip_if_done(weights, "Exp4 weights"):
        return

    from src.buoi4_experiments.exp4_medium import train_exp4_medium
    train_exp4_medium()


def step_compare():
    """BƯỚC 15-17: So sánh tất cả thực nghiệm."""
    section("BƯỚC 15-17: So sánh Thực nghiệm & Biểu đồ")

    from src.buoi4_experiments.compare_experiments import (
        collect_all_results, print_comparison_table, plot_comparison, print_analysis
    )

    # collect_all_results() gọi get_metrics() → gọi evaluate_on_test()
    # Đây là đúng chuỗi notebook: sau khi train xong → val() để so sánh
    comparison = collect_all_results()

    if comparison:
        print()
        print_comparison_table(comparison)
        print()
        plot_comparison(comparison)
        print()
        print_analysis(comparison)
    else:
        print("⚠️  Chưa có thực nghiệm nào hoàn thành.")

    return comparison  # Truyền sang buổi 5 nếu cần


# ============================================================
# STEP 4: Đánh giá chi tiết (Buổi 5)
# ============================================================

def step_confusion_matrix():
    """BƯỚC 18: Confusion Matrix."""
    section("BƯỚC 18: Confusion Matrix")
    from src.buoi5_evaluation.confusion_matrix import show_confusion_matrix
    show_confusion_matrix()


def step_predict_test():
    """BƯỚC 19: Dự đoán trên tập Test."""
    section("BƯỚC 19: Dự đoán trên tập Test")
    from src.buoi5_evaluation.predict_test import predict_on_test_images
    predict_on_test_images(n_images=8)


def step_gradcam():
    """Grad-CAM: Vùng chú ý của mô hình."""
    section("BUỔI 5: Grad-CAM Visualization")
    try:
        from src.buoi5_evaluation.gradcam import visualize_gradcam
        visualize_gradcam(n_images=4)
    except ImportError as e:
        print(f"⚠️  Bỏ qua Grad-CAM: {e}")
        print("   Cài đặt: pip install grad-cam")


def step_final_evaluation():
    """BƯỚC 31: Đánh giá cuối — Baseline vs Best trên Test set."""
    section("BƯỚC 31: Đánh giá cuối cùng trên tập Test")
    from src.buoi5_evaluation.final_evaluation import (
        run_final_evaluation, download_best_weights
    )

    # run_final_evaluation() gọi evaluate_on_test() — cùng hàm với compare_experiments
    # Đây là BƯỚC cuối trong notebook
    test_results = run_final_evaluation()
    print()
    download_best_weights()
    return test_results


# ============================================================
# Pipeline chính
# ============================================================

# Ánh xạ tên bước → hàm (theo đúng thứ tự notebook)
STEPS = {
    "setup":      step_setup,
    "eda":        step_eda,
    "visualize":  step_visualize,
    "yaml":       step_create_yaml,
    "train":      step_train_baseline,
    "evaluate":   step_evaluate_baseline,
    "curves":     step_plot_baseline_curves,
    "exp2":       step_exp2,
    "exp3":       step_exp3,
    "exp4":       step_exp4,
    "compare":    step_compare,
    "cm":         step_confusion_matrix,
    "predict":    step_predict_test,
    "gradcam":    step_gradcam,
    "final":      step_final_evaluation,
}

STEP_ORDER = list(STEPS.keys())


def run_all(skip_steps: list = None, from_step: str = None):
    """Chạy toàn bộ pipeline theo đúng thứ tự notebook."""
    if skip_steps is None:
        skip_steps = []

    # Xác định từ bước nào
    start_idx = 0
    if from_step and from_step in STEP_ORDER:
        start_idx = STEP_ORDER.index(from_step)

    # Chạy theo thứ tự — truyền dữ liệu giữa các bước như notebook
    train_counts = None
    valid_counts  = None

    steps_to_run = STEP_ORDER[start_idx:]
    for step_name in steps_to_run:
        if step_name in skip_steps:
            print(f"\n⏩ Bỏ qua bước: {step_name}")
            continue

        fn = STEPS[step_name]

        # Truyền train_counts từ eda → yaml (giống notebook cell variable)
        if step_name == "yaml":
            step_create_yaml(train_counts=train_counts)
        elif step_name == "eda":
            result = fn()
            if result:
                train_counts, valid_counts = result
        else:
            fn()

    print()
    print("=" * 60)
    print("  🎉 Hoàn thành toàn bộ pipeline!")
    print("=" * 60)


def run_single(step_name: str):
    """Chạy một bước đơn lẻ."""
    if step_name not in STEPS:
        print(f"❌ Bước không hợp lệ: {step_name}")
        print(f"   Các bước hợp lệ: {', '.join(STEP_ORDER)}")
        return

    print(f"▶️  Chạy bước: {step_name}")
    if step_name == "yaml":
        step_create_yaml(train_counts=None)  # tự detect
    else:
        STEPS[step_name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline chạy toàn bộ dự án theo thứ tự notebook"
    )
    parser.add_argument(
        "--step", default=None,
        help=f"Chỉ chạy 1 bước cụ thể. Các bước: {', '.join(STEP_ORDER)}"
    )
    parser.add_argument(
        "--from", dest="from_step", default=None,
        help="Chạy từ bước chỉ định đến cuối"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Bỏ qua các bước (vd: --skip gradcam exp2)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Liệt kê tất cả các bước"
    )

    args = parser.parse_args()

    if args.list:
        print("📋 Danh sách các bước (theo thứ tự notebook):")
        for i, name in enumerate(STEP_ORDER, 1):
            print(f"  {i:2d}. {name}")
    elif args.step:
        run_single(args.step)
    else:
        run_all(skip_steps=args.skip, from_step=args.from_step)
