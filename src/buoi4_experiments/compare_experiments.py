"""
compare_experiments.py — Buổi 4: So sánh tất cả thực nghiệm
BƯỚC 15: Thu thập metrics từ 4 thực nghiệm
BƯỚC 16: Bảng so sánh + Biểu đồ (Báo cáo Buổi 4)
BƯỚC 17: Giải thích sự khác biệt giữa các thực nghiệm

Kết nối notebook (chuỗi dữ liệu):
  Buổi 3 (evaluate_baseline.py)  →  runs/obb/baseline_nano/weights/best.pt
  Buổi 4 Exp2/3/4               →  runs/obb/exp*/weights/best.pt
  Script này                      →  đọc tất cả model trên và so sánh
  BƯỚC 31 (final_evaluation.py)  →  dùng hàm evaluate_on_test cùng source
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_YAML, OUTPUTS_DIR
# Dùng lại evaluate_on_test từ buoi5 (giống notebook: cùng hàm val(), không viết lại)
from src.buoi5_evaluation.final_evaluation import evaluate_on_test


# ============================================================
# Đường dẫn weights của 4 thực nghiệm
# ============================================================
EXPERIMENT_PATHS = {
    "Exp1: YOLOv8n (Baseline)": "runs/obb/baseline_nano/weights/best.pt",
    "Exp2: YOLOv8s":             "runs/obb/exp2_small/weights/best.pt",
    "Exp3: YOLOv8n+Augment":    "runs/obb/exp3_augmented/weights/best.pt",
    "Exp4: YOLOv8m (Best)":     "runs/obb/exp4_medium_best/weights/best.pt",
}

COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']


def get_metrics(model_path: str, yaml_path: str = None) -> dict | None:
    """
    BƯỚC 15: Đánh giá model và trả về dict metrics.
    Gọ evaluate_on_test() từ final_evaluation.py — cùng hàm như notebook STEP 31.

    Args:
        model_path: Đường dẫn tới best.pt
        yaml_path: Đường dẫn data_obb.yaml

    Returns:
        dict hoặc None nếu model không tồn tại
    """
    if not Path(model_path).exists():
        return None

    # Đúng như notebook: gọi model.val() trên tập 'val'
    res = evaluate_on_test(model_path, split='val', yaml_path=yaml_path)
    if not res:
        return None

    return {
        "mAP50":     round(res["mAP50"],     4),
        "mAP50-95":  round(res["mAP50-95"],  4),
        "Precision": round(res["Precision"],  4),
        "Recall":    round(res["Recall"],     4),
    }


def collect_all_results(yaml_path: str = None) -> dict:
    """Thu thập kết quả từ tất cả thực nghiệm."""
    comparison = {}
    for name, rel_path in EXPERIMENT_PATHS.items():
        abs_path = str(PROJECT_ROOT / rel_path)
        metrics = get_metrics(abs_path, yaml_path)
        if metrics:
            comparison[name] = metrics
            print(f"✅ {name}: mAP50={metrics['mAP50']}, "
                  f"P={metrics['Precision']}, R={metrics['Recall']}")
        else:
            print(f"⏩ {name}: Chưa train xong hoặc sai đường dẫn, bỏ qua")
    return comparison


def print_comparison_table(comparison: dict):
    """BƯỚC 16: In bảng so sánh."""
    try:
        import pandas as pd
        df = pd.DataFrame(comparison).T
        print("\n📋 BẢNG SO SÁNH CÁC THỰC NGHIỆM")
        print("=" * 60)
        print(df.to_string())
        print("=" * 60)

        # Lưu CSV
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = str(OUTPUTS_DIR / "comparison_table.csv")
        df.to_csv(csv_path)
        print(f"🖼️  Đã lưu: {csv_path}")

        return df
    except ImportError:
        print("⚠️  Thiếu pandas. Chạy: pip install pandas")
        for name, vals in comparison.items():
            print(f"\n{name}:")
            for k, v in vals.items():
                print(f"  {k}: {v}")
        return None


def plot_comparison(comparison: dict, save_path: str = None):
    """BƯỚC 16: Vẽ biểu đồ so sánh các thực nghiệm."""
    if not comparison:
        print("⚠️  Không có dữ liệu để vẽ biểu đồ")
        return

    metrics_to_plot = ["mAP50", "Precision", "Recall"]
    exp_names = list(comparison.keys())
    short_names = [f"Exp{i+1}" for i in range(len(exp_names))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('So sánh kết quả các thực nghiệm (Buổi 4)',
                 fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, metrics_to_plot):
        values = [comparison[exp][metric] for exp in exp_names]
        bars = ax.bar(short_names, values,
                      color=COLORS[:len(exp_names)],
                      alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.axhline(y=0.85, color='red', linestyle='--',
                   linewidth=1.5, label='Mục tiêu 0.85')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Giá trị')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Số trên cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Legend tên thực nghiệm đầy đủ
    legend_elements = [
        patches.Patch(color=COLORS[i],
                      label=f'Exp{i+1}: {name.split(":")[1].strip()}')
        for i, name in enumerate(exp_names)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)

    plt.tight_layout()

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "experiment_comparison.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n🖼️  Đã lưu: {save_path}")


def print_analysis(comparison: dict):
    """
    BƯỚC 16: Nhận xét kết quả tốt nhất.
    BƯỚC 17: Giải thích sự khác biệt.
    """
    if not comparison:
        return

    try:
        import pandas as pd
        df = pd.DataFrame(comparison).T
        best_exp = df['mAP50'].idxmax()
        print(f"\n🏆 Mô hình tốt nhất: {best_exp}")
        print(f"   mAP50     = {comparison[best_exp]['mAP50']:.4f}")
        print(f"   Precision = {comparison[best_exp]['Precision']:.4f}")
        print(f"   Recall    = {comparison[best_exp]['Recall']:.4f}")

        p = comparison[best_exp]["Precision"]
        r = comparison[best_exp]["Recall"]
        if p >= 0.85 and r >= 0.85:
            print("\n🎉 ĐÃ ĐẠT MỤC TIÊU: Precision & Recall ≥ 0.85!")
        else:
            print(f"\n⚠️  Chưa đạt mục tiêu. Cần cải thiện thêm ở Buổi 5.")
            if p < 0.85:
                print("   → Precision thấp → tăng conf threshold hoặc giảm false positive")
            if r < 0.85:
                print("   → Recall thấp → tăng dữ liệu hoặc giảm conf threshold")
    except ImportError:
        pass

    explanation = """
📝 PHÂN TÍCH SỰ KHÁC BIỆT GIỮA CÁC THỰC NGHIỆM
================================================

1. Exp1 (YOLOv8n - Baseline):
   - Model nhỏ nhất (3.1M params), train nhanh
   - Capacity thấp → có thể underfitting với dataset phức tạp
   - Dùng làm mốc so sánh

2. Exp2 (YOLOv8s - Small):
   - Nhiều tham số hơn (11.4M) → học được trưng phong phú hơn
   - Nếu dataset đủ lớn: mAP thường tăng 5-15% so với Nano
   - Nhược điểm: train chậm hơn ~2x

3. Exp3 (YOLOv8n + Augmentation):
   - Cùng model nhỏ nhưng augmentation mạnh
   - Mosaic: ghép 4 ảnh → model thấy đối tượng trong nhiều bối cảnh
   - Mixup: trộn 2 ảnh → giảm overconfidence của model
   - Rotation ±15°: phù hợp với OBB (đối tượng nằm nghiêng)
   - Kết quả: thường giảm overfitting, tăng recall

4. Exp4 (YOLOv8m - Medium + Augmentation):
   - Kết hợp model lớn + augmentation → best of both worlds
   - 26.4M params → học được đặc trưng chi tiết hơn
   - Train 100 epochs với early stopping → hội tụ sâu hơn
   - Thường cho kết quả tốt nhất nhưng cần GPU và thời gian

KẾT LUẬN:
  - Nếu tài nguyên hạn chế → Exp3 (augment) cho ROI tốt nhất
  - Nếu có GPU mạnh → Exp4 là lựa chọn tối ưu
"""
    print(explanation)


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 15-17: So sánh tất cả thực nghiệm")
    print("=" * 50)

    comparison = collect_all_results()

    if comparison:
        print()
        df = print_comparison_table(comparison)
        print()
        plot_comparison(comparison)
        print()
        print_analysis(comparison)
    else:
        print("\n⚠️  Chưa có thực nghiệm nào hoàn thành.")
        print("   Hãy chạy ít nhất 1 script trong buoi4_experiments/")
