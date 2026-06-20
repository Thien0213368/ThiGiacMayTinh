"""
plot_training_curves.py — Buổi 3: Vẽ đồ thị Loss & Metrics
BƯỚC 11: Vẽ đồ thị Loss trong quá trình train từ results.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, OUTPUTS_DIR


def plot_training_curves(run_name: str = "baseline_nano",
                          save_path: str = None):
    """
    BƯỚC 11: Vẽ đồ thị Loss & Metrics từ results.csv.

    Args:
        run_name: Tên thư mục run trong runs/obb/ (vd: 'baseline_nano')
        save_path: Đường dẫn lưu ảnh output
    """
    try:
        import pandas as pd
    except ImportError:
        print("❌ Thiếu pandas. Chạy: pip install pandas")
        return

    results_csv = PROJECT_ROOT / "runs" / "obb" / run_name / "results.csv"

    if not results_csv.exists():
        print(f"⚠️  Chưa tìm thấy {results_csv}")
        print("   Hãy train xong rồi chạy lại ô này")
        return

    df = pd.read_csv(str(results_csv))
    df.columns = df.columns.str.strip()   # Xóa khoảng trắng tên cột

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Training Curves — {run_name}',
                 fontsize=13, fontweight='bold')

    # --- Train Loss ---
    if 'train/box_loss' in df.columns:
        axes[0].plot(df['epoch'], df['train/box_loss'], 'b-', label='Box Loss')
        if 'train/cls_loss' in df.columns:
            axes[0].plot(df['epoch'], df['train/cls_loss'], 'r-', label='Cls Loss')
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Không có dữ liệu train/box_loss',
                     ha='center', va='center', transform=axes[0].transAxes)

    # --- Validation mAP ---
    if 'metrics/mAP50(B)' in df.columns:
        axes[1].plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', linewidth=2)
        axes[1].set_title('Validation mAP@50')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP@50')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Không có dữ liệu mAP',
                     ha='center', va='center', transform=axes[1].transAxes)

    # --- Precision & Recall ---
    if 'metrics/precision(B)' in df.columns:
        axes[2].plot(df['epoch'], df['metrics/precision(B)'], 'b-', label='Precision')
        axes[2].plot(df['epoch'], df['metrics/recall(B)'],    'r-', label='Recall')
        axes[2].axhline(y=0.85, color='green', linestyle='--', label='Mục tiêu 0.85')
        axes[2].set_title('Precision & Recall')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Không có dữ liệu Precision/Recall',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / f"{run_name}_training_curves.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vẽ training curves")
    parser.add_argument("--run", default="baseline_nano",
                        help="Tên run (default: baseline_nano)")
    args = parser.parse_args()

    print("=" * 50)
    print(f"BƯỚC 11: Vẽ đồ thị Loss — {args.run}")
    print("=" * 50)
    plot_training_curves(run_name=args.run)
