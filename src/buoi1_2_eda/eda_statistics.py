"""
eda_statistics.py — Buổi 1-2: Phân tích thống kê dữ liệu (EDA)
BƯỚC 3: Thống kê số lượng ảnh
BƯỚC 4: Thống kê số lượng object mỗi lớp
BƯỚC 5: Biểu đồ phân bố lớp
"""

import os
import sys
from pathlib import Path
from glob import glob
from collections import Counter

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATASET_DIR, SPLITS, CLASS_NAMES_DATASET, OUTPUTS_DIR


def count_images(dataset_dir=None):
    """BƯỚC 3: Thống kê số lượng ảnh theo split."""
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    print("📊 Thống kê dataset:")
    print("-" * 35)
    total_images = 0
    stats = {}
    for split in SPLITS:
        images = glob(f"{dataset_dir}/{split}/images/*.jpg")
        images += glob(f"{dataset_dir}/{split}/images/*.png")
        labels = glob(f"{dataset_dir}/{split}/labels/*.txt")
        stats[split] = {"images": len(images), "labels": len(labels)}
        print(f"  {split:6s}: {len(images):4d} ảnh | {len(labels):4d} nhãn")
        total_images += len(images)
    print("-" * 35)
    print(f"  {'TỔNG':6s}: {total_images:4d} ảnh")
    return stats


def count_classes(label_dir: str) -> Counter:
    """BƯỚC 4: Đếm số object mỗi lớp trong thư mục nhãn."""
    class_counts = Counter()
    label_files = glob(os.path.join(label_dir, "*.txt"))
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # bỏ qua dòng trống
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts


def print_class_distribution(dataset_dir=None):
    """In bảng thống kê số object mỗi lớp."""
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    train_counts = count_classes(f"{dataset_dir}/train/labels")
    valid_counts = count_classes(f"{dataset_dir}/valid/labels")

    print("\n📈 Số object mỗi lớp:")
    print(f"{'Class':8s} | {'Tên':25s} | {'Train':8s} | {'Valid':8s}")
    print("-" * 60)

    all_classes = sorted(set(list(train_counts.keys()) + list(valid_counts.keys())))
    for cls in all_classes:
        name = CLASS_NAMES_DATASET[cls] if cls < len(CLASS_NAMES_DATASET) else f"defect_{cls}"
        print(f"  Class {cls:2d} | {name:25s} | {train_counts[cls]:8d} | {valid_counts[cls]:8d}")

    return train_counts, valid_counts


def plot_class_distribution(train_counts: Counter, valid_counts: Counter,
                             save_path: str = None):
    """BƯỚC 5: Vẽ biểu đồ phân bố lớp."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phân bố số lượng object theo lớp', fontsize=14, fontweight='bold')

    for ax, (counts, title, color) in zip(axes, [
        (train_counts, 'Train',      '#2196F3'),
        (valid_counts, 'Validation', '#FF9800'),
    ]):
        classes = [f"Class {k}" for k in sorted(counts.keys())]
        values  = [counts[k] for k in sorted(counts.keys())]
        bars = ax.bar(classes, values, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(f'Tập {title}', fontsize=12)
        ax.set_ylabel('Số lượng object')
        ax.set_xlabel('Lớp')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "class_distribution.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu biểu đồ: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 3-5: EDA — Thống kê & Phân bố lớp")
    print("=" * 50)

    count_images()
    print()
    train_counts, valid_counts = print_class_distribution()
    print()
    plot_class_distribution(train_counts, valid_counts)
