"""
visualize_obb.py — Buổi 1-2: Trực quan hóa ảnh + OBB labels
BƯỚC 6: Hiển thị ảnh với Oriented Bounding Box (OBB)

Bám sát notebook BƯỚC 6:
  CLASS_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
  points = np.array([float(x) for x in data[1:9]]).reshape((4, 2))
  points[:, 0] *= w; points[:, 1] *= h
  cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
"""

import os
import sys
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATASET_DIR, CLASS_COLORS_RGB, OUTPUTS_DIR


def visualize_obb(image_path: str, label_path: str, ax):
    """
    Hiển thị ảnh với các OBB (Oriented Bounding Box).
    Bám sát chính xác notebook BƯỚC 6.

    Args:
        image_path: Đường dẫn tới ảnh (.jpg)
        label_path: Đường dẫn tới file nhãn (.txt)
        ax: matplotlib axes để vẽ
    """
    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) < 9:
                continue

            class_id = int(data[0])
            # 4 điểm góc OBB (normalized x1,y1, x2,y2, x3,y3, x4,y4)
            points = np.array([float(x) for x in data[1:9]]).reshape((4, 2))

            # Chuyển đổi tọa độ từ normalized sang pixel
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)

            # Lấy màu theo class_id
            color = CLASS_COLORS_RGB[class_id % len(CLASS_COLORS_RGB)]
            color_bgr = (color[2], color[1], color[0])  # RGB → BGR cho cv2

            # Vẽ đa giác OBB
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            # Ghi tên lớp
            cv2.putText(image, f"C{class_id}", tuple(points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ax.imshow(image)
    ax.set_title(os.path.basename(image_path), fontsize=8)
    ax.axis('off')


def show_sample_images(dataset_dir=None, split: str = "train",
                       n_images: int = 6, save_path: str = None):
    """
    Hiển thị n_images ảnh mẫu từ split với OBB labels.

    Args:
        dataset_dir: Đường dẫn dataset (mặc định DATASET_DIR)
        split: 'train', 'valid', hoặc 'test'
        n_images: Số ảnh hiển thị (default 6)
        save_path: Đường dẫn lưu ảnh output
    """
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    image_files = sorted(glob(f"{dataset_dir}/{split}/images/*.jpg"))[:n_images]
    label_files = [
        f.replace('images', 'labels').replace('.jpg', '.txt')
        for f in image_files
    ]

    if not image_files:
        print(f"❌ Không tìm thấy ảnh trong {dataset_dir}/{split}/images/")
        return

    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f'Mẫu dữ liệu với OBB Labels — Tập {split}',
                 fontsize=14, fontweight='bold')

    for i, (img_path, lbl_path, ax) in enumerate(zip(image_files, label_files, axes)):
        visualize_obb(img_path, lbl_path, ax)

    # Ẩn axes thừa
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "sample_obb.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 6: Trực quan hóa ảnh + OBB Labels")
    print("=" * 50)
    show_sample_images(split="train", n_images=6)
