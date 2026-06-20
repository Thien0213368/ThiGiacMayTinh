"""
create_yaml.py — Buổi 1-2: Tạo file data_obb.yaml cho YOLO
BƯỚC 7: Tạo file cấu hình dataset cho YOLOv8 OBB

Kết nối notebook:
  BƯỚC 3-5 (eda_statistics.py)  →  count_classes()  →  BƯỚC 7 (script này)
  train_counts = count_classes(...) → num_classes = max(train_counts.keys()) + 1
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_DIR, NUM_CLASSES, CLASS_NAMES_YAML
# Import hàm count_classes từ bước trước (đúng như notebook: dùng lại kết quả của STEP 3-5)
from src.buoi1_2_eda.eda_statistics import count_classes


def create_data_yaml(dataset_dir=None, output_path=None, num_classes=None,
                     class_names=None):
    """
    BƯỚC 7: Tạo file data_obb.yaml cho YOLOv8 training.

    Args:
        dataset_dir: Đường dẫn tuyệt đối tới dataset
        output_path: Đường dẫn lưu file yaml
        num_classes: Số lớp
        class_names: Danh sách tên lớp
    """
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)
    if output_path is None:
        output_path = str(PROJECT_ROOT / "data_obb.yaml")
    if num_classes is None:
        num_classes = NUM_CLASSES
    if class_names is None:
        class_names = CLASS_NAMES_YAML

    data_yaml = {
        'path': dataset_dir,
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    num_classes,
        'names': class_names,
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Tạo file {output_path} thành công!")
    print(f"   Số lớp   : {num_classes}")
    print(f"   Tên lớp  : {class_names}")
    print(f"   Dataset  : {dataset_dir}")

    # In nội dung file
    with open(output_path, 'r') as f:
        print("\n📄 Nội dung data_obb.yaml:")
        print(f.read())

    return output_path


def auto_detect_classes(dataset_dir=None):
    """
    Tự động phát hiện số lớp từ label files.
    Sử dụng lại hàm count_classes() từ eda_statistics.py
    (giống notebook: kết quả STEP 3-5 → num_classes = max(train_counts.keys()) + 1)
    """
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    # Dùng lại count_classes từ eda_statistics (bước 3-5) — chính xác như notebook
    train_counts = count_classes(f"{dataset_dir}/train/labels")

    if not train_counts:
        return NUM_CLASSES, CLASS_NAMES_YAML

    num_classes = max(train_counts.keys()) + 1
    class_names = [f"defect_{i}" for i in range(num_classes)]
    return num_classes, class_names


if __name__ == "__main__":
    print("=" * 50)
    print("BƯỚC 7: Tạo file data_obb.yaml")
    print("=" * 50)

    # Tự động phát hiện số lớp từ dataset
    if DATASET_DIR.exists():
        nc, names = auto_detect_classes()
        print(f"📊 Phát hiện {nc} lớp từ dataset: {names}")
    else:
        nc, names = NUM_CLASSES, CLASS_NAMES_YAML
        print(f"📊 Dùng cấu hình mặc định: {nc} lớp")

    create_data_yaml(num_classes=nc, class_names=names)
