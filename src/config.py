"""
config.py — Cấu hình chung cho toàn bộ dự án
Thị Giác Máy Tính: Phát hiện khuyết tật tấm pin năng lượng mặt trời (YOLOv8 OBB)
"""

from pathlib import Path

# ============================================================
# Đường dẫn gốc (tự động phát hiện)
# ============================================================
# Thư mục chứa config.py này
_SRC_DIR = Path(__file__).parent

# Thư mục gốc dự án (một cấp trên src/)
PROJECT_ROOT = _SRC_DIR.parent

# Các thư mục con
DATASET_DIR  = PROJECT_ROOT / "dataset"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
RUNS_DIR     = PROJECT_ROOT / "runs"

# ============================================================
# Cấu hình Dataset
# ============================================================
DATASET_YAML = PROJECT_ROOT / "data_obb.yaml"

# Các split của dataset
SPLITS = ["train", "valid", "test"]

# ============================================================
# Thông tin lớp (class)
# Bám sát notebook BƯỚC 7: names: [defect_0, defect_1, defect_2, defect_3]
# Nhãn thật trong data.yaml gốc của Roboflow
# ============================================================
CLASS_NAMES_YAML    = ["defect_0", "defect_1", "defect_2", "defect_3"]
CLASS_NAMES_DATASET = ["Bypassed Substrings", "Diode", "Multi Hot Spots", "hotspot"]
CLASS_NAMES_WEB     = ["BypassedSubs", "Diode", "MultiHotSpot", "Hotspot"]

NUM_CLASSES = 4

# ============================================================
# Màu sắc (RGB) cho visualize — bám sát notebook BƯỚC 6
# CLASS_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
# ============================================================
CLASS_COLORS_RGB = [
    (255, 0,   0),      # Đỏ   — Lớp 0: Bypassed Substrings
    (0,   255, 0),      # Xanh lá — Lớp 1: Diode
    (0,   0,   255),    # Xanh dương — Lớp 2: Multi Hot Spots
    (255, 255, 0),      # Vàng — Lớp 3: hotspot
]

# Màu BGR (cho OpenCV) — tương phản cao trên ảnh nhiệt
CLASS_COLORS_BGR = [
    (255, 255, 255),    # Trắng   — Lớp 0
    (0,   255, 255),    # Cyan    — Lớp 1
    (255, 0,   255),    # Magenta — Lớp 2
    (0,   255, 0),      # Xanh lá — Lớp 3
]

CLASS_COLORS_HEX = ["#ffffff", "#00ffff", "#ff00ff", "#00ff00"]

# ============================================================
# Cấu hình Training (mặc định bám sát notebook)
# ============================================================
TRAIN_DEFAULTS = {
    "imgsz":   640,
    "batch":   16,
    "device":  "0",    # GPU 0 — đổi thành "cpu" nếu không có GPU
    "verbose": True,
}

# Các thực nghiệm
EXPERIMENTS = {
    "baseline_nano":    {"model": "yolov8n-obb.pt", "epochs": 100, "name": "baseline_nano"},
    "exp2_small":       {"model": "yolov8s-obb.pt", "epochs": 50,  "name": "exp2_small"},
    "exp3_augmented":   {"model": "yolov8n-obb.pt", "epochs": 50,  "name": "exp3_augmented"},
    "exp4_medium_best": {"model": "yolov8m-obb.pt", "epochs": 30,  "name": "exp4_medium_best"},
}

# ============================================================
# Cấu hình Inference
# ============================================================
CONF_THRESHOLD = 0.3   # Bám sát notebook BƯỚC 19

# Thứ tự ưu tiên load model
MODEL_PRIORITY = [
    str(MODELS_DIR / "best.pt"),
    str(PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "obb" / "exp3_augmented"   / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "obb" / "exp2_small"        / "weights" / "best.pt"),
    str(PROJECT_ROOT / "runs" / "obb" / "baseline_nano"     / "weights" / "best.pt"),
]
