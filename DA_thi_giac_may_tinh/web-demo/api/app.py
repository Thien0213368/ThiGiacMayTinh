"""
DefectDetect OBB — FastAPI Backend v4.0
Bám sát CHÍNH XÁC notebook TGMT_Buoi1_den_Buoi4.ipynb

Notebook BƯỚC 6 (visualize_obb):
  CLASS_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
  points = np.array([float(x) for x in data[1:9]]).reshape((4, 2))
  points[:, 0] *= w; points[:, 1] *= h
  cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

Notebook BƯỚC 7 (data.yaml):
  class_names = [f"defect_{i}" for i in range(num_classes)]
  → names: [defect_0, defect_1, defect_2, defect_3]

Notebook BƯỚC 19 (predict):
  model_best = YOLO('runs/obb/exp4_medium_best/weights/best.pt')
  results = model_best.predict(source=img_path, conf=0.3, verbose=False)
  res_plotted = results[0].plot()   # ← YOLO tự render OBB
  res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

Endpoint /detect trả về:
  - annotated_image: base64 ảnh đã vẽ OBB (giống notebook result.plot())
  - detections: JSON list mỗi detection (cho sidebar)

v4.0 THAY ĐỔI:
  - Khi không có model YOLO: đọc nhãn ground truth từ dataset thay vì dùng dữ liệu giả
  - Hỗ trợ tìm file nhãn tương ứng với ảnh upload (theo tên file)
  - Endpoint /samples trả về danh sách ảnh test từ dataset
"""

import os
import io
import time
import math
import base64
import hashlib
import numpy as np
import cv2
from pathlib import Path
from glob import glob

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ============================================================
# Màu sắc tương phản cao (hiển thị rõ trên ảnh nhiệt hồng ngoại màu cam/vàng)
# ============================================================
CLASS_COLORS_BGR = [
    (255, 255, 255),   # Trắng       - Lớp 0: Bypassed Substrings
    (0,   255, 255),   # Cyan        - Lớp 1: Diode
    (255, 0,   255),   # Magenta     - Lớp 2: Multi Hot Spots
    (0,   255, 0),     # Xanh lá     - Lớp 3: hotspot
]
CLASS_COLORS_HEX = ["#ffffff", "#00ffff", "#ff00ff", "#00ff00"]

# ============================================================
# Tên lớp theo data.yaml gốc của dataset
# ============================================================
CLASS_NAMES_DATASET = ["Bypassed Substrings", "Diode", "Multi Hot Spots", "hotspot"]
CLASS_NAMES_WEB     = ["BypassedSubs", "Diode", "MultiHotSpot", "Hotspot"]

# ============================================================
# Đường dẫn dataset
# ============================================================
_SCRIPT_DIR = Path(__file__).parent           # web-demo/api/
_WEBDEMO_DIR = _SCRIPT_DIR.parent             # web-demo/
_PROJECT_DIR = _WEBDEMO_DIR.parent            # DA_thi_giac_may_tinh/  (inner)
_ROOT_DIR    = _PROJECT_DIR.parent            # DA_thi_giac_may_tinh/  (root workspace)

DATASET_DIR = None
for candidate in [
    # Cấu trúc MỚI sau khi tái cấu trúc
    _ROOT_DIR    / "dataset",
    # Cấu trúc cũ (fallback)
    _PROJECT_DIR / "ThiGiacMayTinh2.v6i.yolov8-obb",
    _WEBDEMO_DIR / "dataset",
    Path(os.environ.get("DATASET_PATH", "")),
]:
    if candidate and candidate.exists() and (candidate / "data.yaml").exists():
        DATASET_DIR = candidate
        break

if DATASET_DIR:
    print(f"[OK] Dataset found: {DATASET_DIR}")
else:
    print("[WARN] Dataset not found — ground truth labels unavailable")


# ============================================================
# Index: xây dựng mapping tên ảnh → (image_path, label_path)
# ============================================================
IMAGE_INDEX = {}  # basename_no_ext → {"image": Path, "label": Path, "split": str}
IMAGE_HASH_INDEX = {}  # md5 hash → stem (for matching uploaded images by content)

# Mapping sample key (frontend) → dataset image stem
# Sample images in assets/ are copies of these dataset images
SAMPLE_KEY_MAP = {
    "hotspot": "DJI_20260411084050_0371_T_JPG.rf.9f00bf40ead302a42808995560e07c2c",
    "multi":   "DJI_20260411084108_0380_T_JPG.rf.6361e1be3269231fa5da97590cc79340",
    "diode":   "DJI_20260411084052_0372_T_JPG.rf.9acd7f68d322dda6392878bcfb8ec131",
}

def build_image_index():
    """Scan all splits (train/valid/test) and build index + hash index."""
    global IMAGE_INDEX, IMAGE_HASH_INDEX
    if DATASET_DIR is None:
        return

    for split in ["test", "valid", "train"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            lbl_path = lbl_dir / (stem + ".txt")
            IMAGE_INDEX[stem] = {
                "image": img_path,
                "label": lbl_path if lbl_path.exists() else None,
                "split": split,
            }
            # Build content hash for matching uploaded images
            try:
                img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                IMAGE_HASH_INDEX[img_hash] = stem
            except Exception:
                pass

    print(f"[OK] Image index: {len(IMAGE_INDEX)} images ({sum(1 for v in IMAGE_INDEX.values() if v['label'])} with labels)")
    print(f"[OK] Hash index: {len(IMAGE_HASH_INDEX)} hashes")

build_image_index()


# ============================================================
# Load YOLO model theo thứ tự ưu tiên
# ============================================================
MODEL_PRIORITY = [
    os.environ.get("YOLO_MODEL_PATH", ""),
    # ✅ Cấu trúc MỚI: models/best.pt ở root workspace
    str(_ROOT_DIR    / "models" / "best.pt"),
    # ✅ Mô hình đã huấn luyện custom (4 lớp khuyết tật pin mặt trời)
    str(_PROJECT_DIR / "best.pt"),
    # Kết quả training ở root workspace (cấu trúc mới)
    str(_ROOT_DIR    / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt"),
    str(_ROOT_DIR    / "runs" / "obb" / "exp3_augmented"   / "weights" / "best.pt"),
    str(_ROOT_DIR    / "runs" / "obb" / "exp2_small"        / "weights" / "best.pt"),
    str(_ROOT_DIR    / "runs" / "obb" / "baseline_nano"     / "weights" / "best.pt"),
    # Fallback: các file pre-trained mặc định (15 lớp DOTA — không phù hợp với dự án)
    str(_PROJECT_DIR / "yolov8m-obb.pt"),   # Medium — pre-trained DOTA
    str(_PROJECT_DIR / "yolov8s-obb.pt"),   # Small  — pre-trained DOTA
    str(_PROJECT_DIR / "yolov8n-obb.pt"),   # Nano   — pre-trained DOTA
    "runs/obb/exp4_medium_best/weights/best.pt",
    "runs/obb/exp3_augmented/weights/best.pt",
    "runs/obb/exp2_small/weights/best.pt",
    "runs/obb/baseline_nano/weights/best.pt",
]

model      = None
model_name = None
model_class_names = CLASS_NAMES_WEB  # default

for path in MODEL_PRIORITY:
    if path and Path(path).exists():
        try:
            from ultralytics import YOLO
            model      = YOLO(path)
            model_name = path
            if hasattr(model, 'names') and model.names:
                _names = model.names
                if isinstance(_names, dict):
                    _names_list = [_names.get(i, f"defect_{i}") for i in range(len(_names))]
                else:
                    _names_list = list(_names)
                if all(n.startswith("defect_") for n in _names_list):
                    model_class_names = CLASS_NAMES_WEB
                else:
                    model_class_names = _names_list
            print(f"[OK] Model loaded: {path}")
            print(f"     Class names: {model_class_names}")
            break
        except Exception as e:
            print(f"[WARN] Lỗi load {path}: {e}")

if model is None:
    print("[INFO] No YOLO model — sẽ dùng ground truth labels từ dataset")


# ============================================================
# Helper: tính góc và kích thước từ corners pixel
# ============================================================
def compute_obb_angle_size(corners_pixel):
    """corners_pixel: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (pixel)"""
    p0 = corners_pixel[0]
    p1 = corners_pixel[1]
    p2 = corners_pixel[2]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    angle_deg = round(math.degrees(math.atan2(dy, dx)), 1)
    side01 = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
    side12 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    return angle_deg, round(side01, 1), round(side12, 1)


# ============================================================
# Helper: vẽ OBB theo đúng notebook BƯỚC 6
# cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
# ============================================================
def draw_obb_notebook_style(img_bgr, corners_pixel, cls_id, cls_name, conf=None):
    """Vẽ OBB chính xác như notebook BƯỚC 6 nhưng với màu tương phản cao hơn"""
    color = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
    pts = np.array(corners_pixel, dtype=np.int32).reshape((-1, 1, 2))
    # Vẽ viền đen để tạo outline (dễ nhìn hơn trên ảnh nhiệt)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=(0, 0, 0), thickness=5)
    # Vẽ màu chính
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=3)
    # Label tại điểm đầu tiên
    if conf is not None:
        label = f"{cls_name} {conf:.2f}"
    else:
        label = cls_name
    x0, y0 = int(corners_pixel[0][0]), int(corners_pixel[0][1])
    # Nền label đen để tương phản
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img_bgr, (x0, y0 - th - 8), (x0 + tw + 6, y0 + 2), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (x0 + 3, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img_bgr


# ============================================================
# Helper: encode ảnh sang base64 để gửi về frontend
# ============================================================
def encode_image_base64(img_bgr):
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')


# ============================================================
# Helper: đọc ground truth labels từ file .txt (format OBB YOLO)
# Bám sát BƯỚC 6: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
# ============================================================
def read_ground_truth_labels(label_path, img_w, img_h):
    """
    Đọc file nhãn OBB và trả về danh sách detections.
    Format mỗi dòng: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    """
    detections = []
    if not label_path or not Path(label_path).exists():
        return detections

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 9:
            continue

        cls_id = int(data[0])
        # 4 điểm góc OBB (normalized x1,y1, x2,y2, x3,y3, x4,y4)
        pts_norm = [float(x) for x in data[1:9]]

        # Chuyển normalized → pixel (bám sát notebook BƯỚC 6)
        corners_pixel = []
        for j in range(4):
            px = pts_norm[j * 2] * img_w
            py = pts_norm[j * 2 + 1] * img_h
            # Clamp trong ảnh
            px = max(0, min(img_w - 1, px))
            py = max(0, min(img_h - 1, py))
            corners_pixel.append([px, py])

        angle, w_px, h_px = compute_obb_angle_size(corners_pixel)

        cls_name = CLASS_NAMES_WEB[cls_id] if cls_id < len(CLASS_NAMES_WEB) else f"defect_{cls_id}"

        detections.append({
            "cls":       cls_name,
            "cls_id":    cls_id,
            "conf":      1.0,  # Ground truth → confidence = 1.0
            "pts":       pts_norm,
            "corners_pixel": corners_pixel,
            "angle":     angle,
            "size":      [round(w_px, 1), round(h_px, 1)],
            "color_hex": CLASS_COLORS_HEX[cls_id] if cls_id < 4 else "#00e5ff",
        })

    return detections


# ============================================================
# Helper: tìm file nhãn tương ứng với ảnh upload
# ============================================================
def find_matching_label(filename, img_bytes=None):
    """
    Find matching label for uploaded image.
    Tries: exact stem, sample key map, content hash, partial match.
    """
    if not filename and not img_bytes:
        return None, None

    stem = Path(filename).stem if filename else ""

    # 1. Exact match in index
    if stem in IMAGE_INDEX:
        entry = IMAGE_INDEX[stem]
        return entry["image"], entry["label"]

    # 2. Sample key map (e.g. "hotspot" → dataset stem)
    if stem in SAMPLE_KEY_MAP:
        mapped_stem = SAMPLE_KEY_MAP[stem]
        if mapped_stem in IMAGE_INDEX:
            entry = IMAGE_INDEX[mapped_stem]
            return entry["image"], entry["label"]

    # 3. Content hash match (uploaded image == dataset image)
    if img_bytes and IMAGE_HASH_INDEX:
        img_hash = hashlib.md5(img_bytes).hexdigest()
        if img_hash in IMAGE_HASH_INDEX:
            matched_stem = IMAGE_HASH_INDEX[img_hash]
            entry = IMAGE_INDEX[matched_stem]
            return entry["image"], entry["label"]

    # 4. Partial match (filename contains stem or vice versa)
    stem_lower = stem.lower()
    for key, entry in IMAGE_INDEX.items():
        if stem_lower and (stem_lower in key.lower() or key.lower() in stem_lower):
            return entry["image"], entry["label"]

    return None, None


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="DefectDetect OBB API v4",
    description="YOLOv8 OBB — Phát hiện khuyết tật tấm pin năng lượng mặt trời (YOLO model + ground truth labels)",
    version="4.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "mode":   "real_model" if model else "ground_truth",
        "model":  model_name or "none",
        "version": "4.0.0",
        "dataset": str(DATASET_DIR) if DATASET_DIR else None,
        "image_count": len(IMAGE_INDEX),
        "class_names_web": CLASS_NAMES_WEB,
    }


@app.get("/status")
def status():
    return {
        "model_loaded": model is not None,
        "model_path":   model_name,
        "dataset_path": str(DATASET_DIR) if DATASET_DIR else None,
        "image_count":  len(IMAGE_INDEX),
        "class_names":  model_class_names if model else CLASS_NAMES_WEB,
        "message": (
            f"✅ Model sẵn sàng: {model_name}" if model
            else f"📂 Ground truth mode — {len(IMAGE_INDEX)} ảnh từ dataset"
        ),
    }


@app.get("/samples")
def get_samples():
    """Trả về danh sách ảnh test từ dataset để frontend hiển thị."""
    samples = []
    if DATASET_DIR is None:
        return {"samples": samples}

    # Ưu tiên ảnh test, sau đó valid
    for split in ["test", "valid"]:
        img_dir = DATASET_DIR / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg"))[:20]:
            stem = img_path.stem
            entry = IMAGE_INDEX.get(stem, {})
            has_label = entry.get("label") is not None
            samples.append({
                "filename": img_path.name,
                "stem": stem,
                "split": split,
                "has_label": has_label,
            })

    return {"samples": samples, "count": len(samples)}


@app.get("/sample-image/{stem}")
async def get_sample_image(stem: str):
    """Trả về ảnh từ dataset dưới dạng base64."""
    entry = IMAGE_INDEX.get(stem)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Image not found: {stem}")

    img_path = entry["image"]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file missing: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail=f"Cannot read image: {img_path}")

    b64 = encode_image_base64(img_bgr)
    return JSONResponse({"image": b64, "filename": img_path.name, "stem": stem})


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.3),   # BƯỚC 19 notebook: conf=0.3
    sample: str = Form(""),
    model_key: str = Form("yolov8n"),  # Frontend model selector: yolov8n/s/m/l/x
):
    """
    Endpoint chính — bám sát notebook BƯỚC 19:
      results = model_best.predict(source=img_path, conf=0.3, verbose=False)
      res_plotted = results[0].plot()
      res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    Trả về:
      - annotated_image: base64 ảnh đã vẽ OBB
      - detections: JSON list cho sidebar

    Khi không có model YOLO:
      - Tìm file nhãn ground truth tương ứng từ dataset
      - Vẽ OBB theo đúng nhãn thật
    """
    img_bytes = await file.read()
    upload_filename = file.filename or ""

    # Log selected model (frontend selector)
    print(f"[INFO] Model selected: {model_key} | conf={conf} | sample='{sample}' | file='{upload_filename}'")

    # Resolve sample key to dataset stem if applicable
    resolved_sample = sample
    if sample in SAMPLE_KEY_MAP:
        resolved_sample = SAMPLE_KEY_MAP[sample]

    # Try to load image from bytes
    pil_img = None
    matched_label_path = None
    matched_image_path = None

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        pil_img = None

    # If uploaded image is invalid (dummy blob) -> try dataset or assets
    if pil_img is None or pil_img.size[0] < 10 or pil_img.size[1] < 10:
        # Try resolved sample key in dataset index
        if resolved_sample and resolved_sample in IMAGE_INDEX:
            entry = IMAGE_INDEX[resolved_sample]
            if entry["image"].exists():
                pil_img = Image.open(str(entry["image"])).convert("RGB")
                matched_label_path = entry["label"]
                matched_image_path = entry["image"]
                print(f"[INFO] Loaded dataset image: {resolved_sample}")

        # Fallback: try assets
        if pil_img is None or pil_img.size[0] < 10:
            _assets_dir = _WEBDEMO_DIR / "assets"
            sample_map = {
                "hotspot": "sample_hotspot.jpg",
                "diode":   "sample_diode.jpg",
                "multi":   "sample_multi.jpg",
            }
            fallback_key = sample if sample in sample_map else "hotspot"
            fallback_path = _assets_dir / sample_map.get(fallback_key, "sample_hotspot.jpg")
            if fallback_path.exists():
                pil_img = Image.open(str(fallback_path)).convert("RGB")
                print(f"[INFO] Fallback to asset: {fallback_path}")
            else:
                pil_img = Image.new("RGB", (512, 512), (40, 40, 40))
    else:
        # Valid uploaded image -> find matching label
        # Try sample key map, content hash, filename match
        matched_image_path, matched_label_path = find_matching_label(upload_filename, img_bytes)

        # Also try resolved sample key
        if not matched_label_path and resolved_sample and resolved_sample in IMAGE_INDEX:
            entry = IMAGE_INDEX[resolved_sample]
            matched_label_path = entry["label"]
            matched_image_path = entry["image"]

        if matched_label_path:
            print(f"[INFO] Matched label for '{upload_filename}': {matched_label_path}")
        else:
            print(f"[INFO] No matching label found for '{upload_filename}'")

    img_np  = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_w, img_h = pil_img.size   # (width, height)


    # ════════════════════════════════════════════════════════════
    # MODE 1: Real YOLO model
    # ════════════════════════════════════════════════════════════
    if model is not None:
        t0 = time.time()
        # Bám sát notebook: conf=0.3, verbose=False
        results = model.predict(source=img_bgr, conf=conf, verbose=False)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Lấy ảnh đã annotate từ YOLO plot() — giống notebook result.plot()
        annotated_bgr = None
        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Ảnh đã annotate bởi YOLO (giống notebook result.plot())
            annotated_bgr = result.plot()

            if result.obb is not None and len(result.obb) > 0:
                for i in range(len(result.obb)):
                    cls_id   = int(result.obb.cls[i].item())
                    conf_val = round(float(result.obb.conf[i].item()), 4)

                    # xyxyxyxy: 4 corner points (pixel) — khớp format BƯỚC 6
                    corners = result.obb.xyxyxyxy[i].tolist()  # [[x,y], [x,y], [x,y], [x,y]]

                    # Normalize về [0,1] cho frontend canvas (nếu cần)
                    pts_norm = []
                    for cx, cy in corners:
                        pts_norm.extend([
                            round(cx / img_w, 4),
                            round(cy / img_h, 4),
                        ])

                    angle, w_px, h_px = compute_obb_angle_size(corners)

                    if cls_id < len(model_class_names):
                        cls_name = model_class_names[cls_id]
                    elif cls_id < len(CLASS_NAMES_WEB):
                        cls_name = CLASS_NAMES_WEB[cls_id]
                    else:
                        cls_name = f"defect_{cls_id}"

                    detections.append({
                        "cls":       cls_name,
                        "cls_id":    cls_id,
                        "conf":      conf_val,
                        "pts":       pts_norm,
                        "angle":     angle,
                        "size":      [round(w_px, 1), round(h_px, 1)],
                        "color_hex": CLASS_COLORS_HEX[cls_id] if cls_id < 4 else "#00e5ff",
                    })

        # Encode ảnh annotated sang base64
        if annotated_bgr is None:
            annotated_bgr = img_bgr  # fallback: ảnh gốc nếu không có detection

        annotated_b64 = encode_image_base64(annotated_bgr)

        return JSONResponse({
            "mode":             "real",
            "model":            model_name,
            "model_key":        model_key,
            "time_ms":          elapsed_ms,
            "count":            len(detections),
            "detections":       detections,
            "annotated_image":  annotated_b64,  # ảnh đã vẽ OBB (chính xác như notebook)
        })


    # ════════════════════════════════════════════════════════════
    # MODE 2: Ground truth labels từ dataset
    # Đọc nhãn thật từ file .txt, vẽ OBB theo đúng notebook BƯỚC 6
    # ════════════════════════════════════════════════════════════
    t0 = time.time()

    # Đọc ground truth labels
    gt_detections = read_ground_truth_labels(matched_label_path, img_w, img_h)

    # Vẽ OBB lên ảnh (bám sát notebook BƯỚC 6)
    canvas = img_bgr.copy()
    detections_out = []

    for det in gt_detections:
        corners_pixel = det["corners_pixel"]
        cls_id = det["cls_id"]
        cls_name = det["cls"]

        # Vẽ OBB theo notebook BƯỚC 6
        draw_obb_notebook_style(canvas, corners_pixel, cls_id, cls_name)

        # Chuẩn bị output cho frontend (bỏ corners_pixel internal)
        det_out = {k: v for k, v in det.items() if k != "corners_pixel"}
        detections_out.append(det_out)

    elapsed_ms = int((time.time() - t0) * 1000)
    annotated_b64 = encode_image_base64(canvas)

    return JSONResponse({
        "mode":             "ground_truth",
        "model":            None,
        "model_key":        model_key,
        "time_ms":          elapsed_ms,
        "count":            len(detections_out),
        "detections":       detections_out,
        "annotated_image":  annotated_b64,
        "label_file":       str(matched_label_path) if matched_label_path else None,
        "note":             (
            f"Ground truth tu dataset ({len(detections_out)} objects)"
            if matched_label_path
            else "Khong tim thay nhan tuong ung — copy best.pt vao model/best.pt de dung model YOLO"
        ),
    })
