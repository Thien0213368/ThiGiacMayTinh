"""Test FastAPI v4 endpoints"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import requests, json, base64

BASE = "http://localhost:8003"

# Test 1: Status
print("=== /status ===")
resp = requests.get(f"{BASE}/status")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

# Test 2: Samples
print("\n=== /samples ===")
resp = requests.get(f"{BASE}/samples")
data = resp.json()
print(f"Total: {data['count']} samples")
for s in data["samples"][:5]:
    print(f"  [{s['split']}] {s['filename'][:60]}  label={s['has_label']}")

# Test 3: Detect with a real test image (should find ground truth labels)
print("\n=== /detect (real test image) ===")
img_path = r"d:\DA_thi_giac_may_tinh\ThiGiacMayTinh2.v6i.yolov8-obb\test\images\DJI_20260411084050_0371_T_JPG.rf.9f00bf40ead302a42808995560e07c2c.jpg"
img_bytes = open(img_path, "rb").read()
resp = requests.post(
    f"{BASE}/detect",
    files={"file": ("DJI_20260411084050_0371_T_JPG.rf.9f00bf40ead302a42808995560e07c2c.jpg", img_bytes, "image/jpeg")},
    data={"conf": "0.3"},
)
data = resp.json()
print(f"mode: {data['mode']}")
print(f"count: {data['count']}")
print(f"label_file: {data.get('label_file', 'N/A')}")
print(f"note: {data.get('note', 'N/A')}")
print(f"annotated_image length: {len(data.get('annotated_image', ''))}")
for d in data["detections"]:
    print(f"  {d['cls']:20s} conf={d['conf']:.2f}  angle={d['angle']}")

# Save annotated image
if data.get("annotated_image"):
    raw = base64.b64decode(data["annotated_image"].split(",")[1])
    out_path = r"d:\DA_thi_giac_may_tinh\web-demo\assets\test_gt_annotated.jpg"
    open(out_path, "wb").write(raw)
    print(f"\nSaved annotated image: {out_path} ({len(raw)} bytes)")

# Test 4: Detect with random image (no matching label)
print("\n=== /detect (random name, no label) ===")
resp = requests.post(
    f"{BASE}/detect",
    files={"file": ("random_photo.jpg", img_bytes, "image/jpeg")},
    data={"conf": "0.3"},
)
data = resp.json()
print(f"mode: {data['mode']}")
print(f"count: {data['count']}")
print(f"note: {data.get('note', 'N/A')}")
