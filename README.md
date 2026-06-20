# 🛸 Phát hiện khuyết tật tấm pin năng lượng mặt trời

**Thị Giác Máy Tính — YOLOv8 OBB (Oriented Bounding Box)**

Dự án phát hiện 4 loại khuyết tật trên tấm pin năng lượng mặt trời bằng
ảnh nhiệt hồng ngoại (thermal drone imagery), sử dụng YOLOv8 với
Oriented Bounding Box (OBB) để xử lý đối tượng nghiêng.

---

## 📁 Cấu trúc thư mục

```
DA_thi_giac_may_tinh/
│
├── 📄 README.md               ← File này
├── 📄 requirements.txt        ← Tất cả dependencies
├── 📄 data_obb.yaml           ← Config dataset cho YOLO
│
├── 📁 dataset/                ← Dataset (train/valid/test)
│   ├── train/images/, train/labels/
│   ├── valid/images/, valid/labels/
│   └── test/images/,  test/labels/
│
├── 📁 models/                 ← Model weights đã train
│   └── best.pt
│
├── 📁 notebooks/              ← Jupyter notebooks gốc
│   ├── Do_an_TGMT.ipynb
│   └── TGMT_Buoi1_den_Buoi4.ipynb
│
├── 📁 outputs/                ← Biểu đồ và ảnh đầu ra
│
├── 📁 runs/                   ← Kết quả training (tự tạo bởi YOLO)
│   └── obb/
│       ├── baseline_nano/
│       ├── exp2_small/
│       ├── exp3_augmented/
│       └── exp4_medium_best/
│
├── 📁 src/                    ← Source code Python
│   ├── config.py              ← Cấu hình chung
│   ├── buoi1_2_eda/           ← Buổi 1-2: EDA & chuẩn bị dữ liệu
│   ├── buoi3_training/        ← Buổi 3: Train Baseline
│   ├── buoi4_experiments/     ← Buổi 4: Các thực nghiệm
│   └── buoi5_evaluation/      ← Buổi 5: Đánh giá & Error Analysis
│
└── 📁 web-demo/               ← Web demo (FastAPI + Node.js)
    ├── api/app.py             ← FastAPI backend
    ├── index.html             ← Frontend
    └── server.js              ← Node proxy
```

---

## 🚀 Cài đặt & Chạy

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

---

## ▶️  Chạy nhanh (run_pipeline.py)

Script `run_pipeline.py` chạy toàn bộ dự án theo đúng thứ tự notebook —
dữ liệu được truyền giữa các bước như biến trong notebook cell.

```bash
# Chạy toàn bộ pipeline (EDA → train → evaluate)
python run_pipeline.py

# Xem danh sách các bước
python run_pipeline.py --list

# Chỉ chạy 1 bước
python run_pipeline.py --step eda
python run_pipeline.py --step train
python run_pipeline.py --step compare
python run_pipeline.py --step final

# Chạy từ bước chỉ định đến cuối
python run_pipeline.py --from compare

# Bỏ qua một số bước
python run_pipeline.py --skip gradcam exp2 exp3
```

**Chuỗi dữ liệu (như notebook):**
```
setup → eda → visualize → yaml → train → evaluate → curves
      → exp2 → exp3 → exp4 → compare
      → cm → predict → gradcam → final
```

---

## 📚 Chạy theo từng Buổi học (từng script riêng)

### Buổi 1-2: EDA & Chuẩn bị dữ liệu

```bash
# BƯỚC 0-2: Setup & giải nén dataset
python src/buoi1_2_eda/setup_dataset.py

# BƯỚC 3-5: Thống kê & biểu đồ phân bố lớp
python src/buoi1_2_eda/eda_statistics.py

# BƯỚC 6: Trực quan hóa ảnh + OBB labels
python src/buoi1_2_eda/visualize_obb.py

# BƯỚC 7: Tạo data_obb.yaml
python src/buoi1_2_eda/create_yaml.py
```

### Buổi 3: Train Baseline

```bash
# BƯỚC 8: Train YOLOv8n-OBB (Baseline)
python src/buoi3_training/train_baseline.py

# BƯỚC 9-10: Đánh giá & Phân tích lỗi
python src/buoi3_training/evaluate_baseline.py

# BƯỚC 11: Vẽ đồ thị Loss
python src/buoi3_training/plot_training_curves.py --run baseline_nano
```

### Buổi 4: Các thực nghiệm so sánh

```bash
# BƯỚC 12: Thực nghiệm 2 — YOLOv8s (Small)
python src/buoi4_experiments/exp2_small.py

# BƯỚC 13: Thực nghiệm 3 — YOLOv8n + Augmentation
python src/buoi4_experiments/exp3_augmented.py

# BƯỚC 14: Thực nghiệm 4 — YOLOv8m (Medium) — BEST
python src/buoi4_experiments/exp4_medium.py

# BƯỚC 15-17: So sánh kết quả & Biểu đồ
python src/buoi4_experiments/compare_experiments.py
```

### Buổi 5: Đánh giá chi tiết & Error Analysis

```bash
# BƯỚC 18: Confusion Matrix
python src/buoi5_evaluation/confusion_matrix.py

# BƯỚC 19: Dự đoán trên tập Test
python src/buoi5_evaluation/predict_test.py

# Dự đoán 1 ảnh đơn lẻ
python src/buoi5_evaluation/predict_test.py --image path/to/image.jpg

# Grad-CAM: vùng chú ý của mô hình
python src/buoi5_evaluation/gradcam.py

# BƯỚC 31: Đánh giá cuối — Baseline vs Best model trên Test set
python src/buoi5_evaluation/final_evaluation.py
```

---

## 🌐 Web Demo

### Khởi động Backend (FastAPI)

```bash
cd web-demo/api
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Khởi động Frontend (Node.js proxy)

```bash
cd web-demo
npm install
node server.js
```

Mở trình duyệt: **http://localhost:3000**

---

## 🎯 4 Lớp khuyết tật

| ID | Tên YAML    | Tên đầy đủ          | Màu    |
|----|-------------|---------------------|--------|
| 0  | defect_0    | Bypassed Substrings | Đỏ     |
| 1  | defect_1    | Diode               | Xanh lá|
| 2  | defect_2    | Multi Hot Spots     | Xanh dương|
| 3  | defect_3    | Hotspot             | Vàng   |

---

## 📊 Kết quả thực nghiệm

| Thực nghiệm       | Model     | mAP50 | Precision | Recall |
|-------------------|-----------|-------|-----------|--------|
| Exp1 (Baseline)   | YOLOv8n   | -     | -         | -      |
| Exp2              | YOLOv8s   | -     | -         | -      |
| Exp3              | YOLOv8n+Aug| -    | -         | -      |
| Exp4 ⭐ (Best)    | YOLOv8m   | -     | -         | -      |

> **Mục tiêu:** Precision ≥ 0.85 & Recall ≥ 0.85
