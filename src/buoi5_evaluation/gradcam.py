"""
gradcam.py — Buổi 5: Grad-CAM Visualization cho YOLOv8 OBB
Trực quan hóa vùng chú ý của mô hình (heatmap)

Grad-CAM giúp chúng ta thấy được "vùng chú ý" của mô hình khi đưa ra
quyết định phát hiện khuyết tật → giải thích được AI.

Yêu cầu: pip install grad-cam
"""

import sys
from pathlib import Path
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT, DATASET_DIR, OUTPUTS_DIR


class YOLOv8OBBWrapper:
    """
    Wrapper bọc YOLOv8 OBB để tương thích với pytorch-grad-cam.
    Grad-CAM yêu cầu một scalar output để tính gradient.
    """
    def __init__(self, model):
        import torch
        import torch.nn as nn
        self.model = model
        self._nn = nn

    def __call__(self, x):
        results = self.model(x)
        output = results[0] if isinstance(results, (tuple, list)) else results
        return output.max().unsqueeze(0).unsqueeze(0)


def get_gradcam_obb(model_path: str, img_path: str,
                    target_layer_index: int = -4) -> np.ndarray:
    """
    Tạo Grad-CAM heatmap cho một ảnh.

    Args:
        model_path: Đường dẫn best.pt
        img_path: Đường dẫn ảnh đầu vào
        target_layer_index: Index của lớp backbone/neck cần visualize

    Returns:
        cam_image: Ảnh RGB với heatmap overlay
    """
    import torch
    from ultralytics import YOLO
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    yolo = YOLO(model_path)
    inner_model = yolo.model

    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    rgb_img = np.float32(img) / 255

    # Lớp mục tiêu: thường là lớp Conv cuối cùng của backbone hoặc neck
    target_layers = [inner_model.model[target_layer_index]]

    wrapped_model = inner_model.eval()
    cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)

    device = next(inner_model.parameters()).device
    input_tensor = (
        torch.from_numpy(rgb_img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    input_tensor.requires_grad = True

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image


def visualize_gradcam(model_path: str = None, dataset_dir: str = None,
                       n_images: int = 4, save_path: str = None):
    """
    Thực hiện Grad-CAM trên n_images mẫu từ tập Test.

    Args:
        model_path: Đường dẫn best.pt
        dataset_dir: Đường dẫn dataset
        n_images: Số ảnh visualize
        save_path: Đường dẫn lưu ảnh output
    """
    # Kiểm tra thư viện
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
    except ImportError:
        print("❌ Thiếu thư viện grad-cam. Cài đặt:")
        print("   pip install grad-cam")
        return

    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)
    if model_path is None:
        candidates = [
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "runs" / "obb" / "exp4_medium_best" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if model_path is None:
        print("❌ Không tìm thấy model!")
        return

    test_img_paths = glob(f"{dataset_dir}/test/images/*.jpg")[:n_images]
    if not test_img_paths:
        print(f"❌ Không tìm thấy ảnh test trong {dataset_dir}/test/images/")
        return

    print(f"🔥 Tạo Grad-CAM cho {len(test_img_paths)} ảnh test")
    print(f"   Model: {model_path}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Grad-CAM Visualization — Vùng chú ý của mô hình trên tập Test',
                 fontsize=16)

    for i, img_p in enumerate(test_img_paths):
        ax = axes.flatten()[i]
        try:
            heatmap = get_gradcam_obb(model_path, img_p)
            ax.imshow(heatmap)
            ax.set_title(f"Ảnh: {Path(img_p).name}", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Lỗi: {e}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='red')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = str(OUTPUTS_DIR / "gradcam_visualization.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Đã lưu: {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("BUỔI 5: Grad-CAM Visualization")
    print("=" * 50)
    visualize_gradcam()
