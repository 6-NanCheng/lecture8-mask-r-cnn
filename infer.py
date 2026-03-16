import os
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


# ============================================================
# 构建与训练一致的 Mask R-CNN（默认 28×28 mask head）
# ============================================================
def get_default_maskrcnn(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 替换分类头（与训练一致）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ============================================================
# 加载模型（CPU-only）
# ============================================================
def load_model(weight_path):
    device = torch.device("cpu")  # 强制 CPU
    print("Using device:", device)

    model = get_default_maskrcnn(num_classes=2)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


# ============================================================
# 可视化并保存结果
# ============================================================
def visualize_and_save(image_pil, outputs, save_path, score_thresh=0.5):
    img = np.array(image_pil)[:, :, ::-1].copy()  # RGB->BGR

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    masks = outputs["masks"].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < score_thresh:
            continue

        box = boxes[i].astype(int)
        mask = masks[i, 0]
        mask_bin = (mask > 0.5).astype(np.uint8)

        area = int(mask_bin.sum())

        # 绘制 mask
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=-1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # 绘制 bbox
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        text = f"{scores[i]:.2f}, A={area}"
        cv2.putText(img, text, (box[0], max(0, box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(save_path, img)


# ============================================================
# 主程序（CPU-only）
# ============================================================
def main():
    # 学生只需修改这两行
    weight_path = "outputs/checkpoints/model_epoch_5.pth"
    test_dir = "data/test_images"

    save_dir = "outputs/infer_results"
    os.makedirs(save_dir, exist_ok=True)

    model, device = load_model(weight_path)

    # 支持的图片格式
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    img_files = [f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in exts]

    if not img_files:
        print("测试文件夹中没有找到图片。")
        return

    for fname in tqdm(img_files, desc="Inferencing"):
        img_path = os.path.join(test_dir, fname)
        image = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        save_path = os.path.join(save_dir, f"pred_{fname}")
        visualize_and_save(image, outputs, save_path, score_thresh=0.5)

    print(f"推理完成，结果已保存到：{save_dir}")


if __name__ == "__main__":
    main()
