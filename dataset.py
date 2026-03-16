import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


# ============================================================
# 数据增强（训练用）
# ============================================================
def get_transform():
    return T.Compose([
        T.ToTensor()
    ])


# ============================================================
# collate_fn（Mask R-CNN 必需）
# ============================================================
def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# COCO 格式番茄数据集
# ============================================================
class COCOTomatoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transforms = transforms

        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # -------------------------
        # 读取图像
        # -------------------------
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # -------------------------
        # 解析标注
        # -------------------------
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            if "bbox" not in ann:
                continue

            # bbox
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # mask（RLE 解码）
            rle = ann["segmentation"]
            mask = mask_utils.decode(rle)
            masks.append(mask)

            labels.append(1)  # tomato 类别

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor([ann["area"] for ann in anns], dtype=torch.float32),
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
