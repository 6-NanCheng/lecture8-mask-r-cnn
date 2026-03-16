"""
convert_cvat_to_coco.py

将 CVAT 导出的 merged_raw.json 转换为 COCO RLE 格式的 annotations.json。

学生只需要修改下面的两个路径：
    INPUT_JSON  = "data/user_data/merged_raw.json"
    OUTPUT_JSON = "data/user_data/annotations.json"

然后运行：
    python src/convert_cvat_to_coco.py
"""

import os
import json
import numpy as np
from pycocotools import mask as mask_utils


# ============================================================
# 1. 学生只需修改这两行路径
# ============================================================
INPUT_JSON  = "data/annotations/merged_raw.json"
OUTPUT_JSON = "data/annotations/annotations.json"


# ============================================================
# 2. 转换函数（核心逻辑）
# ============================================================
def convert_cvat_to_coco(input_json, output_json):

    if not os.path.exists(input_json):
        raise FileNotFoundError(f"输入文件不存在：{input_json}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("=== Step 1: 读取 merged_raw.json ===")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "tomato", "supercategory": "plant"}
        ]
    }

    ann_id = 1
    img_id = 1

    print("=== Step 2: 转换为 COCO RLE ===")

    for item in data["items"]:
        img = item["image"]

        coco["images"].append({
            "id": img_id,
            "file_name": img["path"],
            "width": img["size"][0],
            "height": img["size"][1]
        })

        if "annotations" in item:
            for ann in item["annotations"]:
                if ann["type"] != "mask":
                    continue

                rle = ann["rle"]

                coco_ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": {
                        "counts": rle["counts"],
                        "size": rle["size"]
                    },
                    "iscrowd": 0
                }

                # 解码 mask 计算 bbox
                mask = mask_utils.decode(rle)
                ys, xs = np.where(mask == 1)

                if len(xs) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    w = x_max - x_min + 1
                    h = y_max - y_min + 1
                    coco_ann["bbox"] = [int(x_min), int(y_min), int(w), int(h)]
                    coco_ann["area"] = int(mask.sum())
                else:
                    coco_ann["bbox"] = [0, 0, 0, 0]
                    coco_ann["area"] = 0

                coco["annotations"].append(coco_ann)
                ann_id += 1

        img_id += 1

    print("=== Step 3: 写入 COCO JSON ===")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"转换完成！输出文件：{output_json}")


# ============================================================
# 3. 主入口（无需命令行参数）
# ============================================================
if __name__ == "__main__":
    convert_cvat_to_coco(INPUT_JSON, OUTPUT_JSON)
