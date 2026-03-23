import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import get_default_maskrcnn


def load_model(weight_path, device):
    """加载训练好的模型"""
    model = get_default_maskrcnn(num_classes=2)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, test_dir, coco_gt, device, score_thresh=0.5):
    """对测试集进行推理，返回 COCO 格式的预测结果"""
    results = []
    
    img_ids = coco_gt.getImgIds()
    
    for img_id in tqdm(img_ids, desc="Inferencing"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(test_dir, img_info["file_name"])
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
        
        image = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(image).to(device)
        
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        masks = outputs["masks"].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] < score_thresh:
                continue
            
            box = boxes[i]
            mask = masks[i, 0]
            
            # 转换 mask 为 RLE 格式
            from pycocotools import mask as mask_utils
            mask_bin = (mask > 0.5).astype('uint8')
            rle = mask_utils.encode(np.asfortranarray(mask_bin))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            results.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(scores[i]),
                "bbox": [float(box[0]), float(box[1]), 
                        float(box[2] - box[0]), float(box[3] - box[1])]
            })
    
    return results


def evaluate(gt_file, pred_results):
    """使用 COCO API 评估预测结果"""
    coco_gt = COCO(gt_file)
    
    if len(pred_results) == 0:
        print("No predictions to evaluate!")
        return None
    
    coco_dt = coco_gt.loadRes(pred_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取关键指标
    stats = {
        "AP@0.5:0.95": float(coco_eval.stats[0]),
        "AP@0.5": float(coco_eval.stats[1]),
        "AP@0.75": float(coco_eval.stats[2]),
        "AR@0.5:0.95 (maxDets=1)": float(coco_eval.stats[6]),
        "AR@0.5:0.95 (maxDets=10)": float(coco_eval.stats[7]),
        "AR@0.5:0.95 (maxDets=100)": float(coco_eval.stats[8])
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN on test set")
    parser.add_argument("--weight", type=str, 
                       default="outputs/checkpoints/model_epoch_5.pth",
                       help="Path to model weight file")
    parser.add_argument("--test_dir", type=str, 
                       default="data/test_images",
                       help="Directory containing test images")
    parser.add_argument("--gt_file", type=str,
                       default="data/annotations/test_annotations.json",
                       help="Ground truth annotation file (COCO format)")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                       help="Score threshold for predictions")
    parser.add_argument("--output", type=str,
                       default="outputs/evaluation_results.json",
                       help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.weight):
        print(f"Error: Model weight file not found: {args.weight}")
        return
    
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        return
    
    device = torch.device("cpu")
    print(f"Loading model from: {args.weight}")
    print(f"Using device: {device}\n")
    
    # 加载模型
    model = load_model(args.weight, device)
    
    # 加载真值标注
    coco_gt = COCO(args.gt_file)
    
    # 运行推理
    print("Running inference on test set...")
    pred_results = run_inference(model, args.test_dir, coco_gt, device, args.score_thresh)
    
    if len(pred_results) == 0:
        print("No predictions generated. Check your model and test data.")
        return
    
    # 评估
    print("\nEvaluating predictions...")
    stats = evaluate(args.gt_file, pred_results)
    
    if stats is None:
        return
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    import numpy as np
    main()
