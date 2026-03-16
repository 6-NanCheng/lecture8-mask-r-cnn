import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# 默认 Mask R-CNN（28×28 mask head）
# ============================================================
def get_default_maskrcnn(num_classes=2):
    # 加载预训练 Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
