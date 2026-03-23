import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import COCOTomatoDataset, collate_fn, get_transform
from model import get_default_maskrcnn


# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 自动选择最佳可用设备
# ============================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ============================================================
# 训练脚本（自适应 CUDA / MPS / CPU）
# ============================================================
def train(
    img_dir,
    ann_file,
    num_classes=2,
    num_epochs=5,
    batch_size=1,
    lr=0.005,
):

    device = get_device()
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 针对不同设备的优化配置
    # ---------------------------------------------------------
    use_amp = False          # 是否使用自动混合精度
    num_workers = 0          # DataLoader 工作进程数

    if device.type == "cuda":
        # CUDA: 启用 cudnn benchmark + 混合精度 (AMP)
        torch.backends.cudnn.benchmark = True
        use_amp = True
        num_workers = min(4, os.cpu_count() or 0)
        print("  → CUDA 优化: cudnn.benchmark=True, AMP=True, "
              f"num_workers={num_workers}")

    else:
        num_workers = min(2, os.cpu_count() or 0)
        print(f"  → CPU 模式, num_workers={num_workers}")

    set_seed(42)

    # -------------------------
    # 数据集（全部用于训练）
    # -------------------------
    dataset = COCOTomatoDataset(img_dir, ann_file, transforms=get_transform())

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------
    # 模型（默认 mask head）
    # -------------------------
    model = get_default_maskrcnn(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # CUDA AMP: GradScaler + autocast
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -------------------------
    # 训练记录
    # -------------------------
    iter_loss_history = []

    # 五个损失项
    loss_classifier_history = []
    loss_box_reg_history = []
    loss_mask_history = []
    loss_objectness_history = []
    loss_rpn_box_reg_history = []

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if use_amp:
                # CUDA 混合精度训练
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # 总损失
            iter_loss_history.append(losses.item())

            # 五个损失项
            loss_classifier_history.append(loss_dict["loss_classifier"].item())
            loss_box_reg_history.append(loss_dict["loss_box_reg"].item())
            loss_mask_history.append(loss_dict["loss_mask"].item())
            loss_objectness_history.append(loss_dict["loss_objectness"].item())
            loss_rpn_box_reg_history.append(loss_dict["loss_rpn_box_reg"].item())

        # 保存模型
        torch.save(model.state_dict(), f"outputs/checkpoints/model_epoch_{epoch+1}.pth")

    # ============================================================
    # ① loss_total_iter.png（总损失曲线）
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.plot(iter_loss_history, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Total Loss per Iteration")
    plt.grid(True)
    plt.legend()
    plt.savefig("outputs/plots/loss_total_iter.png", dpi=300)
    plt.close()

    # ============================================================
    # ② loss_components_iter.png（五个损失项曲线）
    # ============================================================
    plt.figure(figsize=(12, 7))
    plt.plot(loss_classifier_history, label="loss_classifier")
    plt.plot(loss_box_reg_history, label="loss_box_reg")
    plt.plot(loss_mask_history, label="loss_mask")
    plt.plot(loss_objectness_history, label="loss_objectness")
    plt.plot(loss_rpn_box_reg_history, label="loss_rpn_box_reg")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Components per Iteration")
    plt.grid(True)
    plt.legend()
    plt.savefig("outputs/plots/loss_components_iter.png", dpi=300)
    plt.close()

    print("\n训练完成！图像已保存到 outputs/plots/")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    train(
        img_dir="data/images",
        ann_file="data/annotations/annotations.json",
        num_classes=2,
        num_epochs=5,
        batch_size=1,
        lr=0.005,
    )
