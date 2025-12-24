import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
import os
from model_mamba_highlow_visual1 import MultiScaleMambaFusion
from dataset import MyFusionDataset
from pytorch_msssim import ms_ssim, ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ema import ExponentialMovingAverage


def downsample(gt, size):
    return nn.functional.interpolate(gt, size=size, mode="bilinear", align_corners=True)

def train(model, dataloader, criterion, optimizer, ema, device):
    model.train()
    epoch_loss = 0
    for pan, ms, gt, _ in tqdm(dataloader, desc="Training"):
        pan, ms, gt = pan.to(device), ms.to(device), gt.to(device)
        optimizer.zero_grad()

        out, outputs = model(pan, ms)

        gt_128 = downsample(gt, size=(128, 128))
        gt_64 = downsample(gt, size=(64, 64))

        loss_main = criterion(out, gt)
        loss_256 = criterion(outputs["out_256"], gt)
        loss_128 = criterion(outputs["out_128"], gt_128)
        loss_64 = criterion(outputs["out_64"], gt_64)

        # 添加颜色损失项（颜色直方图或均值对齐）
        color_loss = torch.mean((out.mean(dim=[2, 3]) - gt.mean(dim=[2, 3])) ** 2)

        total_loss = 5 * loss_main + 0.1 * loss_256 + 0.1 * loss_128 + 0.1 * loss_64 + color_loss + (1 - ssim(out, gt, data_range=1.0, size_average=True).to(device))
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update()

        epoch_loss += total_loss.item()
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, ema, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        with ema.average_parameters():
            for pan, ms, gt, _ in tqdm(dataloader, desc="Validation"):
                pan, ms, gt = pan.to(device), ms.to(device), gt.to(device)
                out, _ = model(pan, ms)
                # loss = criterion(out, gt) + (1 - ssim(out, gt, data_range=1.0, size_average=True).to(device)) + (torch.mean((out.mean(dim=[2, 3]) - gt.mean(dim=[2, 3])) ** 2))
                loss = criterion(out, gt)
                epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--root_dir', type=str, default=r"datasets/QB")
    args = parser.parse_args()

    device = torch.device(args.device)

    train_set = MyFusionDataset(split='train', root_dir=args.root_dir)
    val_set = MyFusionDataset(split='val', root_dir=args.root_dir)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiScaleMambaFusion().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, ema, device)
        val_loss = validate(model, val_loader, criterion, ema, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)
        if optimizer.param_groups[0]['lr'] <= 1e-8:
            optimizer.param_groups[0]['lr'] = 1e-8

        checkpoint_dir = "checkpoints/QB/epoch/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/QB/best_model.pth")
            print("Saved best model.")