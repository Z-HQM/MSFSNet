import torch
import os
from model_mamba_highlow_visual1 import MultiScaleMambaFusion
from dataset import MyFusionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
import time
@torch.no_grad()
def tt(model, dataloader, device, save_dir):
    model.eval()
    st=time.time()
    os.makedirs(save_dir, exist_ok=True)
    for idx, (pan, ms, gt, name) in enumerate(tqdm(dataloader, desc="Testing")):
        pan, ms = pan.to(device), ms.to(device)
        out, _ = model(pan, ms)
        out = out.clamp(0, 1)
        # Save result image
        filename = os.path.splitext(os.path.basename(name[0]))[0]
        vutils.save_image(out, os.path.join(save_dir, f"{filename}.png"))
        # vutils.save_image(gt, os.path.join(save_dir, f"gt_{idx}.png"))
    ed=time.time()
    print(f"Testing time: {(ed-st)/23}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--root_dir', type=str, default=r"datasets/Maryland")
    parser.add_argument('--model_path', type=str, default="checkpoints/Maryland/best_model.pth")  #best_model.pth  #epoch/epoch_10.pth
    parser.add_argument('--save_dir', type=str, default="test_results/Maryland")
    args = parser.parse_args()

    device = torch.device(args.device)

    test_set = MyFusionDataset(split='test', root_dir=args.root_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiScaleMambaFusion().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))


    tt(model, test_loader, device, args.save_dir)


