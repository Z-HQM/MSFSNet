import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
import h5py
import numpy as np

class MyFusionDataset(Dataset):
    def __init__(self, split='train', root_dir=r"E:\\zhqm\\datasets\\QB", img_size=256):
        super().__init__()
        self.split = split
        self.pan_dir = os.path.join(root_dir, split, 'pan')
        self.ms_dir = os.path.join(root_dir, split, 'ms')
        self.label_dir = os.path.join(root_dir, split, 'label')
        self.file_list = sorted(os.listdir(self.pan_dir))

        self.pan_paths = sorted(glob.glob(os.path.join(self.pan_dir, '*')))
        self.ms_paths = sorted(glob.glob(os.path.join(self.ms_dir, '*')))
        self.label_paths = sorted(glob.glob(os.path.join(self.label_dir, '*')))

        assert len(self.pan_paths) == len(self.ms_paths) == len(self.label_paths), "Mismatch in dataset lengths"

        self.to_tensor = transforms.ToTensor()
        self.resize_pan_label = transforms.Resize((256, 256))
        self.resize_ms = transforms.Resize((64, 64))

    def __len__(self):
        return len(self.pan_paths)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        pan_img = Image.open(self.pan_paths[idx]).convert('RGB')
        ms_img = Image.open(self.ms_paths[idx]).convert('RGB')
        label_img = Image.open(self.label_paths[idx]).convert('RGB')

        pan = self.to_tensor(self.resize_pan_label(pan_img))      # (3, 256, 256)
        ms = self.to_tensor(self.resize_ms(ms_img))               # (3, 64, 64)
        label = self.to_tensor(self.resize_pan_label(label_img))  # (3, 256, 256)

        return pan, ms, label, name

class MyFusionDataset_mi(Dataset):
    def __init__(self, split='train', root_dir=r"E:\\zhqm\\datasets\\CT-MRI", img_size=256,  vi='CT', mri='MRI'):
        super().__init__()
        self.split = split
        self.pan_dir = os.path.join(root_dir, split, mri)
        self.ms_dir = os.path.join(root_dir, split, vi)

        self.file_list = sorted(os.listdir(self.pan_dir))

        self.pan_paths = sorted(glob.glob(os.path.join(self.pan_dir, '*')))
        self.ms_paths = sorted(glob.glob(os.path.join(self.ms_dir, '*')))

        assert len(self.pan_paths) == len(self.ms_paths), "Mismatch in dataset lengths"

        self.to_tensor = transforms.ToTensor()
        # self.resize_pan_label = transforms.Resize((256, 256))
        # self.resize_ms = transforms.Resize((64, 64))

    def __len__(self):
        return len(self.pan_paths)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        pan_img = Image.open(self.pan_paths[idx]).convert('RGB')
        ms_img = Image.open(self.ms_paths[idx]).convert('RGB')

        pan = self.to_tensor(pan_img)     # (3, 256, 256)
        ms = self.to_tensor(ms_img)              # (3, 64, 64)

        return pan, ms, name

class MyFusionDataset_iv(Dataset):
    def __init__(self, split='train', root_dir=r"E:\\zhqm\\datasets\\CT-MRI", img_size=256,  vi='CT', mri='MRI'):
        super().__init__()
        self.split = split
        self.pan_dir = os.path.join(root_dir, mri)
        self.ms_dir = os.path.join(root_dir, vi)

        self.file_list = sorted(os.listdir(self.pan_dir))

        self.pan_paths = sorted(glob.glob(os.path.join(self.pan_dir, '*')))
        self.ms_paths = sorted(glob.glob(os.path.join(self.ms_dir, '*')))

        assert len(self.pan_paths) == len(self.ms_paths), "Mismatch in dataset lengths"

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pan_paths)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        pan_img = Image.open(self.pan_paths[idx]).convert('RGB')
        ms_img = Image.open(self.ms_paths[idx]).convert('RGB')

        pan = self.to_tensor(pan_img)     # (3, 256, 256)
        ms = self.to_tensor(ms_img)              # (3, 64, 64)

        return pan, ms, name

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)