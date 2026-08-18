import os
import json
import torch
import rasterio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

class Sen1Floods11DataModule(LightningDataModule):
    def __init__(self, model: str,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 **kwargs):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        self.root = self.catalog["sen1floods11"]["path"]

        # Means and stds from user
        s2_mean = [1626.91600224, 1396.03470631, 1364.06118417, 1218.22847919, 1466.07290663, 2386.90297537, 2845.61256277, 2622.95796892, 3077.48221481, 486.87436782, 63.77861008, 2030.64763024, 1179.16607221]
        s2_std = [700.17133846, 739.09452682, 735.2482388, 864.936695, 776.8803358, 921.36834309, 1084.37346097, 1022.63418007, 1196.44255318, 336.61105431, 143.99923282, 980.87061347, 764.60836557]
        s1_mean = [-10.184408, -16.895273]
        s1_std = [4.255339, 5.290568]

        self.s1_normalize = v2.Normalize(mean=s1_mean, std=s1_std)
        self.s2_normalize = v2.Normalize(mean=s2_mean, std=s2_std)
        
        self.spatial_augment = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip()
        ])
        
        self.pixel_augment = v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=(11,11))], p=0.8),
        ])
        
        # Test augment without random elements
        self.spatial_transform = v2.Compose([
            v2.Resize(224, antialias=True)
        ])

        self.setup()

    def setup(self, stage=None):
        if self.model == "Sup":
            self.train_dataset = Sen1Floods11DatasetSup(self.root, split="train", 
                                                        spatial_transform=self.spatial_augment, 
                                                        pixel_transform=self.pixel_augment,
                                                        s1_normalize=self.s1_normalize, 
                                                        s2_normalize=self.s2_normalize)
            self.val_dataset = Sen1Floods11DatasetSup(self.root, split="test", 
                                                      spatial_transform=self.spatial_transform, 
                                                      pixel_transform=None,
                                                      s1_normalize=self.s1_normalize, 
                                                      s2_normalize=self.s2_normalize)
        elif self.model == "CoMM" or self.model == "WoMM":
            self.train_dataset = Sen1Floods11DatasetMMSSL(self.root, split="train", 
                                                          spatial_transform=self.spatial_augment, 
                                                          pixel_transform=self.pixel_augment,
                                                          s1_normalize=self.s1_normalize, 
                                                          s2_normalize=self.s2_normalize)
            self.val_dataset = Sen1Floods11DatasetMMSSL(self.root, split="test", 
                                                        spatial_transform=self.spatial_transform, 
                                                        pixel_transform=None,
                                                        s1_normalize=self.s1_normalize, 
                                                        s2_normalize=self.s2_normalize)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, pin_memory=True)

class Sen1Floods11DatasetBase(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        
        split_mapping = {"train": "train", "val": "valid", "test": "test"}
        mapped_split = split_mapping.get(split, split)
        
        split_file = os.path.join(root, "v1.1", f"splits/flood_handlabeled/flood_{mapped_split}_data.csv")
        data_root = os.path.join(root, "v1.1", "data/flood_events/HandLabeled/")
        
        with open(split_file) as f:
            file_list = f.readlines()
        file_list = [f.rstrip().split(",") for f in file_list]
        
        self.s2_image_list = [os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand")) for f in file_list]
        self.s1_image_list = [os.path.join(data_root, "S1Hand", f[0]) for f in file_list]
        self.target_list = [os.path.join(data_root, "LabelHand", f[1]) for f in file_list]

    def __len__(self):
        return len(self.s2_image_list)

    def get_raw_item(self, idx):
        with rasterio.open(self.s2_image_list[idx]) as src:
            s2_image = src.read()
        with rasterio.open(self.s1_image_list[idx]) as src:
            s1_image = src.read()
        with rasterio.open(self.target_list[idx]) as src:
            target = src.read(1)
            
        # Add channel dim so torchvision transforms apply properly
        target = torch.from_numpy(target).unsqueeze(0).float()
        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()
        
        # Replace NaN values which are present in some S1 images
        s1_image = torch.nan_to_num(s1_image, nan=0.0)
        s2_image = torch.nan_to_num(s2_image, nan=0.0)
        
        return s1_image, s2_image, target

class Sen1Floods11DatasetSup(Sen1Floods11DatasetBase):
    def __init__(self, root, split="train", spatial_transform=None, pixel_transform=None, s1_normalize=None, s2_normalize=None):
        super().__init__(root, split=split)
        self.spatial_transform = spatial_transform
        self.pixel_transform = pixel_transform
        self.s1_normalize = s1_normalize
        self.s2_normalize = s2_normalize

    def __getitem__(self, idx):
        s1, s2, target = self.get_raw_item(idx)
        
        # 1. Spatial Transforms (applied to all)
        if self.spatial_transform is not None:
            from torchvision import tv_tensors
            s1_tv = tv_tensors.Image(s1)
            s2_tv = tv_tensors.Image(s2)
            target_tv = tv_tensors.Mask(target)
            
            s1_aug1, s2_aug1, t_aug1 = self.spatial_transform(s1_tv, s2_tv, target_tv)
            
            s1_aug1 = s1_aug1.as_subclass(torch.Tensor)
            s2_aug1 = s2_aug1.as_subclass(torch.Tensor)
            t_aug1 = t_aug1.as_subclass(torch.Tensor)
        else:
            s1_aug1, s2_aug1, t_aug1 = s1, s2, target
            
        # 2. Pixel-level Transforms (applied only to inputs)
        if self.pixel_transform is not None:
            s1_aug1 = self.pixel_transform(s1_aug1)
            s2_aug1 = self.pixel_transform(s2_aug1)
            
        # 3. Normalization (applied at the end)
        if self.s1_normalize is not None:
            s1_aug1 = self.s1_normalize(s1_aug1)
        if self.s2_normalize is not None:
            s2_aug1 = self.s2_normalize(s2_aug1)
            
        return [s1_aug1, s2_aug1], t_aug1.squeeze(0).long()

class Sen1Floods11DatasetMMSSL(Sen1Floods11DatasetBase):
    def __init__(self, root, split="train", spatial_transform=None, pixel_transform=None, s1_normalize=None, s2_normalize=None):
        super().__init__(root, split=split)
        self.spatial_transform = spatial_transform
        self.pixel_transform = pixel_transform
        self.s1_normalize = s1_normalize
        self.s2_normalize = s2_normalize

    def __getitem__(self, idx):
        s1, s2, target = self.get_raw_item(idx)
        
        # 1. Spatial Transforms (applied to all)
        if self.spatial_transform is not None:
            from torchvision import tv_tensors
            s1_tv = tv_tensors.Image(s1)
            s2_tv = tv_tensors.Image(s2)
            target_tv = tv_tensors.Mask(target)
            
            s1_aug1, s2_aug1, _ = self.spatial_transform(s1_tv, s2_tv, target_tv)
            s1_aug2, s2_aug2, _ = self.spatial_transform(s1_tv, s2_tv, target_tv)
            
            s1_aug1 = s1_aug1.as_subclass(torch.Tensor)
            s2_aug1 = s2_aug1.as_subclass(torch.Tensor)
            s1_aug2 = s1_aug2.as_subclass(torch.Tensor)
            s2_aug2 = s2_aug2.as_subclass(torch.Tensor)
        else:
            s1_aug1 = s1_aug2 = s1
            s2_aug1 = s2_aug2 = s2
            
        # 2. Pixel-level Transforms (applied only to inputs)
        if self.pixel_transform is not None:
            s1_aug1 = self.pixel_transform(s1_aug1)
            s2_aug1 = self.pixel_transform(s2_aug1)
            s1_aug2 = self.pixel_transform(s1_aug2)
            s2_aug2 = self.pixel_transform(s2_aug2)
            
        # 3. Normalization (applied at the end)
        if self.s1_normalize is not None:
            s1_aug1 = self.s1_normalize(s1_aug1)
            s1_aug2 = self.s1_normalize(s1_aug2)
        if self.s2_normalize is not None:
            s2_aug1 = self.s2_normalize(s2_aug1)
            s2_aug2 = self.s2_normalize(s2_aug2)
            
        return [s1_aug1, s2_aug1], [s1_aug2, s2_aug2]
