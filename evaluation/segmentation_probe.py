import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Callback, Trainer, LightningModule, LightningDataModule
from typing import List, Optional
from tqdm import tqdm

class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=1, padding=0)
        )
    
    def forward(self, x, target_size):
        x = self.decoder(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class SegmentationProbingCallback(Callback):
    def __init__(self, downstream_data_modules: List[LightningDataModule],
                 names: Optional[List[str]] = None,
                 epochs: int = 10,
                 lr: float = 1e-3,
                 num_classes: int = 2,
                 ignore_index: int = -1,
                 **extraction_kwargs):
        self.downstream_data_modules = downstream_data_modules
        self.names = names
        self.epochs = epochs
        self.lr = lr
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.extraction_kwargs = extraction_kwargs
        if self.names is None:
            self.names = [d.__class__.__name__ for d in downstream_data_modules]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.segmentation_probing(trainer, pl_module)
            
    def segmentation_probing(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.global_rank == 0:
            device = pl_module.device
            
            for downstream_data_mod, dataset_name in zip(self.downstream_data_modules, self.names):
                train_loader = downstream_data_mod.train_dataloader()
                val_loader = downstream_data_mod.val_dataloader()
                pl_module.eval()
                with torch.no_grad():
                    sample_x, sample_y = next(iter(train_loader))
                    sample_x = [x.to(device) for x in sample_x]
                    sample_feat = pl_module.encoder(sample_x, **self.extraction_kwargs, return_tokens=True)
                    if isinstance(sample_feat, list):
                        sample_feat = sample_feat[0]
                
                # Determine number of modalities being fused to correctly find the spatial L
                mask_mod = self.extraction_kwargs.get("mask_modalities", None)
                if mask_mod is not None:
                    num_mods_active = sum(mask_mod[0]) if isinstance(mask_mod[0], list) else sum(mask_mod)
                else:
                    num_mods_active = pl_module.encoder.num_modalities
                
                # sample_feat is (B, L_total, D). We assume spatial is square per modality.
                B, L_total, D = sample_feat.shape
                L_spatial = L_total // num_mods_active
                H = int(np.sqrt(L_spatial))
                in_channels = D
                
                decoder = SimpleSegmentationDecoder(in_channels, self.num_classes).to(device)
                optimizer = torch.optim.Adam(decoder.parameters(), lr=self.lr)
                criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                
                # Train Loop
                decoder.train()
                with torch.enable_grad():
                    for epoch in range(self.epochs):
                        epoch_loss = 0.0
                        for X_batch, y_batch in train_loader:
                            X_batch = [x.to(device) for x in X_batch]
                            y_batch = y_batch.to(device)
                            
                            with torch.no_grad():
                                features = pl_module.encoder(X_batch, **self.extraction_kwargs, return_tokens=True)
                                if isinstance(features, list):
                                    features = features[0]
                        
                            # Handle concatenated modalities from MMFusion
                            B_feat, L_feat, D_feat = features.shape
                            if L_feat > L_spatial:
                                num_mods = L_feat // L_spatial
                                features = features.view(B_feat, num_mods, L_spatial, D_feat).mean(dim=1)
                            
                            features = features.transpose(1, 2).reshape(-1, in_channels, H, H)
                            
                            optimizer.zero_grad()
                            preds = decoder(features, y_batch.shape[-2:])
                            loss = criterion(preds, y_batch)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                
                # Validation Loop
                decoder.eval()
                intersection, union, target_area = 0, 0, 0
                correct, total = 0, 0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = [x.to(device) for x in X_batch]
                        y_batch = y_batch.to(device)
                        
                        features = pl_module.encoder(X_batch, **self.extraction_kwargs, return_tokens=True)
                        if isinstance(features, list):
                            features = features[0]
                            
                        # Handle concatenated modalities from MMFusion
                        B_feat, L_feat, D_feat = features.shape
                        if L_feat > L_spatial:
                            num_mods = L_feat // L_spatial
                            features = features.view(B_feat, num_mods, L_spatial, D_feat).mean(dim=1)
                            
                        features = features.transpose(1, 2).reshape(-1, in_channels, H, H)
                        
                        preds = decoder(features, y_batch.shape[-2:])
                        pred_labels = preds.argmax(dim=1)
                        
                        valid_mask = (y_batch != self.ignore_index)
                        pred_labels = pred_labels[valid_mask]
                        y_batch_valid = y_batch[valid_mask]
                        
                        correct += (pred_labels == y_batch_valid).sum().item()
                        total += valid_mask.sum().item()
                        
                        # mIoU computation for class 1 (Water)
                        pred_water = (pred_labels == 1)
                        target_water = (y_batch_valid == 1)
                        
                        intersection += (pred_water & target_water).sum().item()
                        union += (pred_water | target_water).sum().item()
                
                acc = correct / total if total > 0 else 0
                iou = intersection / union if union > 0 else 0
                print(f"Segmentation Probe ({dataset_name}) - Acc: {acc:.4f}, IoU (Water): {iou:.4f}")
            
            pl_module.train()
