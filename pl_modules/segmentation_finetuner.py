import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# Re-use the SimpleSegmentationDecoder from segmentation_probe
class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
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

class SegmentationFineTuner(pl.LightningModule):
    def __init__(self, encoder, learning_rate: float = 1e-4, num_classes: int = 2, ignore_index: int = -1, mask_modalities=None, extraction_kwargs=None):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mask_modalities = mask_modalities if mask_modalities is not None else [[True, True]]
        self.extraction_kwargs = extraction_kwargs if extraction_kwargs is not None else {}
        self.extraction_kwargs['mask_modalities'] = self.mask_modalities
        
        # We need to dynamically figure out in_channels based on a dummy pass or known properties.
        # But we'll initialize the decoder lazily during setup or just assume embed_dim.
        # For simplicity, we assume we know embed_dim.
        if hasattr(self.encoder, 'embed_dim'):
            self.in_channels = self.encoder.embed_dim
        else:
            self.in_channels = 512 # Fallback
            
        self.decoder = SimpleSegmentationDecoder(self.in_channels, self.num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
    def forward(self, x_batch, target_size):
        features = self.encoder(x_batch, **self.extraction_kwargs, return_tokens=True)
        if isinstance(features, list):
            features = features[0]
            
        mask_mod = self.mask_modalities
        if mask_mod is not None:
            num_mods_active = sum(mask_mod[0]) if isinstance(mask_mod[0], list) else sum(mask_mod)
        else:
            num_mods_active = self.encoder.num_modalities
            
        B_feat, L_total, D_feat = features.shape
        L_spatial = L_total // num_mods_active
        H = int(np.sqrt(L_spatial))
        
        if L_total > L_spatial:
            num_mods = L_total // L_spatial
            features = features.view(B_feat, num_mods, L_spatial, D_feat).mean(dim=1)
            
        features = features.transpose(1, 2).reshape(-1, D_feat, H, H)
        preds = self.decoder(features, target_size)
        return preds

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.forward(X_batch, y_batch.shape[-2:])
        loss = self.criterion(preds, y_batch)
        self.log('train/loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.forward(X_batch, y_batch.shape[-2:])
        val_loss = self.criterion(preds, y_batch)
        self.log('val/loss', val_loss, sync_dist=True)
        
        # Calculate metrics
        pred_labels = preds.argmax(dim=1)
        valid_mask = (y_batch != self.ignore_index)
        pred_labels = pred_labels[valid_mask]
        y_batch_valid = y_batch[valid_mask]
        
        intersections = torch.zeros(self.num_classes, device=self.device)
        unions = torch.zeros(self.num_classes, device=self.device)
        class_corrects = torch.zeros(self.num_classes, device=self.device)
        class_totals = torch.zeros(self.num_classes, device=self.device)
        
        for c in range(self.num_classes):
            pred_c = (pred_labels == c)
            target_c = (y_batch_valid == c)
            intersections[c] += (pred_c & target_c).sum()
            unions[c] += (pred_c | target_c).sum()
            class_corrects[c] += (pred_c & target_c).sum()
            class_totals[c] += target_c.sum()
            
        ious = intersections / unions.clamp(min=1)
        accs = class_corrects / class_totals.clamp(min=1)
        
        valid_classes = (class_totals > 0)
        mIoU = ious[valid_classes].mean().item() if valid_classes.any() else 0.0
        mAcc = accs[valid_classes].mean().item() if valid_classes.any() else 0.0
        global_acc = class_corrects.sum().item() / class_totals.sum().item() if class_totals.sum().item() > 0 else 0.0
        
        self.log('val/mIoU', mIoU, sync_dist=True)
        self.log('val/mAcc', mAcc, sync_dist=True)
        self.log('val/global_acc', global_acc, sync_dist=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
