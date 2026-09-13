import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import Callback, Trainer, LightningModule, LightningDataModule
from typing import List, Optional
from tqdm import tqdm

class LinearSegmentationHead(nn.Module):
    """Dense linear probe: a 1x1 convolution on the frozen feature map, then bilinear
    upsampling of the *logits* so the loss is taken at full resolution and no label is
    resampled.

    A 1x1 convolution over the token grid is a linear layer applied per token, so only
    `D * num_classes + num_classes` parameters train. That is the protocol: the decoder
    this replaces carried ~2.8M, and measured whether a decoder could recover the labels
    rather than whether the representation made them linearly available.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x, target_size):
        x = self.proj(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class SegmentationProbingCallback(Callback):
    def __init__(self, downstream_data_modules: List[LightningDataModule],
                 names: Optional[List[str]] = None,
                 epochs: int = 10,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 num_classes: int = 2,
                 ignore_index: int = -1,
                 every_n_epochs: int = 5,
                 fuse_modalities: str = "concat",
                 **extraction_kwargs):
        self.downstream_data_modules = downstream_data_modules
        self.names = names
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        # With `return_tokens=True` there is one token grid per stream and no fused
        # vector per spatial position, so the probe has to combine them itself.
        # 'concat' leaves the weighting to the linear layer and contains 'mean' as a
        # special case; 'mean' would cancel complementary components between streams,
        # destroying exactly the unique and synergistic content being measured.
        if fuse_modalities not in ("mean", "concat"):
            raise ValueError(f"fuse_modalities must be 'mean' or 'concat', got {fuse_modalities!r}")
        self.fuse_modalities = fuse_modalities
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.every_n_epochs = every_n_epochs
        self.last_metrics = {}
        self.extraction_kwargs = extraction_kwargs
        if self.names is None:
            self.names = [d.__class__.__name__ for d in downstream_data_modules]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # We check modulo against current_epoch + 1, since epochs are 0-indexed
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.segmentation_probing(trainer, pl_module)
        else:
            for dataset_name in self.names:
                pl_module.log(f"Probe/{dataset_name}_GlobalAcc", self.last_metrics.get(f"Probe/{dataset_name}_GlobalAcc", 0.0), sync_dist=True)
                pl_module.log(f"Probe/{dataset_name}_mAcc", self.last_metrics.get(f"Probe/{dataset_name}_mAcc", 0.0), sync_dist=True)
                pl_module.log(f"Probe/{dataset_name}_mIoU", self.last_metrics.get(f"Probe/{dataset_name}_mIoU", 0.0), sync_dist=True)
            
    def _to_grid(self, features, L_spatial, H):
        """(B, L_total, D) tokens -> (B, C, H, H) feature map."""
        B, L, D = features.shape
        if L > L_spatial:
            n_mods = L // L_spatial
            features = features.view(B, n_mods, L_spatial, D)
            features = (features.mean(dim=1) if self.fuse_modalities == "mean"
                        else features.permute(0, 2, 1, 3).reshape(B, L_spatial, n_mods * D))
        return features.transpose(1, 2).reshape(B, -1, H, H)

    def segmentation_probing(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.global_rank == 0:
            device = pl_module.device
            
            # This probe does not go through `extract_features`, so the choice of
            # network is made here too. CoMM and WoMM expose no `probe_encoder`.
            encoder = getattr(pl_module, "probe_encoder", pl_module.encoder)

            for downstream_data_mod, dataset_name in zip(self.downstream_data_modules, self.names):
                train_loader = downstream_data_mod.train_dataloader()
                val_loader = downstream_data_mod.val_dataloader()
                pl_module.eval()
                with torch.no_grad():
                    sample_x, sample_y = next(iter(train_loader))
                    sample_x = [x.to(device) for x in sample_x]
                    sample_feat = encoder(sample_x, **self.extraction_kwargs, return_tokens=True)
                    if isinstance(sample_feat, list):
                        sample_feat = sample_feat[0]
                
                # Determine number of modalities being fused to correctly find the spatial L
                mask_mod = self.extraction_kwargs.get("mask_modalities", None)
                if mask_mod is not None:
                    num_mods_active = sum(mask_mod[0]) if isinstance(mask_mod[0], list) else sum(mask_mod)
                else:
                    num_mods_active = encoder.num_modalities
                
                # sample_feat is (B, L_total, D). We assume spatial is square per modality.
                B, L_total, D = sample_feat.shape
                L_spatial = L_total // num_mods_active
                H = int(round(np.sqrt(L_spatial)))
                if H * H != L_spatial:
                    raise ValueError(
                        f"{dataset_name}: {L_total} tokens over {num_mods_active} modalities "
                        f"gives {L_spatial} per modality, which is not a square grid. The "
                        f"probe reshapes tokens to (H, H) and cannot infer a non-square one.")
                in_channels = D
                
                probe_channels = (in_channels * num_mods_active
                                  if self.fuse_modalities == "concat" else in_channels)
                decoder = LinearSegmentationHead(probe_channels, self.num_classes).to(device)
                optimizer = torch.optim.Adam(decoder.parameters(), lr=self.lr,
                                             weight_decay=self.weight_decay)
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
                                features = encoder(X_batch, **self.extraction_kwargs, return_tokens=True)
                                if isinstance(features, list):
                                    features = features[0]
                        
                            # Handle concatenated modalities from MMFusion
                            features = self._to_grid(features, L_spatial, H)
                            
                            optimizer.zero_grad()
                            preds = decoder(features, y_batch.shape[-2:])
                            loss = criterion(preds, y_batch)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                
                # Validation Loop
                decoder.eval()
                intersections = torch.zeros(self.num_classes, device=device)
                unions = torch.zeros(self.num_classes, device=device)
                class_corrects = torch.zeros(self.num_classes, device=device)
                class_totals = torch.zeros(self.num_classes, device=device)
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = [x.to(device) for x in X_batch]
                        y_batch = y_batch.to(device)
                        
                        features = encoder(X_batch, **self.extraction_kwargs, return_tokens=True)
                        if isinstance(features, list):
                            features = features[0]
                            
                        # Handle concatenated modalities from MMFusion
                        features = self._to_grid(features, L_spatial, H)
                        
                        preds = decoder(features, y_batch.shape[-2:])
                        pred_labels = preds.argmax(dim=1)
                        
                        valid_mask = (y_batch != self.ignore_index)
                        pred_labels = pred_labels[valid_mask]
                        y_batch_valid = y_batch[valid_mask]
                        
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
                
                global_correct = class_corrects.sum().item()
                global_total = class_totals.sum().item()
                global_acc = global_correct / global_total if global_total > 0 else 0
                
                print(f"Segmentation Probe ({dataset_name}) - Global Acc: {global_acc:.4f}, mAcc: {mAcc:.4f}, mIoU: {mIoU:.4f}")
                pl_module.log(f"Probe/{dataset_name}_GlobalAcc", global_acc, sync_dist=True)
                pl_module.log(f"Probe/{dataset_name}_mAcc", mAcc, sync_dist=True)
                pl_module.log(f"Probe/{dataset_name}_mIoU", mIoU, sync_dist=True)
                self.last_metrics[f"Probe/{dataset_name}_GlobalAcc"] = global_acc
                self.last_metrics[f"Probe/{dataset_name}_mAcc"] = mAcc
                self.last_metrics[f"Probe/{dataset_name}_mIoU"] = mIoU
            
            pl_module.train()
