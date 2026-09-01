import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class ClassificationFineTuner(pl.LightningModule):
    def __init__(self, encoder, learning_rate: float = 1e-4, num_classes: int = 6, mask_modalities=None, extraction_kwargs=None):
        super().__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.mask_modalities = mask_modalities if mask_modalities is not None else [[True, True]]
        self.extraction_kwargs = extraction_kwargs if extraction_kwargs is not None else {}
        self.extraction_kwargs['mask_modalities'] = self.mask_modalities
        
        if hasattr(self.encoder, 'embed_dim'):
            self.in_channels = self.encoder.embed_dim
        else:
            self.in_channels = 512 # Fallback
            
        self.decoder = nn.Linear(self.in_channels, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
    def forward(self, x_batch):
        features = self.encoder(x_batch, **self.extraction_kwargs)
        if isinstance(features, list):
            features = features[0]
            
        preds = self.decoder(features)
        return preds

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.forward(X_batch)
        loss = self.criterion(preds, y_batch)
        acc = self.train_acc(preds, y_batch)
        self.log('train/loss', loss, sync_dist=True)
        self.log('train/acc', acc, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.forward(X_batch)
        val_loss = self.criterion(preds, y_batch)
        acc = self.val_acc(preds, y_batch)
        self.log('val/loss', val_loss, sync_dist=True)
        self.log('val/acc', acc, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.forward(X_batch)
        val_loss = self.criterion(preds, y_batch)
        acc = self.test_acc(preds, y_batch)
        self.log('test/loss', val_loss, sync_dist=True)
        self.log('test/acc', acc, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        milestone = int(self.trainer.max_epochs * 0.8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.1)
        return [optimizer], [scheduler]
