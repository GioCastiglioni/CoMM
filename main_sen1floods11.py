from omegaconf import DictConfig, open_dict
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from evaluation.segmentation_probe import SegmentationProbingCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_results_dir


@hydra.main(version_base=None, config_name="train_sen1floods11", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on Sen1Floods11 dataset.
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    data_module.setup(stage="fit")
    
    base_steps = len(data_module.train_dataloader())
    
    raw_devices = cfg.trainer.get("devices", 1)
    if isinstance(raw_devices, (list, tuple)):
        num_devices = len(raw_devices)
    elif isinstance(raw_devices, str):
        if raw_devices.lower() == "auto":
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        elif "," in raw_devices:
            num_devices = len([d for d in raw_devices.split(",") if d.strip()])
        else:
            num_devices = int(raw_devices)
    else:
        num_devices = int(raw_devices)
        
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    
    calculated_steps_per_epoch = max(1, base_steps // (num_devices * num_nodes))
    print(f"Auto-calculated steps_per_epoch: {calculated_steps_per_epoch} (Base: {base_steps}, Devices: {num_devices}, Nodes: {num_nodes})")

    with open_dict(cfg):
        if "loss_kwargs" in cfg.model.model:
            cfg.model.model.loss_kwargs.steps_per_epoch = calculated_steps_per_epoch
    kwargs = dict()

    if cfg.model.name== "CoMM" or cfg.model.name== "WoMM":
        kwargs["encoder"] = {
            "encoders": instantiate(cfg.model.encoders),
            "input_adapters": instantiate(cfg.model.adapters)}

    if cfg.model.name == "CLIP":
        encoders = instantiate(cfg.model.encoders)
        kwargs["visual"], kwargs["language"] = encoders[0], encoders[1]
        kwargs["image_projection"] = instantiate(cfg.model.clip_image_projection)
        kwargs["text_projection"] = instantiate(cfg.model.clip_text_projection)

    if cfg.model.name == "CrossSelf":
        encoders = instantiate(cfg.model.encoders)
        kwargs["enc1"] = encoders[0]
        kwargs["enc2"] = encoders[1]
        kwargs["head1"] = instantiate(cfg.model.visual_projection)
        kwargs["head2"] = instantiate(cfg.model.visual_projection)

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    downstream_names = ["s1_s2", "s1_only", "s2_only"]
    mask_modalities_list = [
        [[True, True]],   # S1 + S2
        [[True, False]],  # S1 only
        [[False, True]],  # S2 only
    ]
    
    downstream_data_modules = [instantiate(cfg.data.data_module, model="Sup")
                               for _ in downstream_names]
                               
    callbacks = [SegmentationProbingCallback([d_mod],
                                       names=[name],
                                       mask_modalities=mask,
                                       every_n_epochs=5)
                 for d_mod, name, mask in zip(downstream_data_modules, downstream_names, mask_modalities_list)]

    run_name = str(cfg.model.name) + \
        f"_{str(cfg.model.model.loss_kwargs.reconstruction)}" + \
        f"_{str(cfg.model.model.loss_kwargs.regularization)}" + \
        f"_{str(cfg.model.model.loss_kwargs.reg_weight)}" + \
        str("_sg" if getattr(cfg.model.model.loss_kwargs, "stop_grad", False) else "")

    results_dir = setup_results_dir(cfg, run_name)

    checkpoint_callbacks = {
        name: ModelCheckpoint(
            monitor=f"Probe/{name}_mIoU",
            mode="max",
            save_top_k=1,
            filename=f"best-checkpoint-{name}",
            dirpath=results_dir
        ) for name in downstream_names
    }
    callbacks.extend(list(checkpoint_callbacks.values()))



    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=results_dir,
        logger=[
            WandbLogger(project="Sen1Floods11",
                        name=run_name,
                        save_dir=results_dir)],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
        ckpt_path = "best"
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()

    print("Starting fine-tuning stage...")
    from pl_modules.segmentation_finetuner import SegmentationFineTuner
    
    class CustomFinetuningCallback(pl.Callback):
        def __init__(self, unfreeze_at_epoch=5, unfreeze_lr=3e-4):
            self.unfreeze_at_epoch = unfreeze_at_epoch
            self.unfreeze_lr = unfreeze_lr

        def on_fit_start(self, trainer, pl_module):
            for param in pl_module.encoder.parameters():
                param.requires_grad = False
                
        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == self.unfreeze_at_epoch:
                print(f"Unfreezing encoder at epoch {trainer.current_epoch}")
                for param in pl_module.encoder.parameters():
                    param.requires_grad = True
                for opt in trainer.optimizers:
                    for param_group in opt.param_groups:
                        param_group['lr'] = self.unfreeze_lr

    best_ckpt_paths = {name: cb.best_model_path if cfg.mode == "train" else ckpt_path for name, cb in checkpoint_callbacks.items()}
    
    for d_mod, name, mask in zip(downstream_data_modules, downstream_names, mask_modalities_list):
        best_ckpt_path = best_ckpt_paths[name]
        if not best_ckpt_path or not os.path.exists(best_ckpt_path):
            print(f"No valid checkpoint found for fine-tuning {name}!")
            continue
        print(f"Fine-tuning for {name}...")
        
        best_model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
        state_dict = torch.load(best_ckpt_path, map_location='cpu')["state_dict"]
        best_model.load_state_dict(state_dict)
        encoder = best_model.encoder
        
        unfreeze_lr = cfg.optim.lr
        finetuner = SegmentationFineTuner(
            encoder=encoder,
            learning_rate=1e-3, 
            num_classes=2,
            ignore_index=-1,
            mask_modalities=mask
        )
        
        ft_results_dir = os.path.join(results_dir, f"finetune_{name}")
        
        ft_checkpoint_callback = ModelCheckpoint(
            monitor="val/mIoU",
            mode="max",
            save_top_k=1,
            filename=f"best-finetuned-{name}",
            dirpath=ft_results_dir
        )
        
        ft_callbacks = [CustomFinetuningCallback(unfreeze_at_epoch=5, unfreeze_lr=unfreeze_lr), ft_checkpoint_callback]
        
        ft_trainer = instantiate(
            cfg.trainer,
            default_root_dir=ft_results_dir,
            max_epochs=55,
            logger=[WandbLogger(project="Sen1Floods11_Finetune", name=f"Finetune_{name}_{run_name}", save_dir=ft_results_dir)],
            callbacks=ft_callbacks
        )
        
        ft_trainer.fit(finetuner, datamodule=d_mod)
        ft_trainer.test(finetuner, datamodule=d_mod, ckpt_path="best")
        wandb.finish()


if __name__ == '__main__':
    main()