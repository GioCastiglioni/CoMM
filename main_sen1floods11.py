from omegaconf import DictConfig
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
from utils import setup_results_dir, build_run_identity, WANDB_PROJECT


@hydra.main(version_base=None, config_name="train_sen1floods11", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on Sen1Floods11 dataset.
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    data_module.setup(stage="fit")
    
    # steps_per_epoch is injected by BaseModel.on_train_epoch_start from
    # trainer.num_training_batches, so no per-dataset computation is needed here.
    kwargs = dict()

    if cfg.model.name in ("CoMM", "WoMM", "MMSD"):
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
                                       every_n_epochs=1)
                 for d_mod, name, mask in zip(downstream_data_modules, downstream_names, mask_modalities_list)]

    identity = build_run_identity(cfg, dataset="sen1floods11", stage="pretrain")
    run_name = identity.name

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
            WandbLogger(project=WANDB_PROJECT,
                        name=run_name,
                        save_dir=results_dir,
                        **identity.wandb_kwargs())],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
        ckpt_path = "best"
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == '__main__':
    main()