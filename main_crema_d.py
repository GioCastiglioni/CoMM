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
from evaluation.linear_probe import LinearProbingCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_results_dir, build_run_identity, WANDB_PROJECT
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.overrides")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", message=".*meshgrid.*")


@hydra.main(version_base=None, config_name="train_crema_d", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on CREMA-D dataset.
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    # create model + save hyper-parameters
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

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    # Linear probing on each task: Video-Only, Audio-Only, Video+Audio
    downstream_names = ["video_only", "audio_only", "video_audio"]
    mask_modalities_list = [
        [[True, False]],  # video only
        [[False, True]],  # audio only
        [[True, True]]    # video + audio
    ]
    
    downstream_data_modules = [instantiate(cfg.data.data_module, model="Sup")
                               for _ in downstream_names]
                               
    # The probe only logs acc1_* on the epochs it runs, so the checkpoints that
    # monitor those keys must share its cadence: both gate on (current_epoch + 1) % n.
    probe_every_n_epochs = 1

    callbacks = [LinearProbingCallback([d_mod],
                                       names=[name],
                                       val_loaders=False,
                                       mask_modalities=mask,
                                       always_prefix=True,
                                       every_n_epochs=probe_every_n_epochs)
                 for d_mod, name, mask in zip(downstream_data_modules, downstream_names, mask_modalities_list)]

    identity = build_run_identity(cfg, dataset="crema_d", stage="pretrain")
    run_name = identity.name

    results_dir = setup_results_dir(cfg, run_name)

    checkpoint_callbacks = {
        name: ModelCheckpoint(
            monitor=f"acc1_{name}",
            mode="max",
            save_top_k=1,
            every_n_epochs=probe_every_n_epochs,
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
