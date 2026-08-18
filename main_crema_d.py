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
                               
    callbacks = [LinearProbingCallback([d_mod],
                                       names=[name],
                                       val_loaders=False,
                                       mask_modalities=mask,
                                       always_prefix=True)
                 for d_mod, name, mask in zip(downstream_data_modules, downstream_names, mask_modalities_list)]

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=[
            WandbLogger(project="CREMA-D",
                        name=str(cfg.model.name)+ \
                            f"_{str(cfg.model.model.loss_kwargs.reconstruction)}"+ \
                                f"_{str(cfg.model.model.loss_kwargs.regularization)}"+ \
                                    f"_{str(cfg.model.model.loss_kwargs.reg_weight)}"+ \
                                        str("_sg" if getattr(cfg.model.model.loss_kwargs, "stop_grad", False) else ""))],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
        ckpt_path = "best"
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "crema_d")

    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        if getattr(cfg, "ckpt_path", None) is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == '__main__':
    main()
