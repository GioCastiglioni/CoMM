from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from utils import setup_results_dir, build_run_identity, WANDB_PROJECT


# Sentinel-1 + Sentinel-2, so the joint mask keeps the name `both` as in the other
# bimodal datasets. Probing one modality at a time is what makes the PID reading
# falsifiable, and here it also separates what SAR carries from what optical does.
PROBE_MASKS = {"both": [True, True], "mod1": [True, False], "mod2": [False, True]}


@hydra.main(version_base=None, config_name="train_so2sat", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on GeoBench So2Sat.

    Local Climate Zone classification over 17 balanced classes from Sentinel-1 SAR
    (VV, VH) and Sentinel-2 optical (10 bands). Arrays are produced once by
    analysis/convert_so2sat.py; both encoders are trained from raw input.

    Models currently implemented are:
        - CoMM [ours!]
        - WoMM [ours!]
        - MMSD [ours!]
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    # create model + save hyper-parameters
    dataset = "so2sat"
    kwargs = dict()
    if cfg.model.name in ("CoMM", "WoMM", "MMSD"):
        kwargs["encoder"] = {
            "encoders": instantiate(cfg.model.encoders),
            "input_adapters": instantiate(cfg.model.adapters)}

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    downstream_data_module = instantiate(cfg.data.data_module, model="Sup")

    # One probe per mask. `always_prefix` is required even with a single name:
    # without it every callback would log the bare key `acc1` and overwrite the
    # others.
    callbacks = [instantiate(cfg.linear_probing,
                             downstream_data_modules=[downstream_data_module],
                             names=[f"{dataset}_{m}"],
                             mask_modalities=[mask],
                             always_prefix=True,
                             _convert_="all")
                 for m, mask in PROBE_MASKS.items()]

    # Resuming: `ckpt_path` continues training from a finished run and `wandb_id`
    # keeps the curves in that run instead of opening a second one.
    resume_ckpt = getattr(cfg, "ckpt_path", None) if cfg.mode == "train" else None
    wandb_id = getattr(cfg, "wandb_id", None)
    wandb_resume = {"id": wandb_id, "resume": "allow"} if wandb_id else {}

    identity = build_run_identity(cfg, dataset=dataset, stage="pretrain")
    run_name = identity.name
    results_dir = setup_results_dir(cfg, run_name)

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=results_dir,
        logger=[
            WandbLogger(project=WANDB_PROJECT,
                        name=run_name,
                        save_dir=results_dir,
                        **wandb_resume,
                        **identity.wandb_kwargs())],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
        # Test the final weights: nothing is selected on the eval split.
        ckpt_path = None
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == '__main__':
    main()
