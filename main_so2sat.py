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
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import setup_resume, build_run_identity, WANDB_PROJECT


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

    identity = build_run_identity(cfg, dataset=dataset, stage="pretrain")
    run_name = identity.name

    # Resuming. The cluster requeues preempted jobs without warning, so this has to
    # work unattended: the run's directory is keyed on its identity rather than the
    # clock, `last.ckpt` is written every epoch, and a requeue picks it up here. An
    # explicit `ckpt_path` still wins, for resuming a run by hand.
    results_dir, ckpt_dir, found_ckpt, wandb_id = setup_resume(cfg, run_name)
    resume_ckpt = (getattr(cfg, "ckpt_path", None) or found_ckpt) \
        if cfg.mode == "train" else None
    wandb_resume = {"id": wandb_id, "resume": "allow"} if wandb_id else {}
    if resume_ckpt:
        print(f"[main] resuming from {resume_ckpt}")

    # `save_last` is the whole point: one rolling checkpoint per run, which bounds a
    # preemption's cost to the epoch in flight. `save_top_k=0` keeps no others --
    # nothing is selected on a metric here, the reported number comes from the final
    # weights, so a "best" checkpoint would only cost disk.
    callbacks = callbacks + [ModelCheckpoint(dirpath=ckpt_dir, save_last=True,
                                             save_top_k=0, every_n_epochs=1)]

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
