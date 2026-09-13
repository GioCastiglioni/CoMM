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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from utils import setup_results_dir, build_run_identity, WANDB_PROJECT


# Image + text, so the joint mask keeps the name `both` as in the other bimodal
# datasets. Probing one modality at a time is what makes the PID reading falsifiable.
PROBE_MASKS = {"both": [True, True], "mod1": [True, False], "mod2": [False, True]}


@hydra.main(version_base=None, config_name="train_hateful_memes", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on Hateful Memes dataset.

    Models currently implemented are:
        - CoMM [ours!]
        - WoMM [ours!]
        - CLIP
        - SLIP
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    # create model + save hyper-parameters
    dataset = "hateful_memes"
    model_kwargs = dict()
    if cfg.model.name in ("CoMM", "WoMM", "MMSD"):  # encoders + adapters for MMFusion
        encoders = instantiate(cfg[dataset]["encoders"])  # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"])  # adapters also specific
        model_kwargs = dict(encoder=dict(encoders=encoders, input_adapters=adapters))

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **model_kwargs)
    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    downstream_data_module = instantiate(cfg.data.data_module, model="Sup")

    # One probe per mask. `always_prefix` is required even with a single name:
    # without it each callback would log the bare key `roc_auc` and overwrite
    # the others, which is also why the checkpoint monitors the prefixed key.
    callbacks = [instantiate(cfg.linear_probing,
                             downstream_data_modules=[downstream_data_module],
                             names=[f"{dataset}_{m}"],
                             mask_modalities=[mask],
                             always_prefix=True,
                             _convert_="all")
                 for m, mask in PROBE_MASKS.items()]
    # `save_last` gives the resume path a deterministic target; the monitored copy
    # keeps CoMM's original protocol (best epoch on the eval split) available.
    callbacks.append(ModelCheckpoint(monitor="roc_auc_{}_both".format(dataset),
                                     mode="max", save_top_k=1, save_last=True))

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
