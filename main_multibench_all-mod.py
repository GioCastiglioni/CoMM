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
from main_multibench import modality_masks


@hydra.main(version_base=None, config_name="train_multibench_all-mod", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on MultiBench datasets, all modalities.

    Models currently implemented are:
        - CoMM [ours!]
        - WoMM [ours!]
        - CMC
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    # create model + save hyper-parameters
    dataset = cfg.data.data_module.dataset # Which MultiBench dataset to load
    kwargs = dict()
    if cfg.model.name in ("CoMM", "WoMM", "MMSD"):
        encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"]) # adapters also specific
        kwargs["encoder"] = {
            "encoders": encoders,
            "input_adapters": adapters}
    elif cfg.model.name == "CMC":
        encoders = instantiate(cfg[dataset]["encoders"])
        heads = instantiate(cfg[dataset]["cmc_heads"])
        kwargs["encoders"] = encoders
        kwargs["heads"] = heads

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    # Data loading code
    modalities = list(cfg[dataset]["modalities"])
    data_module = instantiate(cfg.data.data_module,
                              model=cfg.model.name,
                              modalities=modalities,
                              task=cfg[dataset]["task"],
                              **cfg[dataset]["kwargs"])

    downstream_data_module = instantiate(cfg.data.data_module,
                                         model="Sup",
                                         modalities=modalities,
                                         task=cfg[dataset]["task"])

    # One probe per mask: the joint representation plus each modality alone.
    # `always_prefix` is required even with a single name, otherwise every
    # callback logs the bare key `acc1` and overwrites the others.
    callbacks = [instantiate(cfg.linear_probing,
                             downstream_data_modules=[downstream_data_module],
                             names=[f"{dataset}_{m}"],
                             mask_modalities=[mask],
                             always_prefix=True,
                             _convert_="all")
                 for m, mask in modality_masks(len(modalities)).items()]

    # Resuming: `ckpt_path` continues training from a finished run and `wandb_id`
    # keeps the curves in that run instead of opening a second one.
    resume_ckpt = getattr(cfg, "ckpt_path", None) if cfg.mode == "train" else None
    wandb_id = getattr(cfg, "wandb_id", None)
    wandb_resume = {"id": wandb_id, "resume": "allow"} if wandb_id else {}

    identity = build_run_identity(cfg, dataset=dataset, stage="pretrain",
                                  extra={"n_modalities": len(modalities)},
                                  group_suffix="allmod")
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
