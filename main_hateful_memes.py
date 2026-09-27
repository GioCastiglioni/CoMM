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
    # No monitored checkpoint: the reported number is the probe of the final weights,
    # so picking a best epoch on the eval split is never used, and with the probe
    # thinned (`every_n_epochs` > 1) a monitored key is missing on most epochs, which
    # Lightning raises on. The default callback keeps one checkpoint, the last epoch,
    # which is also what RESUME reads. The frozen encoders make each one ~1 GB.

    # Resuming: `ckpt_path` continues training from a finished run and `wandb_id`
    # keeps the curves in that run instead of opening a second one.
    resume_ckpt = getattr(cfg, "ckpt_path", None) if cfg.mode == "train" else None
    wandb_id = getattr(cfg, "wandb_id", None)
    wandb_resume = {"id": wandb_id, "resume": "allow"} if wandb_id else {}

    # `group_suffix` marks an ablation of the pipeline itself, so its runs do not
    # land in the same cell as the reference ones when the results are harvested.
    identity = build_run_identity(cfg, dataset=dataset, stage="pretrain",
                                  group_suffix=getattr(cfg, "group_suffix", None))
    print(f"[main] {dataset}: embed_dim={cfg.embed_dim} lr={cfg.optim.lr} "
          f"wd={cfg.optim.weight_decay} epochs={cfg.trainer.max_epochs}")
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
