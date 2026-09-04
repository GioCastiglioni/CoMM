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
from dataset.trifeatures import BimodalTrifeatures
from evaluation.linear_probe import LinearProbingCallback
from utils import setup_results_dir, build_run_identity


@hydra.main(version_base=None, config_name="train_trifeatures", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on synthetic toy data (bimodal trifeatures) with
    controllable attributes (shape, color, texture).

    Models currently implemented are:
        - CoMM [ours!]
        - WoMM [ours!]
        - CLIP
        - CrossSelf
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
    biased = bool(cfg.data.data_module.get("biased", True))
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    # Linear probing on each task from BimodalTrifeatures. Always on unbiased pairs:
    # under `biased=True` every pair is correlated, which makes the synergy label
    # constant. `task="all"` lets one feature extraction serve the four tasks.
    downstream_names = list(BimodalTrifeatures.TASKS)
    downstream_data_module = instantiate(cfg.data.data_module, model="Sup",
                                         biased=False, task="all")

    # Each task is also probed from one modality at a time: a unique attribute must
    # be readable from its own modality only, and synergy from neither alone.
    probe_masks = {"both": [True, True], "mod1": [True, False], "mod2": [False, True]}
    # There is no fine-tuning stage here: these accuracies are the result.
    probe_every_n_epochs = 1

    callbacks = [LinearProbingCallback([downstream_data_module],
                                       names=[f"{t}_{m}" for t in downstream_names],
                                       val_loaders=False,
                                       mask_modalities=[mask],
                                       split_label_columns=True,
                                       fastsearch=True,
                                       every_n_epochs=probe_every_n_epochs)
                 for m, mask in probe_masks.items()]

    identity = build_run_identity(cfg, stage="pretrain",
                                  extra={"biased": biased},
                                  group_suffix="biased" if biased else "unbiased")
    run_name = identity.name
    results_dir = setup_results_dir(cfg, run_name)

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=results_dir,
        logger=[
            WandbLogger(project="Trifeatures",
                        name=run_name,
                        save_dir=results_dir,
                        **identity.wandb_kwargs())],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
        # Test the final weights: nothing is selected on the eval split.
        ckpt_path = None
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == '__main__':
    main()
