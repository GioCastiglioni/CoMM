"""CREMA-D as pre-extracted frozen features (see extract_crema_d_features.py).

Shaped like the MultiBench affect datasets so it uses the same encoders: each
modality is a `(T, p)` sequence of feature vectors that a small Transformer turns
into tokens for the fusion module.

    video  (16, 1024)  LeVJEPA patch tokens, mean-pooled over space per frame
    audio  (25, 512)   BYOL-A per-frame embeddings
    label  6 emotion classes, actor-disjoint train/test split

Arrays are opened with `mmap_mode="r"`, so a worker reads only the rows it is
asked for and several concurrent runs share one page cache. Stored as float16 and
cast on read: half the disk and bandwidth, and these are frozen features whose
useful precision is far below fp32.
"""
import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

EMOTIONS = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
MODALITIES = ("video", "audio")


class CREMADFeatures(Dataset):
    """One split of the extracted features.

    :param root: directory written by extract_crema_d_features.py
    :param split: "train" or "test"
    :param modalities: subset and order of modalities to return
    :param pooled: read the pooled vectors instead of the sequences, returned as
        1-length sequences so the fusion module sees the same rank either way
    :param augmentations: per-modality augmentation, "noise", "drop+noise" or None,
        matching the time-series augmentation used for the MultiBench datasets
    """

    def __init__(self, root: str, split: str = "train",
                 modalities: Tuple[str, ...] = MODALITIES,
                 pooled: bool = False,
                 augmentations: Optional[Union[str, List[Optional[str]]]] = None,
                 noise_std: float = 0.1, drop_max: float = 0.8):
        super().__init__()
        self.root, self.split = root, split
        self.modalities = tuple(modalities)
        self.pooled = pooled
        self.noise_std, self.drop_max = noise_std, drop_max

        unknown = set(self.modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {unknown}; available: {MODALITIES}")

        meta_path = os.path.join(root, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"{meta_path} not found. Run extract_crema_d_features.py first "
                "(GPU job: run_extract_crema.sh).")
        self.meta = json.load(open(meta_path))

        suffix = "_pooled" if pooled else ""
        self.data = {m: np.load(os.path.join(root, f"{split}_{m}{suffix}.npy"),
                                mmap_mode="r") for m in self.modalities}
        self.labels = np.load(os.path.join(root, f"{split}_labels.npy"))
        ids_path = os.path.join(root, f"{split}_ids.json")
        self.ids = json.load(open(ids_path)) if os.path.isfile(ids_path) else None

        n = len(self.labels)
        for m, arr in self.data.items():
            if len(arr) != n:
                raise ValueError(f"{m} has {len(arr)} rows but labels have {n}")

        if augmentations is None or isinstance(augmentations, str):
            self.augmentations = [augmentations] * len(self.modalities)
        else:
            self.augmentations = list(augmentations)
            if len(self.augmentations) != len(self.modalities):
                raise ValueError("one augmentation per modality is required")

    def __len__(self):
        return len(self.labels)

    @property
    def feature_dims(self):
        """Input width per modality, for configuring the encoders."""
        return {m: int(arr.shape[-1]) for m, arr in self.data.items()}

    def _augment(self, x: torch.Tensor, kind: Optional[str]) -> torch.Tensor:
        if not kind:
            return x
        if "drop" in kind:
            # Zero a random contiguous share of the sequence, up to drop_max.
            T = x.shape[0]
            n_drop = int(torch.randint(0, int(T * self.drop_max) + 1, (1,)).item())
            if n_drop:
                start = int(torch.randint(0, T - n_drop + 1, (1,)).item())
                x = x.clone()
                x[start:start + n_drop] = 0
        if "noise" in kind:
            x = x + torch.randn_like(x) * self.noise_std
        return x

    def __getitem__(self, i):
        out = []
        for m, aug in zip(self.modalities, self.augmentations):
            # np.asarray materialises just this row out of the memory map.
            x = torch.from_numpy(np.asarray(self.data[m][i], dtype=np.float32))
            if self.pooled:
                x = x.unsqueeze(0)  # (p,) -> (1, p): a 1-length sequence
            out.append(self._augment(x, aug))
        return out, torch.tensor(int(self.labels[i]), dtype=torch.long)


class CREMADFeaturesSSL(CREMADFeatures):
    """Two augmented views per item, for the contrastive objectives.

    Returns `[[view1_mod1, view1_mod2], [view2_mod1, view2_mod2]], label`, the
    shape WoMM's `forward` expects from the other multimodal datasets.
    """

    def __getitem__(self, i):
        views = []
        for _ in range(2):
            v = []
            for m, aug in zip(self.modalities, self.augmentations):
                x = torch.from_numpy(np.asarray(self.data[m][i], dtype=np.float32))
                if self.pooled:
                    x = x.unsqueeze(0)
                v.append(self._augment(x, aug))
            views.append(v)
        return views, torch.tensor(int(self.labels[i]), dtype=torch.long)


class CREMADFeaturesDataModule(LightningDataModule):
    """CREMA-D features for pre-training (`model` in {CoMM, WoMM}) or probing.

    The test split doubles as the validation split: it is actor-disjoint from
    train, nothing is selected on it during pre-training, and the linear probe is
    the reported measurement.
    """

    def __init__(self, model: str,
                 root: str = "/home/gcastiglioni/workspace/datasets/CREMA-D-features",
                 modalities: Tuple[str, ...] = MODALITIES,
                 pooled: bool = False,
                 augmentations: Optional[Union[str, List[Optional[str]]]] = "drop+noise",
                 batch_size: int = 64,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.model = model
        self.root = root
        self.modalities = tuple(modalities)
        self.pooled = pooled
        # Augmentations only make sense for the self-supervised views; the probe
        # must see the features as they are.
        self.augmentations = augmentations if model in ("CoMM", "WoMM") else None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _build(self, split):
        cls = CREMADFeaturesSSL if self.model in ("CoMM", "WoMM") else CREMADFeatures
        return cls(self.root, split=split, modalities=self.modalities,
                   pooled=self.pooled, augmentations=self.augmentations)

    def setup(self, stage=None):
        self.train_dataset = self._build("train")
        self.val_dataset = self._build("test")
        self.test_dataset = self._build("test")

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=shuffle)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
