"""BYOL-A audio encoder, reimplemented here so the repository has no dependency
on the reference implementation.

The architecture is the DCASE2020 Task-6 NTT audio embedding network used as the
BYOL-A encoder: three 3x3 conv blocks (64 channels, BN, ReLU, 2x2 max-pool) over
the log-mel spectrogram, then a two-layer MLP applied per time step. Module names
and `nn.Sequential` indices match the released checkpoints, so
`AudioNTT2020-BYOLA-64x96d512.pth` loads without renaming keys.

Reference:
    Niizumi et al., "BYOL for Audio: Exploring Pre-trained General-purpose Audio
    Representations", IEEE/ACM TASLP 31:137-151, 2023.
"""
import torch
import torch.nn as nn
import torchaudio


# Front-end settings the released weights were trained with. `unit_sec` is not
# used here: the convolutional trunk accepts any number of frames, and the caller
# decides the clip length.
BYOLA_MELSPEC = dict(sample_rate=16000, n_fft=1024, win_length=1024,
                     hop_length=160, n_mels=64, f_min=60, f_max=7800, power=2)


class LogMelSpectrogram(nn.Module):
    """Waveform to normalized log-mel spectrogram, as BYOL-A expects it.

    `stats` is the (mean, std) of the log-mel values over the *training* split of
    the dataset being encoded. BYOL-A is sensitive to this: the released weights
    were trained on normalized inputs, and the reference implementation requires
    the statistics to be measured per dataset rather than reused across datasets.
    """

    def __init__(self, stats=None, **melspec_kwargs):
        super().__init__()
        cfg = {**BYOLA_MELSPEC, **melspec_kwargs}
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**cfg)
        self.register_buffer("stats", torch.tensor([0.0, 1.0] if stats is None
                                                   else list(stats), dtype=torch.float))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, samples) -> (B, 1, n_mels, frames)."""
        lms = (self.to_melspec(wav) + torch.finfo(torch.float).eps).log()
        lms = (lms - self.stats[0]) / self.stats[1]
        return lms.unsqueeze(1) if lms.ndim == 3 else lms


class AudioNTT2020(nn.Module):
    """BYOL-A encoder. Returns either the per-frame sequence or the pooled vector.

    :param n_mels: mel bins of the input spectrogram; sets the flattened width
        fed to `fc`, which must match the checkpoint (64 mels -> 64ch * 8 = 512).
    :param d: embedding width, 512 / 1024 / 2048 depending on the checkpoint.
    """

    def __init__(self, n_mels: int = 64, d: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2 ** 3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x: torch.Tensor, pooled: bool = False) -> torch.Tensor:
        """(B, 1, n_mels, frames) -> (B, frames//8, d), or (B, d) if `pooled`.

        The pooled form is BYOL-A's own output: max over time plus mean over time.
        """
        x = self.features(x)               # (B, ch, mel, time)
        x = x.permute(0, 3, 2, 1)          # (B, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))       # (B, time, mel*ch)
        x = self.fc(x)                     # (B, time, d)
        if not pooled:
            return x
        return x.max(dim=1).values + x.mean(dim=1)

    @torch.no_grad()
    def load_byola_weights(self, weight_file: str, map_location="cpu"):
        """Load a released BYOL-A checkpoint and freeze the model."""
        state = torch.load(weight_file, map_location=map_location)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        # The released files carry bare `features.*` / `fc.*` keys; anything else
        # comes from a training wrapper and is stripped to the same layout.
        cleaned = {}
        for k, v in state.items():
            for prefix in ("features.", "fc."):
                if prefix in k:
                    cleaned[k[k.index(prefix):]] = v
                    break
        missing, unexpected = self.load_state_dict(cleaned, strict=True), None
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
