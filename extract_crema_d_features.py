"""Extract frozen CREMA-D features once, so training reads arrays instead of video.

Video goes through LeVJEPA (a ViT trained with LeJEPA on video) and audio through
BYOL-A. Both are frozen, so this runs once and every later experiment reads the
result. That removes video decoding from the training dataloader -- which is the
actual bottleneck on this dataset -- and shrinks 7.1 GB of .flv to a few hundred
MB of arrays.

It also makes CREMA-D comparable to the MultiBench cells: those encode
pre-extracted per-frame features with a small Transformer, whereas the current
CREMA-D pipeline trains AlexNets end to end, so any objective compared across the
two confounds the loss with encoder capacity.

Layout written to --out:
    meta.json                    config, shapes, dtypes, norm stats, label map
    <split>_video.npy            (N, 16, 1024)  float16   temporal token sequence
    <split>_audio.npy            (N, 25, 512)   float16   per-frame sequence
    <split>_video_pooled.npy     (N, 1024)      float16   LeVJEPA pooler_output
    <split>_audio_pooled.npy     (N, 512)       float16   BYOL-A max+mean over time
    <split>_labels.npy           (N,)           int64
    <split>_ids.json             list of clip basenames, aligned with the rows

Sequences keep the temporal axis because that is what the fusion module consumes
as tokens; the pooled vectors are each model's own canonical output, kept because
they cost almost nothing and save a re-extraction if a 1-length sequence is wanted.

GPU job -- submit with run_extract_crema.sh, do not run it on the login node.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import av
import torchaudio
from torch.utils.data import Dataset, DataLoader

from models.byol_a import AudioNTT2020, LogMelSpectrogram

EMOTIONS = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
# Actor-disjoint split, identical to dataset/crema_d.py so results stay comparable.
TRAIN_ACTORS = {str(i) for i in range(1001, 1071)}
TEST_ACTORS = {str(i) for i in range(1071, 1092)}
# CREMA-D clips run 1.2-5 s; 2 s at 16 kHz is the length the existing pipeline uses.
AUDIO_SAMPLES = 32000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def enumerate_samples(root, split):
    """List (basename, video_path, audio_path, label) for one split, sorted."""
    video_dir, audio_dir = os.path.join(root, "VideoFlash"), os.path.join(root, "AudioWAV")
    actors = TRAIN_ACTORS if split == "train" else TEST_ACTORS
    out = []
    for vf in sorted(os.listdir(video_dir)):
        if not vf.endswith(".flv"):
            continue
        base = vf[:-4]
        parts = base.split("_")
        if parts[0] not in actors or parts[2] not in EMOTIONS:
            continue
        audio_path = os.path.join(audio_dir, base + ".wav")
        if not os.path.isfile(audio_path):
            continue
        out.append((base, os.path.join(video_dir, vf), audio_path, EMOTIONS[parts[2]]))
    return out


def load_audio(path):
    """Mono 16 kHz waveform, centre-cropped or right-padded to a fixed length."""
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
    if wav.shape[0] > AUDIO_SAMPLES:
        off = (wav.shape[0] - AUDIO_SAMPLES) // 2
        wav = wav[off:off + AUDIO_SAMPLES]
    elif wav.shape[0] < AUDIO_SAMPLES:
        wav = F.pad(wav, (0, AUDIO_SAMPLES - wav.shape[0]))
    return wav


def decode_frames(path):
    """All frames of a clip as a uint8 tensor (T, C, H, W), decoded with PyAV.

    PyAV is used directly rather than `torchvision.io.read_video`: CREMA-D ships
    VP6-in-FLV whose container carries no duration or frame count, and
    torchvision's pyav backend returns zero frames for it. PyAV decodes it fine,
    and this also avoids depending on torchvision being built with its optional
    `video_reader` backend.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise RuntimeError("no frames decoded")
    arr = np.stack(frames)                       # (T, H, W, C)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)   # (T, C, H, W)


def load_video(path, num_frames=16, size=224):
    """(3, num_frames, size, size), ImageNet-normalised, frames evenly spaced."""
    frames = decode_frames(path)
    idx = torch.linspace(0, frames.shape[0] - 1, num_frames).long()
    clip = frames[idx].float().div_(255.0)
    # Resize the shorter side then centre-crop, so the aspect ratio survives.
    h, w = clip.shape[-2:]
    scale = size / min(h, w)
    clip = F.interpolate(clip, size=(max(size, int(round(h * scale))),
                                     max(size, int(round(w * scale)))),
                         mode="bilinear", align_corners=False, antialias=True)
    h, w = clip.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    clip = clip[:, :, top:top + size, left:left + size]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    return clip.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)


class ClipDataset(Dataset):
    """Decodes and preprocesses one clip per item, in dataloader workers.

    Decoding is the expensive part, so it is parallelised here while the two
    frozen encoders stay on the GPU in the main process. A clip that fails to
    decode returns ok=False and is dropped later instead of aborting the run.
    """

    def __init__(self, samples, num_frames=16, size=224):
        self.samples = samples
        self.num_frames, self.size = num_frames, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        base, vpath, apath, label = self.samples[i]
        try:
            video = load_video(vpath, self.num_frames, self.size)
            wav = load_audio(apath)
            return i, True, video, wav, label
        except Exception as e:  # keep the index so the caller can report it
            print(f"[warn] {base}: {type(e).__name__}: {e}", flush=True)
            return (i, False,
                    torch.zeros(3, self.num_frames, self.size, self.size),
                    torch.zeros(AUDIO_SAMPLES), label)


def collate(batch):
    idx, ok, video, wav, label = zip(*batch)
    return (torch.tensor(idx), torch.tensor(ok), torch.stack(video),
            torch.stack(wav), torch.tensor(label))


def compute_norm_stats(samples, max_clips, workers):
    """Mean/std of BYOL-A's log-mel over the training split.

    BYOL-A's released weights expect normalised input and the statistics are
    dataset-specific, so they are measured here rather than reused. Train split
    only: using the test clips would leak.
    """
    front = LogMelSpectrogram(stats=None)  # identity normalisation for measuring
    n = min(max_clips, len(samples))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(samples), size=n, replace=False).tolist())
    ds = ClipDataset([samples[i] for i in pick])
    # Audio only: skip video decoding entirely for this pass.
    total, total_sq, count = 0.0, 0.0, 0
    loader = DataLoader(_AudioOnly(ds), batch_size=64, num_workers=workers)
    for wav in loader:
        lms = front(wav)
        total += lms.sum().item()
        total_sq += (lms ** 2).sum().item()
        count += lms.numel()
    mean = total / count
    std = (total_sq / count - mean ** 2) ** 0.5
    print(f"[stats] log-mel mean={mean:.7f} std={std:.7f} over {n} clips", flush=True)
    return [mean, std]


class _AudioOnly(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        _, _, apath, _ = self.base.samples[i]
        return load_audio(apath)


def build_video_reshaper(model_cfg, n_tokens, num_frames):
    """Work out how LeVJEPA's patch tokens map onto (time, height, width).

    Derived from the config and checked against the real token count, so a model
    with different patch or tubelet settings fails loudly here instead of being
    silently pooled along the wrong axis.
    """
    patch = getattr(model_cfg, "patch_size", 16)
    tubelet = getattr(model_cfg, "tubelet_size", 1)
    img = getattr(model_cfg, "img_size", 224)
    grid_t, grid_hw = num_frames // tubelet, img // patch
    expected = grid_t * grid_hw * grid_hw
    if n_tokens != expected + 1:
        raise RuntimeError(
            f"expected {expected}+1 tokens for grid {grid_t}x{grid_hw}x{grid_hw}, "
            f"model returned {n_tokens}. Check patch_size/tubelet_size/num_frames.")
    print(f"[video] token grid {grid_t}x{grid_hw}x{grid_hw} (+1 CLS), "
          f"pooling over space -> {grid_t} tokens", flush=True)

    def reshape(hidden):  # (B, 1+T*H*W, D) -> (B, T, D)
        patches = hidden[:, 1:, :]
        b, _, d = patches.shape
        return patches.reshape(b, grid_t, grid_hw * grid_hw, d).mean(dim=2)

    return reshape, grid_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/gcastiglioni/workspace/datasets/crema-d-mirror")
    ap.add_argument("--out", default="/home/gcastiglioni/workspace/datasets/CREMA-D-features")
    ap.add_argument("--byola-weights",
                    default="/home/gcastiglioni/workspace/datasets/AudioNTT2020-BYOLA-64x96d512.pth")
    ap.add_argument("--byola-dim", type=int, default=512)
    ap.add_argument("--levjepa", default="galilai-group/LeVJEPA-VideoMix-Large")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats-clips", type=int, default=2000)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--min-success", type=float, default=0.95,
                    help="abort a split if fewer than this fraction of clips decode")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    splits = {s: enumerate_samples(args.root, s) for s in args.splits}
    for s, items in splits.items():
        print(f"[setup] {s}: {len(items)} clips", flush=True)
    if not any(splits.values()):
        raise SystemExit("no clips found -- check --root")

    # Normalisation statistics come from the training split only.
    stats_path = os.path.join(args.out, "byola_norm_stats.json")
    if os.path.isfile(stats_path):
        stats = json.load(open(stats_path))
        print(f"[stats] reusing {stats}", flush=True)
    else:
        base = splits.get("train") or next(iter(splits.values()))
        stats = compute_norm_stats(base, args.stats_clips, args.workers)
        json.dump(stats, open(stats_path, "w"))

    audio_front = LogMelSpectrogram(stats=stats).to(device).eval()
    audio_model = AudioNTT2020(n_mels=64, d=args.byola_dim)
    audio_model.load_byola_weights(args.byola_weights, map_location="cpu")
    audio_model = audio_model.to(device).eval()

    from transformers import AutoModel
    video_model = AutoModel.from_pretrained(args.levjepa, trust_remote_code=True)
    video_model = video_model.to(device).eval()
    for p in video_model.parameters():
        p.requires_grad = False
    reshape_video, grid_t = None, None

    meta = dict(root=args.root, num_frames=args.num_frames, size=args.size,
                audio_samples=AUDIO_SAMPLES, byola_weights=args.byola_weights,
                byola_dim=args.byola_dim, levjepa=args.levjepa,
                byola_norm_stats=stats, labels=EMOTIONS, dtype="float16",
                modalities=["video", "audio"], splits={})

    for split, samples in splits.items():
        if not samples:
            continue
        n = len(samples)
        loader = DataLoader(ClipDataset(samples, args.num_frames, args.size),
                            batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, collate_fn=collate, pin_memory=True)
        v_seq = a_seq = v_pool = a_pool = None
        labels = np.zeros(n, dtype=np.int64)
        valid = np.zeros(n, dtype=bool)

        for bi, (idx, ok, video, wav, label) in enumerate(loader):
            video, wav = video.to(device, non_blocking=True), wav.to(device, non_blocking=True)
            with torch.no_grad():
                out = video_model(pixel_values=video)
                hidden = out["last_hidden_state"] if isinstance(out, dict) else out.last_hidden_state
                pooled_v = out["pooler_output"] if isinstance(out, dict) else out.pooler_output
                if reshape_video is None:
                    reshape_video, grid_t = build_video_reshaper(
                        video_model.config, hidden.shape[1], args.num_frames)
                vs = reshape_video(hidden)
                lms = audio_front(wav)
                as_ = audio_model(lms)
                ap_ = audio_model(lms, pooled=True)

            if v_seq is None:  # allocate once the real widths are known
                v_seq = np.zeros((n, vs.shape[1], vs.shape[2]), dtype=np.float16)
                a_seq = np.zeros((n, as_.shape[1], as_.shape[2]), dtype=np.float16)
                v_pool = np.zeros((n, pooled_v.shape[-1]), dtype=np.float16)
                a_pool = np.zeros((n, ap_.shape[-1]), dtype=np.float16)
                print(f"[{split}] video {v_seq.shape[1:]} audio {a_seq.shape[1:]}", flush=True)

            rows = idx.numpy()
            v_seq[rows] = vs.float().cpu().numpy().astype(np.float16)
            a_seq[rows] = as_.float().cpu().numpy().astype(np.float16)
            v_pool[rows] = pooled_v.float().cpu().numpy().astype(np.float16)
            a_pool[rows] = ap_.float().cpu().numpy().astype(np.float16)
            labels[rows] = label.numpy()
            valid[rows] = ok.numpy()

            if bi % 20 == 0:
                done = min((bi + 1) * args.batch_size, n)
                print(f"[{split}] {done}/{n}", flush=True)

        keep = np.flatnonzero(valid)
        if len(keep) < n:
            print(f"[{split}] dropping {n - len(keep)} clips that failed to decode", flush=True)
        # A systematic decode failure -- a codec the reader cannot handle, a wrong
        # path -- looks exactly like a successful run that wrote a tiny file. Fail
        # loudly instead, so the job state reflects what happened.
        if len(keep) < args.min_success * n:
            raise SystemExit(
                f"[{split}] only {len(keep)}/{n} clips decoded "
                f"({100*len(keep)/n:.1f}%), below --min-success="
                f"{100*args.min_success:.0f}%. Nothing written for this split.")
        ids = [samples[i][0] for i in keep]

        np.save(os.path.join(args.out, f"{split}_video.npy"), v_seq[keep])
        np.save(os.path.join(args.out, f"{split}_audio.npy"), a_seq[keep])
        np.save(os.path.join(args.out, f"{split}_video_pooled.npy"), v_pool[keep])
        np.save(os.path.join(args.out, f"{split}_audio_pooled.npy"), a_pool[keep])
        np.save(os.path.join(args.out, f"{split}_labels.npy"), labels[keep])
        json.dump(ids, open(os.path.join(args.out, f"{split}_ids.json"), "w"))

        meta["splits"][split] = dict(
            n=int(len(keep)), dropped=int(n - len(keep)),
            video=list(v_seq.shape[1:]), audio=list(a_seq.shape[1:]),
            video_pooled=[int(v_pool.shape[1])], audio_pooled=[int(a_pool.shape[1])])
        print(f"[{split}] wrote {len(keep)} rows", flush=True)

    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print("[done] meta.json written", flush=True)
    for f in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, f)) / 1e6
        print(f"  {f:30s} {size:8.1f} MB")


if __name__ == "__main__":
    main()
