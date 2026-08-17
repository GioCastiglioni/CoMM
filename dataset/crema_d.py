import os
import json
import torch
from pytorch_lightning import LightningDataModule
from torchvision.models.video import R3D_18_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io
import torchaudio
from utils import make_dirs, GaussianBlur

class CREMADDataModule(LightningDataModule):
    def __init__(self, model: str,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 **kwargs):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        catalog_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "catalog.json")
        with open(catalog_path) as f:
            self.catalog = json.load(f)
        self.root = self.catalog["crema_d"]["path"]

        # Base Video transform: R3D_18 default transform
        self.video_transform = R3D_18_Weights.KINETICS400_V1.transforms()

        self.audio_transform = transforms.Compose([
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512),
            torchaudio.transforms.AmplitudeToDB(),
            transforms.Resize((224, 224), antialias=True)
        ])
        self.audio_augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        ])

        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])

        self.video_augment = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.08, 1.), antialias=True),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.setup()

    def setup(self, stage=None):
        if self.model == "Sup":
            self.train_dataset = CREMADDatasetSup(self.root, split="train", video_transform=self.video_transform, audio_transform=self.audio_transform)
            self.val_dataset = CREMADDatasetSup(self.root, split="test", video_transform=self.video_transform, audio_transform=self.audio_transform)
        elif self.model == "CoMM" or self.model == "WoMM":
            self.train_dataset = CREMADDatasetMMSSL(self.root, split="train", video_transform=self.video_transform, audio_transform=self.audio_transform, video_augment=self.video_augment, audio_augment=self.audio_augment)
            self.val_dataset = CREMADDatasetMMSSL(self.root, split="test", video_transform=self.video_transform, audio_transform=self.audio_transform, video_augment=self.video_augment, audio_augment=self.audio_augment)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=True)


class CREMADDatasetBase(Dataset):
    EMOTIONS = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

    def __init__(self, root, split="train", num_frames=16):
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.audio_dir = os.path.join(root, "AudioWAV")
        self.video_dir = os.path.join(root, "VideoFlash")
        self.samples = []

        video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.flv')]
        video_files.sort()
        
        train_actors = set([str(i) for i in range(1001, 1071)])
        test_actors = set([str(i) for i in range(1071, 1092)])

        for vf in video_files:
            base_name = vf[:-4] # remove .flv
            actor_id = base_name.split("_")[0]
            if split == "train" and actor_id not in train_actors:
                continue
            if split == "test" and actor_id not in test_actors:
                continue

            af = base_name + ".wav"
            emo = base_name.split("_")[2]
            if emo in self.EMOTIONS:
                self.samples.append({
                    "video_path": os.path.join(self.video_dir, vf),
                    "audio_path": os.path.join(self.audio_dir, af),
                    "label": self.EMOTIONS[emo]
                })

    def _sample_frames(self, video_tensor):
        T = video_tensor.shape[0]
        if T == 0:
            return torch.zeros((self.num_frames, 3, 112, 112)) # dummy
        indices = torch.linspace(0, T - 1, self.num_frames).long()
        sampled = video_tensor[indices]
        # To (T, C, H, W)
        sampled = sampled.permute(0, 3, 1, 2)
        return sampled

    def __len__(self):
        return len(self.samples)

    def get_raw_item(self, idx):
        sample = self.samples[idx]
        
        # Audio
        waveform, sample_rate = torchaudio.load(sample["audio_path"])
        
        target_length = 32000
        if waveform.shape[1] > target_length:
            diff=(waveform.shape[1] - target_length)
            waveform = waveform[:, diff//2 : diff//2 + target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Video
        video_tensor, audio, info = torchvision.io.read_video(sample["video_path"], pts_unit="sec")
        sampled_video = self._sample_frames(video_tensor)

        return sampled_video, waveform, sample["label"]


class CREMADDatasetSup(CREMADDatasetBase):
    def __init__(self, root, split="train", video_transform=None, audio_transform=None):
        super().__init__(root, split=split)
        self.video_transform = video_transform
        self.audio_transform = audio_transform

    def __getitem__(self, idx):
        video, audio, label = self.get_raw_item(idx)
        if self.video_transform is not None:
            video = self.video_transform(video)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
            
        return [video, audio], label


class CREMADDatasetMMSSL(CREMADDatasetBase):
    def __init__(self, root, split="train", video_transform=None, audio_transform=None, video_augment=None, audio_augment=None):
        super().__init__(root, split=split)
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.video_augment = video_augment
        self.audio_augment = audio_augment

    def __getitem__(self, idx):
        video, audio, label = self.get_raw_item(idx)
        
        # Scale to 0-1 for video augmentations
        video = video.float() / 255.0

        if self.video_augment is not None:
            video_aug1 = self.video_augment(video).permute(1, 0, 2, 3) # (C, T, H, W)
            video_aug2 = self.video_augment(video).permute(1, 0, 2, 3)
        elif self.video_transform is not None:
            video_aug1 = self.video_transform(video)
            video_aug2 = self.video_transform(video)
        else:
            video_aug1 = video
            video_aug2 = video

        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        if self.audio_augment is not None:
            audio_aug1 = self.audio_augment(audio)
            audio_aug2 = self.audio_augment(audio)
        else:
            audio_aug1 = audio
            audio_aug2 = audio

        return [video_aug1, audio_aug1], [video_aug2, audio_aug2]

