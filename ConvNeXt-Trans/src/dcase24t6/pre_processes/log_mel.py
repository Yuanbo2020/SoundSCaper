import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from typing import Any, Dict, Union
from pathlib import Path
import copy
from dcase24t6.nn.ckpt import CNEXT_REGISTRY
from dcase24t6.nn.encoders.convnext import convnext_tiny
from dcase24t6.nn.functional import remove_index_nd
from dcase24t6.pre_processes.common import (
    batchify,
    is_audio_batch,
    sanitize_batch,
    unbatchify,
)
from dcase24t6.pre_processes.resample import Resample

class ResampleLogMel(nn.Module):
    """Offline transform applied to audio inputs for log-mel feature extraction.

    This module handles single example and batch of examples as input.
    """

    def __init__(
        self,
        model_sr: int = 44100,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 256,
        device: Union[str, torch.device, None] = "cuda_if_available",
        input_time_dim: int = -1,
        keep_batch: bool = False,
    ) -> None:
        # Call super().__init__() first
        super().__init__()

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "cuda_if_available" else device

        # Initialize resample module
        resample = Resample(model_sr, input_time_dim)

        # Initialize Log-Mel feature extraction
        self.mel_transform = MelSpectrogram(
            sample_rate=model_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(device)
        self.amplitude_to_db = AmplitudeToDB().to(device)

        self.resample = resample
        self.input_time_dim = input_time_dim
        self.keep_batch = keep_batch
        self.device = device

    def forward(self, item_or_batch: Dict[str, Any]) -> Dict[str, Any]:
        if is_audio_batch(item_or_batch):
            batch = sanitize_batch(item_or_batch)
            return self.forward_batch(batch)

        item = item_or_batch
        batch = batchify(item)
        batch = self.forward_batch(batch)
        if self.keep_batch:
            return batch
        else:
            item = unbatchify(batch)
            return item

    def forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Resample audio
        batch = self.resample(batch)

        # Copy batch and extract audio
        batch = copy.copy(batch)
        audio = batch.pop("audio")
        audio_shape = batch.pop("audio_shape")

        # Remove channel dim by taking mean
        channel_dim = 1
        audio = audio.mean(dim=channel_dim)
        audio_shape = remove_index_nd(audio_shape, index=channel_dim - 1, dim=1)

        # Move audio to device
        audio = audio.to(device=self.device)

        # Extract Log-Mel features
        mel_spec = self.mel_transform(audio)  # (batch, n_mels, time)
        log_mel = self.amplitude_to_db(mel_spec)  # (batch, n_mels, time)

        # Save Log-Mel features as frame_embs and frame_embs_shape
        batch["frame_embs"] = log_mel.unsqueeze(0).transpose(2, 3)  # (batch, n_mels, time)
        log_mel1 = log_mel.squeeze(dim=0)
        batch["frame_embs_shape"] = torch.tensor([log_mel1.shape], device=self.device)  # (n_mels, time)

        return batch