"""
RMVPE (Robust Model for Vocal Pitch Estimation) Architecture
Official RVC RMVPE implementation for pitch detection.
Corrected to match the official checkpoint structure from RVC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import numpy as np
from typing import List
from librosa.filters import mel as librosa_mel


# ==========================================
#      MODEL ARCHITECTURE (Matches RVC Checkpoint)
# ==========================================

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


# ==========================================
#      MEL SPECTROGRAM (Matches RVC)
# ==========================================

class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        keyshift_key = str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(self.win_length).to(audio.device)
        
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis.to(magnitude.device), magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


# ==========================================
#      INFERENCE WRAPPER
# ==========================================

class RMVPE_Infer:
    def __init__(self, model_path, device):
        self.device = device
        # n_gru=1 matches the official RVC checkpoint!
        self.model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16).to(device)
        self.model.eval()
        
        # Mel Spectrogram (Matches RVC/RMVPE Training Config)
        self.mel_extractor = MelSpectrogram(
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=None,
            mel_fmin=30,
            mel_fmax=8000,
            clamp=1e-5,
        ).to(device)
        
        self._load_weights(model_path)
        
        # Cents Mapping (Matches RVC: 20 * np.arange(360) + 1997.3794084376191)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # Pad to 368 for local average

    def _load_weights(self, path):
        if not os.path.exists(path):
            print(f"[AI] RMVPE weights missing. Downloading...")
            self._download(path)
        
        print(f"[AI] Loading RMVPE from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        try:
            self.model.load_state_dict(checkpoint, strict=True)
            print("     > Success: RMVPE model loaded (strict=True).")
        except RuntimeError as e:
            print(f"     > [FATAL] Model load failed - architecture mismatch!")
            print(e)
            raise e

    def _download(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
        print(f"[AI] Downloading RMVPE weights from {url}...")
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[AI] Download complete: {path}")

    def to_local_average_cents(self, salience, thred=0.03):
        """
        Convert salience (prob per bin) to cents using local weighted average.
        This is the official RVC decoding method.
        """
        center = np.argmax(salience, axis=1)  # [T] - index of max
        salience = np.pad(salience, ((0, 0), (4, 4)))  # Pad to [T, 368]
        
        todo_salience = []
        todo_cents_mapping = []
        starts = center  # After padding, original center is at center+4, so window is center to center+9
        ends = center + 9
        
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx]:ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
        
        todo_salience = np.array(todo_salience)  # [T, 9]
        todo_cents_mapping = np.array(todo_cents_mapping)  # [T, 9]
        
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        devided = product_sum / (weight_sum + 1e-8)
        
        # Apply threshold using max probability
        maxx = np.max(salience[:, 4:-4], axis=1)  # Original unpadded max
        devided[maxx <= thred] = 0
        
        return devided

    def infer(self, audio, thred=0.03):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float().to(self.device)
        if audio.dim() == 1: 
            audio = audio.unsqueeze(0)
        
        with torch.no_grad():
            mel = self.mel_extractor(audio, center=True)
            
            # Pad to multiple of 32
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            
            # Forward pass
            hidden = self.model(mel)  # [B, T, 360]
            hidden = hidden[:, :n_frames]  # Remove padding
            
            # Decode to F0 using local average cents
            hidden = hidden.squeeze(0).cpu().numpy()  # [T, 360]
            cents_pred = self.to_local_average_cents(hidden, thred=thred)
            
            # Convert cents to F0
            f0 = 10 * (2 ** (cents_pred / 1200))
            f0[f0 == 10] = 0  # Where cents_pred was 0
            
            return f0
