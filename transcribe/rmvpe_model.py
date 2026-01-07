"""
RMVPE (Robust Model for Vocal Pitch Estimation) Architecture
Official RVC RMVPE implementation for pitch detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import requests
import numpy as np


# ==========================================
#      MODEL ARCHITECTURE
# ==========================================

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(Encoder(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(Encoder(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        # Handle size mismatch by interpolating
        if x.shape[2:] != concat_tensor.shape[2:]:
            x = torch.nn.functional.interpolate(x, size=concat_tensor.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = nn.ModuleList()
        # Intermediate doubles channels: 256 -> 512
        self.intermediate = Intermediate(en_out_channels * (2 ** (en_de_layers - 1)), en_out_channels * (2 ** en_de_layers), inter_layers, n_blocks)
        self.decoder = nn.ModuleList()

        for i in range(en_de_layers):
            self.encoder.append(
                Encoder(
                    in_channels,
                    en_out_channels * (2**i),
                    kernel_size,
                    n_blocks,
                )
            )
            in_channels = en_out_channels * (2**i)

        for i in range(en_de_layers):
            self.decoder.append(
                Decoder(
                    en_out_channels * (2 ** (en_de_layers - i)),
                    en_out_channels * (2 ** (en_de_layers - i - 1)),
                    kernel_size,
                    n_blocks,
                )
            )

    def forward(self, x):
        xs = []
        for encoder in self.encoder:
            x_before_pool, x = encoder(x)
            xs.append(x_before_pool)
        x = self.intermediate(x)
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, xs[-1 - i])
        return x


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


# ==========================================
#      INFERENCE WRAPPER
# ==========================================

class RMVPE_Infer:
    def __init__(self, model_path, device):
        self.device = device
        self.model = E2E(n_blocks=4, n_gru=2, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16).to(device)
        self.model.eval()
        
        # Mel Spectrogram (Matches RVC/RMVPE Training Config)
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            f_min=40,
            f_max=8000,
            n_mels=128,
            center=True,
            power=1.0,
        ).to(device)
        
        self._load_weights(model_path)
        
        # Cents Mapping (0 to 7200 cents approx)
        self.cents_mapping = torch.linspace(0, 7180, 360).to(device) + 1997.37940586230

    def _load_weights(self, path):
        if not os.path.exists(path):
            print(f"[AI] RMVPE weights missing. Downloading...")
            self._download(path)
        
        print(f"[AI] Loading RMVPE from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            # If it's a checkpoint dict with 'model' key, extract it
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[6:]
            new_state[k] = v
            
        try:
            self.model.load_state_dict(new_state, strict=False)
            print("     > Success: RMVPE model loaded.")
        except RuntimeError as e:
            print(f"     > [FATAL] Model load failed.")
            print(e)
            raise e

    def _download(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    def infer(self, audio, thred=0.03):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float().to(self.device)
        if audio.dim() == 1: 
            audio = audio.unsqueeze(0)
        
        with torch.no_grad():
            mel = self.mel_extractor(audio)
            # Log Mel is crucial for RMVPE
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            # Forward pass
            hidden = self.model(mel)  # [B, T, 360]
            
            # Decode to F0
            hidden = hidden.squeeze(0)  # [T, 360]
            
            # Apply confidence threshold using max probability
            max_vals, _ = hidden.max(dim=1)
            mask = max_vals > thred
            
            # Calculate weighted frequency
            cents = torch.sum(hidden * self.cents_mapping, dim=1) / (torch.sum(hidden, dim=1) + 1e-8)
            f0 = 10 * (2 ** (cents / 1200))
            
            # Apply mask
            f0 = f0 * mask.float()
            
            return f0.cpu().numpy()
