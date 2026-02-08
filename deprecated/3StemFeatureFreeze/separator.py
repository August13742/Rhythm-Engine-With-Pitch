'''separator.py'''

import argparse
import os
import torch
import librosa
import numpy as np
import soundfile as sf
from demucs.apply import apply_model
from demucs.pretrained import get_model

def separate_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AI] Initializing Demucs on {device}...")
    model = get_model(name='htdemucs')
    model.to(device)

    print(f"[AI] Loading {audio_path}...")
    wav, sr = librosa.load(audio_path, sr=44100, mono=False)
    if len(wav.shape) == 1: wav = np.stack([wav, wav])
    
    # Normalize
    ref_mean = wav.mean()
    ref_std = wav.std()
    wav = (wav - ref_mean) / (ref_std + 1e-8)
    wav_t = torch.tensor(wav).float().to(device)

    print("[AI] Separating Stems...")
    with torch.no_grad():
        sources = apply_model(model, wav_t[None], shifts=1, split=True, overlap=0.25, progress=True)[0]
    
    sources = sources.cpu().numpy()
    
    # Denormalize sources back to original range
    sources = (sources * ref_std) + ref_mean
    
    # Demucs: 0=Drums, 1=Bass, 2=Other, 3=Vocals
    print("[AI] Exporting 3-Way Split...")
    
    # 1. VOCALS (Pure Melody)
    vocals = sources[3]
    
    # 2. INSTRUMENTS (Backup Melody)
    other = sources[2]
    
    # 3. RHYTHM (Timing Grid)
    rhythm = sources[0] + sources[1] # Drums + Bass
    
    # Create folder for the song
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    folder_path = base_name
    os.makedirs(folder_path, exist_ok=True)
    
    paths = {
        "vocals": os.path.join(folder_path, f"{base_name}_vocals.wav"),
        "other": os.path.join(folder_path, f"{base_name}_other.wav"),
        "rhythm": os.path.join(folder_path, f"{base_name}_rhythm.wav")
    }
    
    sf.write(paths["vocals"], vocals.T, sr)
    sf.write(paths["other"], other.T, sr)
    sf.write(paths["rhythm"], rhythm.T, sr)
    
    print(f"[SUCCESS] Splits ready in '{folder_path}/' folder.")
    return folder_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    separate_audio(args.file)