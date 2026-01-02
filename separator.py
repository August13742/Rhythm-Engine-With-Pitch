"""
separator.py
Splits audio into 6 stems: Drums, Bass, Other, Vocals, Guitar, Piano.
"""

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
    
    # LOAD 6-STEM MODEL
    # 'htdemucs_6s' adds 'guitar' and 'piano' to the standard split.
    model = get_model(name='htdemucs_6s')
    model.to(device)

    print(f"[AI] Loading {audio_path}...")
    wav, sr = librosa.load(audio_path, sr=44100, mono=False)
    if len(wav.shape) == 1: wav = np.stack([wav, wav])
    
    # Normalize
    ref_mean = wav.mean()
    ref_std = wav.std()
    wav = (wav - ref_mean) / (ref_std + 1e-8)
    wav_t = torch.tensor(wav).float().to(device)

    print(f"[AI] Separating into {model.sources}...")
    with torch.no_grad():
        # shifts=1 is fast, shifts=2 or 4 is better quality but slower
        sources = apply_model(model, wav_t[None], shifts=4, split=True, overlap=0.25, progress=True)[0]
    
    sources = sources.cpu().numpy()
    sources = (sources * ref_std) + ref_mean
    
    # Export Dynamic Stems
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    folder_path = os.path.join("stems", base_name) # Cleaner folder structure
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"[AI] Exporting Splits to '{folder_path}'...")
    
    # Dynamic loop using model.sources (No more hardcoded indices)
    # htdemucs_6s order: ["drums", "bass", "other", "vocals", "guitar", "piano"]
    for i, source_name in enumerate(model.sources):
        filename = f"{source_name}.wav" # simpler names: vocals.wav, drums.wav
        path = os.path.join(folder_path, filename)
        sf.write(path, sources[i].T, sr)
        print(f"    > Saved {filename}")
        
    print(f"[SUCCESS] Processing complete.")
    return folder_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    separate_audio(args.file)