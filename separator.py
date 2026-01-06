"""
SEPARATION ENGINE V2.1
Updates:
  - Stage 1: BS-Roformer (Stems)
  - Stage 2: Mel-Roformer-Viperx (Lead vs Backing) - Replaces UVR-BVE
  - Logic: Optimized for V208 Tri-Cameral Input
"""
import argparse
import os
import shutil
import logging
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import json
from audio_separator.separator import Separator
from huggingface_hub import hf_hub_download

# Suppress internal logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def ensure_custom_models_exist(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    
    # --- MODEL 1: BS-Roformer (Stems) ---
    rofo_files = {
        "BS-Rofo-SW-Fixed.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-Rofo-SW-Fixed.yaml": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
    }
    
    # --- MODEL 2: Mel-Roformer-Viperx (Lead vs Backing) ---
    # This model is SOTA for separating Main Vocals from "Accompaniment" (Backing)
    viper_files = {
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml": "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
    }

    print(f"[INIT] Checking Models...")
    
    # Download BS-Roformer
    for remote, local in rofo_files.items():
        target = os.path.join(model_dir, local)
        if not os.path.exists(target):
            try:
                print(f"  [DL] Downloading {remote}...")
                hf_hub_download(repo_id="jarredou/BS-ROFO-SW-Fixed", filename=remote, local_dir=model_dir, local_dir_use_symlinks=False)
                if remote != local and os.path.exists(os.path.join(model_dir, remote)):
                    os.rename(os.path.join(model_dir, remote), target)
            except Exception as e: print(f"  [ERR] {e}")

    # Download Viperx
    for remote, local in viper_files.items():
        target = os.path.join(model_dir, local)
        if not os.path.exists(target):
            try:
                print(f"  [DL] Downloading {remote}...")
                # Note: This is a common mirror for the Viperx model
                hf_hub_download(repo_id="jarredou/Mel-Band-Roformer-Karaoke-Aufr33-Viperx", filename=remote, local_dir=model_dir, local_dir_use_symlinks=False)
            except Exception as e: print(f"  [ERR] {e}")

def _prepare_audio(input_path: str, temp_dir: Path) -> str:
    """
    Normalizes audio to -1.0 dB to ensure BS-Roformer activates correctly.
    """
    clean_wav = temp_dir / "normalized_input.wav"
    print(f"[PREP] Analyzing gain levels for {Path(input_path).name}...")
    
    try:
        y, sr = librosa.load(input_path, sr=44100, mono=False)
        
        # Ensure Stereo
        if y.ndim == 1:
            y = np.stack([y, y])
            
        peak = np.max(np.abs(y))
        target_peak = 0.9  # -1.0 dB approx
        
        if peak < 0.001:
            return input_path 

        if peak < 0.8 or peak > 1.0:
            gain = target_peak / peak
            print(f"  [GAIN] Applying Gain: {gain:.2f}x (Peak {peak:.2f} -> {target_peak:.2f})")
            y = y * gain
        else:
            print(f"  [GAIN] Levels OK (Peak: {peak:.2f}).")

        sf.write(str(clean_wav), y.T, 44100, subtype='FLOAT')
        return str(clean_wav)

    except Exception as e:
        print(f"  [WARN] Normalization failed: {e}. Using original file.")
        return input_path

def separate_audio(audio_path: str, mode: str = "high"):
    base_name = Path(audio_path).stem
    root_dir = Path("stems") / base_name
    s1_dir = root_dir / "stage1_temp"
    s2_dir = root_dir / "stage2_temp"
    model_dir = Path(os.getcwd()) / "models"
    
    if root_dir.exists(): shutil.rmtree(root_dir)
    for d in [root_dir, s1_dir, s2_dir]: d.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing: {base_name} ---")
    ensure_custom_models_exist(str(model_dir))

    # NORMALIZE INPUT
    processing_file = _prepare_audio(audio_path, root_dir)
    
    # ---------------------------------------------------------
    # STAGE 1: 6-Stem Split (BS-Roformer)
    # ---------------------------------------------------------
    print(f"\n[1/2] BS-Roformer (Stems)...")
    sep_s1 = Separator(output_dir=str(s1_dir), model_file_dir=str(model_dir), output_format="WAV", log_level=logging.ERROR)
    sep_s1.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    sep_s1.separate(processing_file)
    
    s1_map = {"vocals": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other"}
    vocal_active = False
    
    for f in s1_dir.glob("*.wav"):
        for tag, target in s1_map.items():
            if f"({tag})" in f.name.lower():
                dest = root_dir / f"{target}.wav"
                shutil.move(str(f), str(dest))
                if target == "vocals": vocal_active = _check_activity(dest)

    # ---------------------------------------------------------
    # STAGE 2: Lead/Backing Split (Mel-Roformer-Viperx)
    # ---------------------------------------------------------
    # Viperx is a "Karaoke" model. 
    # Input: Vocals (Lead + Backing)
    # Output 1: "Vocals" -> This is the Lead
    # Output 2: "Instrumental" -> This is the Backing
    if vocal_active:
        print(f"\n[2/2] Mel-Roformer-Viperx (Lead/Backing)...")
        sep_s2 = Separator(output_dir=str(s2_dir), model_file_dir=str(model_dir), output_format="WAV", log_level=logging.ERROR)
        sep_s2.load_model('model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
        sep_s2.separate(str(root_dir / "vocals.wav"))
        
        for f in s2_dir.glob("*.wav"):
            # Viperx Output Mapping
            if "(Vocals)" in f.name:
                shutil.move(str(f), str(root_dir / "vocals_lead.wav"))
            elif "(Instrumental)" in f.name:
                shutil.move(str(f), str(root_dir / "vocals_backing.wav"))
    else:
        print(f"\n[SKIP] Stage 2 (No Vocals Detected)")
        _create_dummy(root_dir / "vocals_lead.wav")
        _create_dummy(root_dir / "vocals_backing.wav")

    # Cleanup
    shutil.rmtree(s1_dir)
    shutil.rmtree(s2_dir)
    if processing_file != audio_path and os.path.exists(processing_file):
        os.remove(processing_file)
    
    _optimize_stems(root_dir)
    _match_stem_lengths(root_dir)
    finalize_separation(str(root_dir))
    return str(root_dir)

# --- HELPER FUNCTIONS ---

def _check_activity(path: Path, threshold: float = 0.015) -> bool:
    if not path.exists(): return False
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames == 0: return False
        
        # Read the first 60 seconds (optimization)
        y, _ = sf.read(str(path), frames=44100*60) 
        if y.ndim > 1: peak = np.max(np.abs(y))
        else: peak = np.max(np.abs(y))
            
        return peak > threshold
    except Exception as e:
        print(f"  [WARN] Activity check failed for {path.name}: {e}")
        return False

def _create_dummy(path: Path, sr: int = 44100):
    sf.write(str(path), np.zeros(1024), sr)

def _optimize_stems(root_dir: Path):
    print("\n[OPTIMIZE] Cleaning silent stems...")
    for stem in root_dir.glob("*.wav"):
        if not _check_activity(stem):
            try:
                if stem.exists(): os.remove(str(stem))
                _create_dummy(stem)
            except Exception: pass

def _match_stem_lengths(root_dir: Path):
    import soundfile as sf
    stems = list(root_dir.glob("*.wav"))
    max_len = 0
    reference_sr = 44100
    active_stems = []

    for stem in stems:
        try:
            info = sf.info(str(stem))
            if info.duration > 5.0:
                active_stems.append(stem)
                if info.frames > max_len:
                    max_len = info.frames
                    reference_sr = info.samplerate
        except Exception: pass

    if max_len == 0: return

    print(f"  [SYNC] Aligning {len(active_stems)} active stems to {max_len/reference_sr:.2f}s...")
    
    for stem in active_stems:
        try:
            y, _ = librosa.load(str(stem), sr=reference_sr)
            current_len = len(y)
            
            if current_len != max_len:
                if current_len < max_len:
                    y = np.pad(y, (0, max_len - current_len))
                elif current_len > max_len:
                    y = y[:max_len]
                sf.write(str(stem), y, reference_sr)
        except Exception as e:
            print(f"  [WARN] Failed to sync {stem.name}: {e}")

def finalize_separation(output_dir: str):
    print("\n--- Analysis ---")
    stems = ["vocals", "vocals_lead", "vocals_backing", "drums", "bass", "guitar", "piano", "other"]
    manifest = {}
    stats = {"lead": 0.0, "back": 0.0}
    
    for stem in stems:
        path = os.path.join(output_dir, f"{stem}.wav")
        info = {"exists": False, "peak_energy": 0.0}
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None)
            peak = float(np.max(np.abs(y)))
            rms = float(np.sqrt(np.mean(y**2)))
            info = {"exists": True, "peak_energy": peak, "is_silent": peak < 0.02}
            if stem == "vocals_lead": stats["lead"] = rms
            if stem == "vocals_backing": stats["back"] = rms
            state = "ACTIVE" if not info["is_silent"] else "_"
            print(f"  > {stem:<15} : {state:<6} (Peak: {peak:.2f})")
        manifest[stem] = info

    ratio = (stats["back"] / stats["lead"]) if stats["lead"] > 0 else 0.0
    manifest["vocal_type"] = "polyphonic" if (stats["back"] > 0.01 and ratio > 0.15) else "monophonic"
    print(f"[DECISION] Vocal Mode: {manifest['vocal_type'].upper()}")
    
    with open(os.path.join(output_dir, "stems_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--mode", default="high")
    args = parser.parse_args()
    separate_audio(args.file, mode=args.mode)