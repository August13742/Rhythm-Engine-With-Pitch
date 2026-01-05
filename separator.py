"""
V204.2 - SEPARATOR ENGINE
Changes:
  - Architecture: Isolation Chambers (stage1_temp / stage2_temp) to prevent tag collision.
  - Logic: Inverted BVE Mapping (Vocals->Backing, Inst->Lead) based on testing.
  - Optimization: Early Exit (skips Stage 2 if vocals are silent).
  - Storage: Silent stems are overwritten with 1KB dummy files.
  - Stability: Length matching to prevent Visualizer drift.
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
    """Checks for required models. Downloads Roformer if missing."""
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. BS-Roformer (Stage 1: 6-Stem SOTA)
    rofo_files = {
        "BS-Rofo-SW-Fixed.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-Rofo-SW-Fixed.yaml": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
    }
    print(f"[INIT] Checking BS-Roformer Model...")
    for remote, local in rofo_files.items():
        target = os.path.join(model_dir, local)
        if not os.path.exists(target):
            print(f"  > Downloading {local}...")
            try:
                hf_hub_download(
                    repo_id="jarredou/BS-ROFO-SW-Fixed",
                    filename=remote,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                # Rename if necessary (hf_download sometimes keeps original name)
                downloaded_path = os.path.join(model_dir, remote)
                if remote != local and os.path.exists(downloaded_path):
                    os.rename(downloaded_path, target)
            except Exception as e:
                print(f"  [ERROR] Failed to download Roformer: {e}")

    # 2. UVR-BVE (Stage 2: Lead/Backing)
    # Manual check since automatic download is often 401/Gatekept
    bve_model = os.path.join(model_dir, "UVR-BVE-4B_SN-44100-1.pth")
    if not os.path.exists(bve_model):
        print(f"  [WARN] UVR-BVE-4B_SN-44100-1.pth missing in {model_dir}!")
        print("         Stage 2 will fail if needed. Please place file manually.")

def separate_audio(audio_path: str, mode: str = "high"):
    base_name = Path(audio_path).stem
    root_dir = Path("stems") / base_name
    
    # Isolation Chambers to prevent file collision
    s1_dir = root_dir / "stage1_temp"
    s2_dir = root_dir / "stage2_temp"
    model_dir = Path(os.getcwd()) / "models"
    
    # Reset Environment
    if root_dir.exists(): shutil.rmtree(root_dir)
    for d in [root_dir, s1_dir, s2_dir]: d.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing: {base_name} ---")
    ensure_custom_models_exist(str(model_dir))
    
    # ---------------------------------------------------------
    # STAGE 1: 6-Stem Split (BS-Roformer)
    # ---------------------------------------------------------
    print(f"\n[1/2] Stage 1: Initial Split (BS-Roformer)")
    sep_s1 = Separator(output_dir=str(s1_dir), model_file_dir=str(model_dir), output_format="WAV", log_level=logging.ERROR)
    sep_s1.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    sep_s1.separate(audio_path)
    
    # Precise movement from Isolation Chamber 1 -> Root
    s1_map = {"vocals": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "piano", "other": "other"}
    vocal_is_active = False
    
    for f in s1_dir.glob("*.wav"):
        for tag, target_name in s1_map.items():
            if f"({tag})" in f.name.lower():
                dest = root_dir / f"{target_name}.wav"
                shutil.move(str(f), str(dest))
                print(f"  [S1 -> ROOT] {target_name}.wav")
                
                # Immediate Detection for Stage 2 Decision
                if target_name == "vocals":
                    vocal_is_active = _check_activity(dest)

    # ---------------------------------------------------------
    # STAGE 2: Lead/Backing Split (Conditional)
    # ---------------------------------------------------------
    if vocal_is_active:
        print(f"\n[2/2] Stage 2: Lead/Backing Split (BVE)")
        sep_s2 = Separator(output_dir=str(s2_dir), model_file_dir=str(model_dir), output_format="WAV", log_level=logging.ERROR)
        sep_s2.load_model('UVR-BVE-4B_SN-44100-1.pth')
        sep_s2.separate(str(root_dir / "vocals.wav"))
        
        # BVE VR-Arch Mapping (INVERTED based on testing)
        # (Vocals) -> Often catches the "Target" (Backing/Noise) in BVE logic
        # (Instrumental) -> Often keeps the "Residual" (Main Lead)
        for f in s2_dir.glob("*.wav"):
            if "(Vocals)" in f.name:
                shutil.move(str(f), str(root_dir / "vocals_backing.wav"))
                print("  [S2 -> ROOT] vocals_backing.wav (Target Extracted)")
            elif "(Instrumental)" in f.name:
                shutil.move(str(f), str(root_dir / "vocals_lead.wav"))
                print("  [S2 -> ROOT] vocals_lead.wav (Main Signal)")
        
        # VALIDATION: Did we accidentally kill the lead?
        lead_active = _check_activity(root_dir / "vocals_lead.wav")
        if not lead_active:
            print(f"\n[!!! WARNING !!!] Stage 2 neutralized the Lead Vocal.")
            print(f"    > Lead stem is silent. BVE may have misinterpreted the signal.")
    else:
        print(f"\n[SKIP] Stage 2 bypassed: No vocal energy detected in Stage 1.")
        # Create dummy files for lead/back to satisfy Visualizer requirements
        _create_dummy(root_dir / "vocals_lead.wav")
        _create_dummy(root_dir / "vocals_backing.wav")

    # ---------------------------------------------------------
    # CLEANUP & OPTIMIZATION
    # ---------------------------------------------------------
    shutil.rmtree(s1_dir)
    shutil.rmtree(s2_dir)
    
    # 1. Overwrite silent stems with 1KB dummy files
    _optimize_stems(root_dir)
    
    # 2. Ensure all stems match exact length of vocals (prevents visualizer drift)
    _match_stem_lengths(root_dir)
    
    finalize_separation(str(root_dir))
    return str(root_dir)

# --- HELPER FUNCTIONS ---

def _check_activity(path: Path, threshold: float = 0.015) -> bool:
    """Mechanistic check for audio energy above noise floor."""
    if not path.exists(): return False
    try:
        # Check first 30s for speed
        y, _ = librosa.load(str(path), sr=None, duration=30) 
        if len(y) == 0: return False
        return float(np.max(np.abs(y))) > threshold
    except Exception:
        return False

def _create_dummy(path: Path, sr: int = 44100):
    """Creates a minimal silence file to save disk space."""
    sf.write(str(path), np.zeros(1024), sr)
    # print(f"  [DUMMY] Created minimal silent stem: {path.name}")

def _optimize_stems(root_dir: Path):
    """Replaces non-active stems with dummy files across the entire project."""
    print("\n[OPTIMIZE] Checking for silent stems to compress...")
    for stem in root_dir.glob("*.wav"):
        if not _check_activity(stem):
            try:
                # Remove the existing file first to avoid lock issues
                if stem.exists():
                    os.remove(str(stem))
                _create_dummy(stem)
                print(f"  [DUMMY] Replaced silent stem: {stem.name}")
            except Exception as e:
                print(f"  [ERROR] Failed to replace {stem.name} with dummy: {e}")

def _match_stem_lengths(root_dir: Path):
    """Ensures all stems match the duration of the original vocals."""
    target_file = root_dir / "vocals.wav"
    if not target_file.exists(): return
    
    y_target, sr = librosa.load(str(target_file), sr=None)
    target_len = len(y_target)
    
    for stem in root_dir.glob("*.wav"):
        if stem.name == "vocals.wav": continue
        try:
            y, _ = librosa.load(str(stem), sr=sr)
            if len(y) != target_len:
                if len(y) < target_len:
                    y_padded = np.pad(y, (0, target_len - len(y)))
                    sf.write(str(stem), y_padded, sr)
                elif len(y) > target_len:
                    sf.write(str(stem), y[:target_len], sr)
        except Exception as e:
            print(f"  [WARN] Length match failed for {stem.name}: {e}")

def finalize_separation(output_dir: str):
    print("\n--- Analysis & Manifest ---")
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
            
            info["exists"] = True
            info["peak_energy"] = peak
            info["is_silent"] = peak < 0.02
            
            if stem == "vocals_lead": stats["lead"] = rms
            if stem == "vocals_backing": stats["back"] = rms
            
            # If we optimized it to a dummy earlier, it shows as silent here naturally
            state = "ACTIVE" if not info["is_silent"] else "_"
            print(f"  > {stem:<15} : {state:<6} (Peak: {peak:.2f})")
            
        manifest[stem] = info

    # DSP Decision: Choir Mode
    # If backing vocals are significant (> 15% of lead), enable Polyphonic mode
    ratio = (stats["back"] / stats["lead"]) if stats["lead"] > 0 else 0.0
    is_choir = (stats["back"] > 0.01) and (ratio > 0.15)
    
    manifest["vocal_type"] = "polyphonic" if is_choir else "monophonic"
    print(f"\n[DECISION] Vocal Mode: {manifest['vocal_type'].upper()} (Backing Ratio: {ratio:.2f})")
    
    with open(os.path.join(output_dir, "stems_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--mode", default="high")
    args = parser.parse_args()
    separate_audio(args.file, mode=args.mode)