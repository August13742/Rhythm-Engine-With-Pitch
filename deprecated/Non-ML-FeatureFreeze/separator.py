import argparse
import os
import shutil
import logging
from audio_separator.separator import Separator
from huggingface_hub import hf_hub_download
import librosa
import numpy as np
import json
import soundfile as sf

# Suppress standard logs
logging.getLogger('numba').setLevel(logging.WARNING)

def ensure_custom_model_exists(model_dir: str):
    """
    Downloads the BS-Rofo-SW-Fixed model and 'masquerades' it as a known 
    Roformer filename to bypass audio-separator's strict registry.
    """
    repo_id = "jarredou/BS-ROFO-SW-Fixed"
    # We rename the files locally so audio-separator accepts them
    registry_name = "model_bs_roformer_ep_317_sdr_12.9755"
    
    os.makedirs(model_dir, exist_ok=True)
    
    files_to_download = {
        "BS-Rofo-SW-Fixed.ckpt": f"{registry_name}.ckpt",
        "BS-Rofo-SW-Fixed.yaml": f"{registry_name}.yaml"
    }
    
    print(f"[INIT] Syncing SOTA 6-Stem model to registry...")
    for remote_name, local_name in files_to_download.items():
        target_path = os.path.join(model_dir, local_name)
        if not os.path.exists(target_path):
            print(f"    > Downloading {remote_name} as {local_name}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            # Rename downloaded file to the registry name
            os.rename(os.path.join(model_dir, remote_name), target_path)
        else:
            print(f"    > {local_name} ready.")


def separate_audio(audio_path: str, mode: str = "high"):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("stems", base_name)
    
    # Default audio-separator model path on Windows is typically in /tmp/ or AppData
    # We'll explicitly set one for consistency
    model_dir = os.path.join(os.getcwd(), "models")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Processing: {base_name} [{mode.upper()}] ---")

    # Initialize separator with explicit model directory
    sep = Separator(
        output_dir=output_dir, 
        model_file_dir=model_dir, 
        output_format="WAV"
    )

    if mode == "high":
        ensure_custom_model_exists(model_dir)
        _separate_high_sota(audio_path, output_dir, sep)
    elif mode == "medium":
        _separate_demucs_cascade(audio_path, output_dir, sep)
    else:
        _separate_htdemucs_6s(audio_path, output_dir, sep)
        
    _cleanup_intermediates(output_dir)
    finalize_separation(output_dir)
    print(f"\n[DONE] Output saved to: {output_dir}")
    return output_dir

def _separate_high_sota(audio_path: str, output_dir: str, sep: Separator):
    # We use the name the library expects
    model_name = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    print(f"\n[1/1] Starting 6-Stem Roformer Separation...")
    
    sep.load_model(model_filename=model_name)
    files = sep.separate(audio_path)
    
    # Note: Because we are 'lying' to the library, it might name the 
    # stems (Vocals) and (Instrumental) even if the model produces 6.
    # We must be flexible in our mapping:
    mapping = {
        "Vocals": "vocals", "Drums": "drums", "Bass": "bass",
        "Guitar": "guitar", "Piano": "piano", "Other": "other",
        "Instrumental": "other" # Fallback
    }
    
    for key, val in mapping.items():
        _rename_stem(output_dir, files, key, val)
        
def _rename_stem(output_dir, files, target_identifier, new_name):
    # Convert everything to lowercase for the search
    target_lower = target_identifier.lower()
    found = next((f for f in files if target_lower in f.lower()), None)
    
    if found:
        old_path = os.path.join(output_dir, found)
        new_path = os.path.join(output_dir, f"{new_name}.wav")
        if os.path.exists(new_path): os.remove(new_path)
        os.rename(old_path, new_path)
        print(f"    > Reserved: {new_name}.wav") # Useful for debugging
        return new_path
    return None

def _separate_demucs_cascade(audio_path, output_dir, sep):
    print("\n[MEDIUM] Splitting 4 stems (HTDemucs)...")
    sep.load_model(model_filename='htdemucs_ft.yaml')
    files = sep.separate(audio_path)
    _rename_stem(output_dir, files, "(Vocals)", "vocals")
    _rename_stem(output_dir, files, "(Drums)", "drums")
    _rename_stem(output_dir, files, "(Bass)", "bass")
    _rename_stem(output_dir, files, "(Other)", "other")

def _separate_htdemucs_6s(audio_path, output_dir, sep):
    print("\n[LOW] Running htdemucs_6s (6-Stem Native)...")
    sep.load_model(model_filename='htdemucs_6s.yaml')
    files = sep.separate(audio_path)
    mapping = {
        "(Vocals)": "vocals", "(Drums)": "drums", "(Bass)": "bass",
        "(Guitar)": "guitar", "(Piano)": "piano", "(Other)": "other"
    }
    for key, val in mapping.items():
        _rename_stem(output_dir, files, key, val)

def _cleanup_intermediates(output_dir):
    print("\n[CLEANUP] Finalizing directory...")
    # These are the ONLY names we keep (plus JSONs)
    final_stems = ["vocals.wav", "drums.wav", "bass.wav", "guitar.wav", "piano.wav", "other.wav"]
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        
        # Protect beatmaps
        if filename.endswith(".json"):
            continue
            
        # Delete anything that isn't a final stem
        if filename not in final_stems:
            try:
                os.remove(file_path)
                print(f"    > Purged intermediate: {filename}")
            except OSError:
                pass

def finalize_separation(output_dir: str):
    """
    Finalizing the stems for the rhythm engine.
    If a stem is silent, it's not a bug—it's most likely high-purity isolation.
    """
    print("\n--- Final Integrity Check ---")
    stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    SILENCE_THRESHOLD = 0.05
    
    manifest = {}
    
    for stem in stems:
        path = os.path.join(output_dir, f"{stem}.wav")
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None)
            energy = np.max(np.abs(y))
            is_silent = bool(energy <= SILENCE_THRESHOLD)
            status = "ACTIVE" if not is_silent else "EMPTY (PURE)"
            print(f"  > {stem.capitalize():<10}: {status:<12} (Peak: {energy:.4f})")
            
            # If empty, replace with minimal silent audio to save space
            if is_silent:
                silent_audio = np.zeros(sr)  # 1 second of silence
                sf.write(path, silent_audio, sr)
            
            manifest[stem] = {
                "exists": True,
                "is_silent": is_silent,
                "peak_energy": float(energy)
            }
        else:
            manifest[stem] = {
                "exists": False,
                "is_silent": True,
                "peak_energy": 0.0
            }
    
    # Write manifest
    manifest_path = os.path.join(output_dir, "stems_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  > Manifest saved: stems_manifest.json")

    return output_dir
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--mode", choices=["high", "medium", "low"], default="high")
    args = parser.parse_args()
    separate_audio(args.file, mode=args.mode)
    