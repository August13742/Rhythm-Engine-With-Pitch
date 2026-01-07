"""
separator.py
Orchestrator for audio separation and transcription.
"""
import argparse
import os
import shutil
import logging
import json
import subprocess
import sys
import numpy as np
import soundfile as sf
import librosa

# Suppress standard logs
logging.getLogger('numba').setLevel(logging.WARNING)

def generate_speech_notes(vocals_path: str, output_dir: str = None, use_holds: bool = True) -> list:
    """
    Main entry point.
    Extracts pitch data using a Voting System (Crepe + pYIN).
    """
    if output_dir is None:
        output_dir = os.path.dirname(vocals_path)

    worker_script = os.path.join(os.getcwd(), "transcribe_worker.py")
    temp_json = os.path.join(output_dir, "temp_transcription.json")

    print(f"\n[SPEECH] Launching Subprocess Worker...")
    try:
        subprocess.run([sys.executable, worker_script, vocals_path, temp_json], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Worker subprocess failed with code {e.returncode}.")
        raise

    print(f"[SPEECH] Loading worker results...")
    with open(temp_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if os.path.exists(temp_json): os.remove(temp_json)

    anchors = data['anchors']
    pitch = np.array(data['pitch'])
    conf = np.array(data['conf'])
    times = np.array(data['times'])

    # Load audio for pYIN voting
    print(f"[SPEECH] Loading audio for Consensus Voting...")
    y, sr = librosa.load(vocals_path, sr=None)

    return _construct_voting_notes(anchors, pitch, conf, times, y, sr, use_holds)

def _construct_voting_notes(anchors, pitch, conf, times, y, sr, use_holds):
    notes = []
    
    # pYIN Params
    fmin, fmax = 60, 500
    
    for w in anchors:
        t_start = w['start']
        t_end = w['end']
        dur = t_end - t_start
        
        # 1. Crepe Data
        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)
        
        crepe_midi = 0
        crepe_conf = 0.0
        
        if idx_end > idx_start:
            seg_pitch = pitch[idx_start:idx_end]
            seg_conf = conf[idx_start:idx_end]
            
            # Weighted average for Crepe Pitch based on confidence
            valid = seg_conf > 0.2
            if np.any(valid):
                hz = np.average(seg_pitch[valid], weights=seg_conf[valid])
                crepe_midi = librosa.hz_to_midi(hz)
                crepe_conf = float(np.median(seg_conf[valid]))

        # 2. pYIN Data (The Second Opinion)
        pyin_midi = 0
        pyin_conf = 0.0 
        
        # Optimization: Only run pYIN if Crepe isn't 100% sure
        if crepe_conf < 0.85:
            sample_start = int(t_start * sr)
            sample_end = int(t_end * sr)
            
            # Ensure slice is long enough to be meaningful
            if sample_end - sample_start > 512:
                y_slice = y[sample_start:sample_end]
                
                # FIX: Increased frame_length to 2048 to satisfy fmin=60Hz requirement
                f0, _, voiced_prob = librosa.pyin(
                    y_slice, 
                    fmin=fmin, 
                    fmax=fmax, 
                    sr=sr, 
                    frame_length=2048 
                )
                
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    pyin_midi = librosa.hz_to_midi(np.median(valid_f0))
                    pyin_conf = np.count_nonzero(~np.isnan(f0)) / len(f0)

        notes.append({
            "time": t_start,
            "dur": dur if use_holds else 0.0,
            "word": w['word'],
            "source": "vocals",
            "score": float(w['probability']),
            
            # VOTING DATA
            "crepe_midi": float(crepe_midi),
            "crepe_conf": float(crepe_conf),
            "pyin_midi": float(pyin_midi),
            "pyin_conf": float(pyin_conf)
        })
    
    return notes

def separate_audio(audio_path: str, mode: str = "high"):
    # Lazy import to keep top-level clean
    from audio_separator.separator import Separator

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("stems", base_name)
    model_dir = os.path.join(os.getcwd(), "models")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Processing: {base_name} [{mode.upper()}] ---")
    sep = Separator(output_dir=output_dir, model_file_dir=model_dir, output_format="WAV")

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

def ensure_custom_model_exists(model_dir: str):
    from huggingface_hub import hf_hub_download
    repo_id = "jarredou/BS-ROFO-SW-Fixed"
    registry_name = "model_bs_roformer_ep_317_sdr_12.9755"
    os.makedirs(model_dir, exist_ok=True)
    files = {"BS-Rofo-SW-Fixed.ckpt": f"{registry_name}.ckpt", "BS-Rofo-SW-Fixed.yaml": f"{registry_name}.yaml"}
    
    print(f"[INIT] Syncing SOTA 6-Stem model...")
    for remote, local in files.items():
        target = os.path.join(model_dir, local)
        if not os.path.exists(target):
            print(f"    > Downloading {remote}...")
            hf_hub_download(repo_id=repo_id, filename=remote, local_dir=model_dir, local_dir_use_symlinks=False)
            os.rename(os.path.join(model_dir, remote), target)

def _separate_high_sota(audio_path, output_dir, sep):
    sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    files = sep.separate(audio_path)
    mapping = {"Vocals": "vocals", "Drums": "drums", "Bass": "bass", "Guitar": "guitar", "Piano": "piano", "Other": "other", "Instrumental": "other"}
    for k, v in mapping.items(): _rename_stem(output_dir, files, k, v)

def _separate_demucs_cascade(audio_path, output_dir, sep):
    sep.load_model(model_filename='htdemucs_ft.yaml')
    files = sep.separate(audio_path)
    for k, v in {"(Vocals)":"vocals", "(Drums)":"drums", "(Bass)":"bass", "(Other)":"other"}.items():
        _rename_stem(output_dir, files, k, v)

def _separate_htdemucs_6s(audio_path, output_dir, sep):
    sep.load_model(model_filename='htdemucs_6s.yaml')
    files = sep.separate(audio_path)
    mapping = {"(Vocals)": "vocals", "(Drums)": "drums", "(Bass)": "bass", "(Guitar)": "guitar", "(Piano)": "piano", "(Other)": "other"}
    for k, v in mapping.items(): _rename_stem(output_dir, files, k, v)

def _rename_stem(output_dir, files, target, new_name):
    found = next((f for f in files if target.lower() in f.lower()), None)
    if found:
        old = os.path.join(output_dir, found)
        new = os.path.join(output_dir, f"{new_name}.wav")
        if os.path.exists(new): os.remove(new)
        os.rename(old, new)

def _cleanup_intermediates(output_dir):
    print("\n[CLEANUP] Finalizing directory...")
    keeps = ["vocals.wav", "drums.wav", "bass.wav", "guitar.wav", "piano.wav", "other.wav"]
    for f in os.listdir(output_dir):
        if f.endswith(".json") or f in keeps: continue
        try: os.remove(os.path.join(output_dir, f))
        except: pass

def finalize_separation(output_dir: str):
    print("\n--- Final Integrity Check ---")
    stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    manifest = {}
    for stem in stems:
        path = os.path.join(output_dir, f"{stem}.wav")
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None)
            energy = np.max(np.abs(y))
            is_silent = energy <= 0.05
            print(f"  > {stem.capitalize():<10}: {'EMPTY (PURE)' if is_silent else 'ACTIVE':<12} (Peak: {energy:.4f})")
            if is_silent: sf.write(path, np.zeros(sr), sr)
            manifest[stem] = {"exists": True, "is_silent": is_silent, "peak_energy": float(energy)}
        else:
            manifest[stem] = {"exists": False, "is_silent": True, "peak_energy": 0.0}
    
    with open(os.path.join(output_dir, "stems_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--mode", choices=["high", "medium", "low"], default="high")
    args = parser.parse_args()
    separate_audio(args.file, mode=args.mode)