"""
SEPARATION ENGINE V2.2
Updates:
  - Stage 1: BS-Roformer (Stems)
  - Stage 2: Mel-Roformer-Viperx (Lead vs Backing)
  - Analysis: Integrated Polyphony "Council" (Spatial + Harmonic)
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
from transcribe.basic_pitch import BasicPitchTranscriber

# Suppress internal logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

class PolyphonyDetector:
    """
    Analyzes vocal stems to detect Polyphony (Choir/Harmony/Double-Tracking).
    Uses a 'Council' of two methods:
    1. Stereo Width (Unison/Double-Track detection)
    2. Harmonic Density (Chords/Harmony detection via BasicPitch)
    
    V2.2: Integrated ClearVoice MossFormer2 for SOTA dereverb quality.
    """
    
    _clearvoice_model = None  # Singleton for model reuse
    
    @staticmethod
    def _init_clearvoice():
        """Initialize ClearVoice model (singleton)."""
        if PolyphonyDetector._clearvoice_model is None:
            import sys
            from pathlib import Path
            clearvoice_path = Path(__file__).parent / "transcribe" / "ClearerVoice-Studio-main" / "clearvoice"
            if str(clearvoice_path) not in sys.path:
                sys.path.insert(0, str(clearvoice_path))
            
            from clearvoice import ClearVoice
            PolyphonyDetector._clearvoice_model = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
        return PolyphonyDetector._clearvoice_model
    
    @staticmethod
    def _apply_compression(audio, sr, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50):
        """Simple compressor to even out volume swings from ClearVoice."""
        audio_abs = np.abs(audio)
        audio_db = 20 * np.log10(audio_abs + 1e-10)
        
        gain_db = np.zeros_like(audio_db)
        over_threshold = audio_db > threshold_db
        gain_db[over_threshold] = (audio_db[over_threshold] - threshold_db) * (1 - 1/ratio)
        
        attack_samples = int(attack_ms * sr / 1000)
        release_samples = int(release_ms * sr / 1000)
        
        gain_smoothed = np.zeros_like(gain_db)
        for i in range(len(gain_db)):
            if i == 0:
                gain_smoothed[i] = gain_db[i]
            else:
                alpha = (1.0 / max(attack_samples, 1)) if gain_db[i] > gain_smoothed[i-1] else (1.0 / max(release_samples, 1))
                gain_smoothed[i] = alpha * gain_db[i] + (1 - alpha) * gain_smoothed[i-1]
        
        gain_linear = 10 ** (-gain_smoothed / 20)
        return audio * gain_linear
    
    @staticmethod
    def _normalize_rms(audio, target_rms=0.1):
        """Normalize to target RMS level."""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            return audio * (target_rms / current_rms)
        return audio
    
    @staticmethod
    def _dereverb_clearvoice(audio_path: str) -> str:
        """
        Apply ClearVoice MossFormer2 dereverb + normalization.
        Returns path to processed temporary file.
        """
        import tempfile
        
        # Initialize model
        clearer = PolyphonyDetector._init_clearvoice()
        
        # Enhance
        output_wav = clearer(input_path=audio_path, online_write=False)
        
        # Save to temp file
        temp_enhanced = tempfile.NamedTemporaryFile(suffix='_clearvoice.wav', delete=False)
        clearer.write(output_wav, output_path=temp_enhanced.name)
        temp_enhanced.close()
        
        # Load and normalize
        y, sr = librosa.load(temp_enhanced.name, sr=None, mono=False)
        
        if y.ndim > 1:
            processed = np.zeros_like(y)
            for ch in range(y.shape[0]):
                compressed = PolyphonyDetector._apply_compression(y[ch], sr)
                processed[ch] = PolyphonyDetector._normalize_rms(compressed)
        else:
            compressed = PolyphonyDetector._apply_compression(y, sr)
            processed = PolyphonyDetector._normalize_rms(compressed)
        
        # Save final normalized version
        temp_final = tempfile.NamedTemporaryFile(suffix='_normalized.wav', delete=False)
        if processed.ndim > 1:
            sf.write(temp_final.name, processed.T, sr)
        else:
            sf.write(temp_final.name, processed, sr)
        temp_final.close()
        
        # Clean up intermediate temp file
        os.remove(temp_enhanced.name)
        
        return temp_final.name
    
    @staticmethod
    def analyze(audio_path: str, bp_transcriber: BasicPitchTranscriber = None, skip_clearvoice: bool = False) -> dict:
        results = {
            "is_polyphonic": False,
            "confidence": 0.0,
            "details": {}
        }
        
        try:
            from pathlib import Path
            clean_path_obj = Path(audio_path)
            raw_path_obj = clean_path_obj.parent / "vocals_lead_raw.wav"
            
            # --- SOURCE SELECTION ---
            # Width needs RAW (to detect panning/reverb width).
            # Pitch needs CLEAN (to avoid hallucinating reverb tails as notes).
            
            if raw_path_obj.exists():
                path_for_width = str(raw_path_obj)
                path_for_pitch = str(clean_path_obj) # Use the already processed file
                print(f"  [PolyDetector] Using RAW for Width, CLEAN for Pitch.")
            else:
                path_for_width = audio_path
                path_for_pitch = audio_path
                # If we don't have a raw file, we might need to clean the input for pitch
                if not skip_clearvoice:
                     # (Logic to run clearvoice temp generation if needed, same as before)
                     pass

            # --- DETECTOR 1: Stereo Width (The "Studio Trick") ---
            # Use path_for_width (Raw if avail)
            y_width, sr_width = librosa.load(path_for_width, sr=None, mono=False)
            
            width_score = 0.0
            is_wide = False
            
            if y_width.ndim > 1:
                L = y_width[0]
                R = y_width[1]
                side = (L - R) / 2.0
                mid = (L + R) / 2.0
                
                rmse_side = np.sqrt(np.mean(side**2))
                rmse_mid = np.sqrt(np.mean(mid**2))
                
                if rmse_mid > 0:
                    width_ratio = rmse_side / rmse_mid
                    width_score = width_ratio
                    if width_ratio > 0.20:
                        is_wide = True
            
            results["details"]["stereo_width"] = width_score
            
            # --- DETECTOR 2: Harmonic Density (The "Chord" Detector) ---
            # Use path_for_pitch (Cleaned)
            # Since Stage 3 already cleaned 'audio_path', we use it directly!
            # We DO NOT need to run _dereverb_clearvoice again if Stage 3 ran.
            
            density_score = 0.0
            is_dense = False
            
            if bp_transcriber:
                try:
                    # Direct transcribe on the Clean path
                    notes = bp_transcriber.transcribe(path_for_pitch, instrument_name="vocals", 
                                                    onset_threshold=0.4, frame_threshold=0.3)
                    
                    if notes:
                        duration = max(n.time + n.duration for n in notes)
                        time_steps = int(duration * 10) 
                        counts = np.zeros(time_steps + 2)
                        
                        for n in notes:
                            start = int(n.time * 10)
                            end = int((n.time + n.duration) * 10)
                            counts[start:end] += 1
                            
                        poly_frames = np.sum(counts >= 2)
                        total_active = np.sum(counts >= 1)
                        
                        if total_active > 0:
                            density_score = poly_frames / total_active
                            if density_score > 0.15:
                                is_dense = True
                except Exception as e:
                    print(f"  [PolyDetector] BasicPitch failed: {e}")

            results["details"]["harmonic_density"] = density_score

            if is_wide or is_dense:
                results["is_polyphonic"] = True
                results["confidence"] = max(width_score, density_score * 2.0)
            
            return results

        except Exception as e:
            print(f"  [PolyDetector] Analysis failed: {e}")
            return results


def ensure_custom_models_exist(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    
    # --- MODEL 1: BS-Roformer (Stems) ---
    rofo_files = {
        "BS-Rofo-SW-Fixed.ckpt": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-Rofo-SW-Fixed.yaml": "model_bs_roformer_ep_317_sdr_12.9755.yaml"
    }
    
    # --- MODEL 2: Mel-Roformer-Viperx (Lead vs Backing) ---
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

def separate_audio(audio_path: str, output_path: str = None):
    base_name = Path(audio_path).stem
    
    if output_path:
        root_dir = Path(output_path)
    else:
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
    if vocal_active:
        print(f"\n[2/2] Mel-Roformer-Viperx (Lead/Backing)...")
        sep_s2 = Separator(output_dir=str(s2_dir), model_file_dir=str(model_dir), output_format="WAV", log_level=logging.ERROR)
        sep_s2.load_model('model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
        sep_s2.separate(str(root_dir / "vocals.wav"))
        
        for f in s2_dir.glob("*.wav"):
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
    
    # ---------------------------------------------------------
    # STAGE 3: ClearVoice Enhancement (vocals_lead only)
    # ---------------------------------------------------------
    vocals_lead_path = root_dir / "vocals_lead.wav"
    vocals_lead_raw_path = root_dir / "vocals_lead_raw.wav"
    
    # Only process if vocals are active (not silent)
    vocals_lead_active = False
    if vocal_active and vocals_lead_path.exists():
        # Check if vocals_lead has significant energy (not silent)
        vocals_lead_active = _check_activity(vocals_lead_path, threshold=0.02)  # Use same threshold as is_silent check
    
    if vocals_lead_active:
        print(f"\n[3/3] ClearVoice Enhancement...")
        try:
            # Backup raw vocals_lead
            shutil.copy2(vocals_lead_path, vocals_lead_raw_path)
            print(f"  [BACKUP] vocals_lead.wav → vocals_lead_raw.wav")
            
            # Initialize ClearVoice (singleton)
            from pathlib import Path as PathLib
            import sys
            clearvoice_path = PathLib(__file__).parent / "transcribe" / "ClearerVoice-Studio-main" / "clearvoice"
            if str(clearvoice_path) not in sys.path:
                sys.path.insert(0, str(clearvoice_path))
            
            from clearvoice import ClearVoice
            clearer = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
            
            # Enhance
            print(f"  [ENHANCE] Applying MossFormer2...")
            output_wav = clearer(input_path=str(vocals_lead_raw_path), online_write=False)
            
            # Save to temp
            temp_enhanced = root_dir / "temp_clearvoice.wav"
            clearer.write(output_wav, output_path=str(temp_enhanced))
            
            # Load and apply compression + normalization
            print(f"  [NORMALIZE] Applying compression and RMS normalization...")
            y, sr = librosa.load(str(temp_enhanced), sr=None, mono=False)
            
            if y.ndim > 1:
                processed = np.zeros_like(y)
                for ch in range(y.shape[0]):
                    compressed = PolyphonyDetector._apply_compression(y[ch], sr)
                    processed[ch] = PolyphonyDetector._normalize_rms(compressed)
                sf.write(str(vocals_lead_path), processed.T, sr)
            else:
                compressed = PolyphonyDetector._apply_compression(y, sr)
                processed = PolyphonyDetector._normalize_rms(compressed)
                sf.write(str(vocals_lead_path), processed, sr)
            
            # Clean up temp
            temp_enhanced.unlink(missing_ok=True)
            
            print(f"  [DONE] vocals_lead.wav enhanced successfully!")
            
        except Exception as e:
            print(f"  [WARN] ClearVoice enhancement failed: {e}")
            print(f"  [FALLBACK] Restoring raw vocals_lead.wav")
            if vocals_lead_raw_path.exists():
                shutil.copy2(vocals_lead_raw_path, vocals_lead_path)
    else:
        print(f"\n[SKIP] Stage 3 (No Active Vocals or Silent)")

    _optimize_stems(root_dir)
    _match_stem_lengths(root_dir)
    # Run Final Analysis (With Polyphony Detection)
    finalize_separation(str(root_dir))
    
    return str(root_dir)


# --- HELPER FUNCTIONS ---

def _check_activity(path: Path, threshold: float = 0.015) -> bool:
    if not path.exists(): return False
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames == 0: return False
        
        # Read the first 60 seconds
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
    print("\n--- Analysis & Manifest Generation ---")
    stems = ["vocals", "vocals_lead", "vocals_backing", "drums", "bass", "guitar", "piano", "other"]
    manifest = {}
    
    # Initialize BasicPitch just once for analysis
    bp = BasicPitchTranscriber()
    
    # Polyphony Analysis Targets
    lead_path = os.path.join(output_dir, "vocals_lead.wav")
    
    # 1. Gather Stem Stats
    for stem in stems:
        path = os.path.join(output_dir, f"{stem}.wav")
        info = {"exists": False, "peak_energy": 0.0}
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=None)
            peak = float(np.max(np.abs(y)))
            info = {"exists": True, "peak_energy": peak, "is_silent": peak < 0.02}
            
            state = "ACTIVE" if not info["is_silent"] else "_"
            print(f"  > {stem:<15} : {state:<6} (Peak: {peak:.2f})")
        manifest[stem] = info

    # 2. Run Polyphony Council on Lead Vocals
    # We analyze Lead because that's where the 'main' polyphony (choir/duet) would be if the separator failed to split it,
    # OR if it's a stylistic choice (double tracking).
    print(f"  > Running Polyphony Council on vocals_lead...")
    poly_result = PolyphonyDetector.analyze(lead_path, bp_transcriber=bp)
    
    manifest["vocal_type"] = "polyphonic" if poly_result["is_polyphonic"] else "monophonic"
    manifest["polyphony_details"] = poly_result["details"]
    
    
    print(f"[DECISION] Vocal Mode: {manifest['vocal_type'].upper()} (Width: {poly_result['details'].get('stereo_width',0):.2f}, Density: {poly_result['details'].get('harmonic_density',0):.2f})")
    
    with open(os.path.join(output_dir, "stems_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    args = parser.parse_args()
    separate_audio(args.file)