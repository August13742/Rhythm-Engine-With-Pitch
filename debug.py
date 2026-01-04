import sys
import os
import gc
import time
import numpy as np
import json
import torch
import librosa
import soundfile as sf
import traceback

# Force unbuffered output so we see the log exactly when it happens
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def safe_transcription_test(audio_path):
    log(f"--- START DEBUGGING: {audio_path} ---")
    
    if not os.path.exists(audio_path):
        log("ERROR: File not found.")
        return

    # --- STEP 1: LIBROSA LOAD ---
    log("STEP 1: Loading Audio (Librosa)...")
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        log(f"  > Loaded audio: {y.shape} samples, SR={sr}")
        del y # Free this copy
        gc.collect()
    except Exception:
        log("FAIL: Librosa load crashed.")
        traceback.print_exc()
        return

    # --- STEP 2: WHISPER ---
    log("STEP 2: Initializing Whisper...")
    try:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  > Device: {device}")
        
        model = WhisperModel("large-v3", device=device, compute_type="float16")
        log("  > Model loaded. Starting transcription...")
        
        segments, _ = model.transcribe(
            audio_path, 
            word_timestamps=True, 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500),
            language="ja"
        )
        
        # Aggressive extraction loop with logs
        anchors = []
        count = 0
        log("  > Iterating segments...")
        
        for s in segments:
            for w in s.words:
                # Test accessing C++ properties explicitly
                try:
                    wd = {
                        "word": str(w.word),
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": float(w.probability)
                    }
                    if wd["probability"] > 0.25:
                        anchors.append(wd)
                    count += 1
                except Exception as e:
                    log(f"CRASH WARNING: Failed to read word object: {e}")

        log(f"  > Whisper complete. Processed {count} words. Kept {len(anchors)} anchors.")
        
        # EXPLICIT CLEANUP
        log("  > Cleaning up Whisper model...")
        del model
        del segments
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("  > Whisper memory freed.")
        
    except Exception:
        log("FAIL: Whisper step crashed.")
        traceback.print_exc()
        return

    # --- STEP 3: CREPE ---
    log("STEP 3: Initializing CREPE...")
    pitch_data = None
    conf_data = None
    times_data = None
    
    try:
        import torchcrepe
        
        # Reload audio for Crepe
        y, sr = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.tensor(y).unsqueeze(0).to(device)
        hop_len = int(16000 / 100)
        
        log("  > Running Predict...")
        pitch, periodicity = torchcrepe.predict(
            audio_tensor, 16000, hop_len, 
            fmin=50, fmax=1000, model='full', 
            batch_size=2048, device=device, return_periodicity=True
        )
        
        log("  > Moving to CPU...")
        pitch_data = pitch.squeeze().cpu().numpy()
        conf_data = periodicity.squeeze().cpu().numpy()
        times_data = librosa.times_like(pitch_data, sr=16000, hop_length=hop_len)
        
        log(f"  > Crepe done. Pitch shape: {pitch_data.shape}")
        
        del audio_tensor
        del pitch
        del periodicity
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception:
        log("FAIL: Crepe step crashed.")
        traceback.print_exc()
        return

    # --- STEP 4: NOTE CONSTRUCTION (The Suspect) ---
    log("STEP 4: Entering Note Construction Logic...")
    
    try:
        notes = []
        log(f"  > Starting loop over {len(anchors)} anchors...")
        
        for i, w in enumerate(anchors):
            # Log every 50 words to see where it dies
            if i % 50 == 0:
                log(f"    > Processing anchor {i}/{len(anchors)}")
                
            t_start = w['start']
            t_end = w['end']
            
            # TEST: Search Sorted
            idx_start = np.searchsorted(times_data, t_start)
            idx_end = np.searchsorted(times_data, t_end)
            
            if idx_end <= idx_start:
                continue
                
            # TEST: Slicing
            seg_pitch = pitch_data[idx_start:idx_end]
            seg_conf = conf_data[idx_start:idx_end]
            
            # TEST: Math
            valid_mask = seg_conf > 0.4
            if np.any(valid_mask):
                hz = np.median(seg_pitch[valid_mask])
                midi = librosa.hz_to_midi(hz)
            else:
                hz = np.median(seg_pitch)
                midi = 60 # dummy
            
            # Construct dict
            note = {
                "time": t_start,
                "word": w['word'],
                "midi": midi
            }
            notes.append(note)
            
        log(f"  > Loop finished. Generated {len(notes)} notes.")
        
    except Exception:
        log("FAIL: Note construction logic crashed.")
        traceback.print_exc()
        return

    log("--- SUCCESS: Pipeline finished without hard crash. ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pipeline.py <audio_file>")
    else:
        safe_transcription_test(sys.argv[1])