"""
transcribe_worker.py
Independent worker process for Whisper + CREPE.
Usage: python transcribe_worker.py <input_audio_path> <output_json_path>
"""
import sys
import os
import json
import numpy as np
import torch
import librosa
import traceback

# 1. Setup Encoders
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            val = float(obj)
            return val if np.isfinite(val) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_pipeline(audio_path, output_json):
    print(f"[WORKER] Processing: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WORKER] Device: {device}")

    # --- STEP A: LOAD AUDIO ---
    # We load once for Whisper/Librosa, convert for Crepe
    y, sr = librosa.load(audio_path, sr=16000)
    
    # --- STEP B: WHISPER ---
    print("[WORKER] Running Whisper...")
    from faster_whisper import WhisperModel
    
    # Load model
    model = WhisperModel("large-v3", device=device, compute_type="float16")
    
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language=None
    )

    anchors = []
    whisper_dump = [] # For debugging visualization

    for s in segments:
        segment_words = []
        for w in s.words:
            # STRICT CASTING: Convert C++ objects to Python primitives immediately
            wd = {
                "word": str(w.word),
                "start": float(w.start),
                "end": float(w.end),
                "probability": float(w.probability)
            }
            segment_words.append(wd)
            
            if wd["probability"] > 0.25:
                anchors.append(wd)

        whisper_dump.append({
            "text": str(s.text),
            "start": float(s.start),
            "end": float(s.end),
            "words": segment_words
        })

    print(f"[WORKER] Whisper found {len(anchors)} anchors.")
    
    # Clean up Whisper to free VRAM for Crepe
    del model
    del segments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- STEP C: CREPE ---
    print("[WORKER] Running CREPE...")
    import torchcrepe
    
    audio_tensor = torch.tensor(y).unsqueeze(0).to(device)
    hop_len = int(16000 / 100) # 10ms hop

    pitch, periodicity = torchcrepe.predict(
        audio_tensor, 16000, hop_len,
        fmin=50, fmax=1000, model='full',
        batch_size=2048, device=device, return_periodicity=True
    )

    pitch_np = pitch.squeeze().cpu().numpy()
    conf_np = periodicity.squeeze().cpu().numpy()
    times_np = librosa.times_like(pitch_np, sr=16000, hop_length=hop_len)

    # --- STEP D: SAVE RESULT ---
    result = {
        "anchors": anchors,
        "pitch": pitch_np,
        "conf": conf_np,
        "times": times_np,
        "whisper_dump": whisper_dump
    }

    print(f"[WORKER] Saving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, cls=NumpyEncoder, ensure_ascii=False)
    
    print("[WORKER] Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_worker.py <input_audio> <output_json>")
        sys.exit(1)
    
    try:
        run_pipeline(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"[WORKER ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)