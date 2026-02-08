import os
import argparse
import numpy as np
import librosa
import torch
import torchcrepe
from faster_whisper import WhisperModel
import json

# ==========================================
#  CONFIG (32GB VRAM FLEX)
# ==========================================
# We use 'large-v3' because you have the hardware. 
# It has the best timestamp alignment precision.
WHISPER_SIZE = "large-v3" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_hybrid_engine(audio_path):
    print(f"--- HYBRID ENGINE DIAGNOSTIC ---")
    print(f"Target: {os.path.basename(audio_path)}")
    print(f"Device: {DEVICE.upper()} | Model: {WHISPER_SIZE}")

    # 1. LOAD AUDIO
    # ---------------------------------------------------
    print("\n[1/3] Loading Audio (Resampling to 16k for ML)...")
    # Whisper and Crepe both prefer 16kHz
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    print(f"    > Duration: {duration:.2f}s")

    # 2. RUN WHISPER (THE RHYTHM ANCHOR)
    # ---------------------------------------------------
    print(f"\n[2/3] Running Whisper ({WHISPER_SIZE})...")
    # compute_type="float16" is standard for GPUs
    model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type="float16")

    # word_timestamps=True is the magic switch
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    word_anchors = []
    print("    > Stream output:")
    for segment in segments:
        for word in segment.words:
            word_anchors.append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "prob": word.probability
            })
            # Print words as they come to verify alignment
            print(f"      [{word.start:.2f}s - {word.end:.2f}s] '{word.word.strip()}'")

    print(f"    > Captured {len(word_anchors)} rhythmic anchors.")

    # 3. RUN CREPE (THE MELODY FILLER)
    # ---------------------------------------------------
    print(f"\n[3/3] Running TorchCREPE (Full Model)...")
    # Prepare tensor
    audio_tensor = torch.tensor(y).unsqueeze(0).to(DEVICE)
    
    # 10ms hop length (standard for rhythm games)
    hop_length = int(sr / 100) 
    
    # Run prediction
    # We use a wider pitch range to catch deep rap voices
    fmin, fmax = 50, 800 
    
    batch_size = 2048 # You have 32GB VRAM, crank it up
    pitch, periodicity = torchcrepe.predict(
        audio_tensor, sr, hop_length, 
        fmin, fmax, 
        model='full', 
        batch_size=batch_size, 
        device=DEVICE, 
        return_periodicity=True
    )
    
    # Move to CPU for processing
    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()
    times = librosa.times_like(pitch, sr=sr, hop_length=hop_length)

    # 4. FUSION CHECK
    # ---------------------------------------------------
    print("\n[4/4] Fusing Data (Rap/Speech Detection)...")
    
    notes = []
    speech_singing_count = 0
    tonal_count = 0
    
    last_known_pitch = 60 # Default MIDI middle C
    
    for anchor in word_anchors:
        t_start = anchor['start']
        t_end = anchor['end']
        
        # Slice the pitch data for this word
        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)
        
        if idx_end <= idx_start: continue
        
        chunk_conf = periodicity[idx_start:idx_end]
        chunk_pitch = pitch[idx_start:idx_end]
        
        # DECISION LOGIC:
        # If the average confidence in this word is low (< 0.4), it's "Speech/Rap".
        # If it's high, it's "Singing".
        
        avg_conf = np.mean(chunk_conf) if len(chunk_conf) > 0 else 0
        
        if avg_conf > 0.4:
            # Tonal: Use median pitch
            hz = np.median(chunk_pitch)
            midi = librosa.hz_to_midi(hz)
            mode = "SUNG"
            last_known_pitch = midi # Update anchor
            tonal_count += 1
        else:
            # Atonal: Use last known pitch (Rap Mode)
            midi = last_known_pitch
            mode = "RAP " # Space for alignment
            speech_singing_count += 1
            
        notes.append({
            "time": t_start,
            "word": anchor['word'],
            "midi": int(round(midi)),
            "conf": float(avg_conf),
            "mode": mode
        })

    # 5. REPORT
    # ---------------------------------------------------
    print("-" * 60)
    print(f" RESULTS FOR: {os.path.basename(audio_path)}")
    print("-" * 60)
    print(f" Total Notes: {len(notes)}")
    print(f" Sung Notes:  {tonal_count} (High Confidence)")
    print(f" Rap/Speech:  {speech_singing_count} (Rescued by Whisper)")
    print("-" * 60)
    
    # Show first 10 and last 10 notes
    print(" Sample Output:")
    for n in notes[:5]:
        print(f" {n['time']:.3f}s | {n['mode']} | MIDI {n['midi']} | Conf {n['conf']:.2f} | '{n['word']}'")
    print(" ...")
    for n in notes[-5:]:
        print(f" {n['time']:.3f}s | {n['mode']} | MIDI {n['midi']} | Conf {n['conf']:.2f} | '{n['word']}'")

    # Save small dump
    with open("hybrid_test_output.json", "w") as f:
        json.dump(notes, f, indent=2)
    print("\n[DONE] Saved details to hybrid_test_output.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to vocal stem")
    args = parser.parse_args()
    
    test_hybrid_engine(args.file)