import librosa
import numpy as np
import sys
import os

def check_energy(path, target_time):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    y, sr = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Loaded {path}")
    print(f"Duration: {duration:.2f}s")
    
    # Check around target time (window of 0.5s)
    start_sample = int((target_time - 0.25) * sr)
    end_sample = int((target_time + 0.25) * sr)
    
    if start_sample >= len(y):
        print(f"Target time {target_time}s is beyond audio duration.")
        return

    end_sample = min(end_sample, len(y))
    segment = y[start_sample:end_sample]
    
    rms = np.sqrt(np.mean(segment**2))
    peak = np.max(np.abs(segment))
    
    print(f"Analysis at {target_time}s +/- 0.25s:")
    print(f"  RMS Energy: {rms:.6f}")
    print(f"  Peak Amp:   {peak:.6f}")
    
    # Also check tail average (last 5 seconds)
    tail_start = int(max(0, duration - 5) * sr)
    tail_segment = y[tail_start:]
    tail_rms = np.sqrt(np.mean(tail_segment**2))
    print(f"Tail (last 5s) RMS: {tail_rms:.6f}")

if __name__ == "__main__":
    stem_path = "stems/betelgeuse/guitar.wav"
    check_energy(stem_path, 236.25)
