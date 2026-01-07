import os
import sys
import numpy as np
import soundfile as sf
import argparse
from typing import List, Tuple

# Add parent directory to path to import engine modules
# We want the directory containing 'transcribe', which is the parent of 'debug_tools'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from transcribe.basic_pitch import BasicPitchTranscriber
    from beatmap import NoteEvent
except ImportError as e:
    print(f"Error importing rhythm_engine modules: {e}")
    print("Please run this script from the debug_tools directory or ensure rhythm_engine is in python path.")
    sys.exit(1)

def generate_click_track(num_clicks=1000, bpm=120.0, sr=22050, randomize=False) -> Tuple[np.ndarray, int, List[float]]:
    """Generates a click track with perfect timing.
       If randomize=True, intervals vary between 0.2s (300 BPM) and 0.8s (75 BPM).
    """
    import random
    
    # Estimate duration: average interval 0.5s * num_clicks
    avg_dur = 0.5 if randomize else (60.0 / bpm)
    duration_sec = (num_clicks * avg_dur) + 5.0 # Generous buffer
    
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = np.zeros_like(t)
    
    clicks = []
    
    # Create a 1kHz sine blip for 100ms with fast attack
    blip_dur = 0.10
    blip_t = np.linspace(0, blip_dur, int(blip_dur * sr), endpoint=False)
    blip = np.sin(2 * np.pi * 1000 * blip_t) * 0.8
    # Envelop
    blip[:50] *= np.linspace(0, 1, 50) # Fast Attack
    blip[-100:] *= np.linspace(1, 0, 100) # Release
    
    current_time = 0.5 # Start with 0.5s silence
    
    for _ in range(num_clicks):
        if current_time + blip_dur >= duration_sec:
           break
           
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(blip)
        
        if end_sample < len(audio):
            audio[start_sample:end_sample] += blip
            clicks.append(current_time)
        
        if randomize:
            # Random interval between 0.2s (300 BPM) and 0.8s (75 BPM)
            interval = random.uniform(0.2, 0.8)
        else:
            interval = 60.0 / bpm
            
        current_time += interval
        
    return audio, sr, clicks
    
    # Create a 1kHz sine blip for 100ms with fast attack
    blip_dur = 0.10
    blip_t = np.linspace(0, blip_dur, int(blip_dur * sr), endpoint=False)
    blip = np.sin(2 * np.pi * 1000 * blip_t) * 0.8
    # Envelop
    blip[:50] *= np.linspace(0, 1, 50) # Fast Attack
    blip[-100:] *= np.linspace(1, 0, 100) # Release
    
    current_time = 0.5 # Start with 0.5s silence
    while current_time < duration_sec - 1.0:
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(blip)
        if end_sample < len(audio):
            audio[start_sample:end_sample] += blip
            clicks.append(current_time)
        current_time += beat_interval
        
    return audio, sr, clicks

def benchmark_basic_pitch(audio_path, ground_truth_clicks):
    print(f"[Benchmark] Testing BasicPitch on {os.path.basename(audio_path)}...")
    try:
        transcriber = BasicPitchTranscriber()
    except Exception as e:
        print(f"Failed to init BasicPitch: {e}")
        return

    # Use standard params
    notes = transcriber.transcribe(audio_path, "benchmark_track", onset_threshold=0.5, frame_threshold=0.3)
    
    if not notes:
        print("[Benchmark] No notes detected at all.")
        return

    detected_times = sorted([n.time for n in notes])
    
    latencies = []
    
    print(f"[Benchmark] Processing {len(ground_truth_clicks)} clicks vs {len(detected_times)} detected notes...")
    
    for click_t in ground_truth_clicks:
        # Find nearest detected note
        # Window: +/- 150ms
        candidates = [t for t in detected_times if abs(t - click_t) < 0.15]
        if candidates:
            # Pick the closest one
            best = min(candidates, key=lambda x: abs(x - click_t))
            latency = best - click_t
            latencies.append(latency)
        else:
            # print(f"  Missed click at {click_t:.3f}s")
            pass
            
    if not latencies:
        print("[Benchmark] No matching notes found!")
        return
        
    latencies = np.array(latencies)
    mean_lat_ms = np.mean(latencies) * 1000
    std_lat_ms = np.std(latencies) * 1000
    min_lat_ms = np.min(latencies) * 1000
    max_lat_ms = np.max(latencies) * 1000
    
    print("-" * 60)
    print(f"RESULTS: Basic Pitch ({len(latencies)}/{len(ground_truth_clicks)} matched)")
    print(f"Mean Latency: {mean_lat_ms:+.2f} ms  (Positive = Late, Negative = Early)")
    print(f"Jitter (Std): {std_lat_ms:.2f} ms")
    print(f"Range: [{min_lat_ms:+.2f}, {max_lat_ms:+.2f}] ms")
    print("-" * 60)
    
    return mean_lat_ms, std_lat_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_click.wav", help="Output audio file")
    parser.add_argument("--count", type=int, default=1000, help="Number of clicks to generate")
    args = parser.parse_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, args.output)

    # 1. Generate
    # Randomize intervals to simulate real complex rhythms
    audio, sr, clicks = generate_click_track(num_clicks=args.count, randomize=True)
    sf.write(out_path, audio, sr)
    print(f"[Benchmark] Generated randomized click track at {out_path} ({len(clicks)} clicks)")
    
    # 2. Analyze
    benchmark_basic_pitch(out_path, clicks)
