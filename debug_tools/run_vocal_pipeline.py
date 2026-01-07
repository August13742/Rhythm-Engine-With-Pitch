import argparse
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.signal
from torchfcpe import spawn_bundled_infer_model

try:
    from synthbank import SynthBank
except ImportError:
    print("[WARN] synthbank.py not found. Using Fallback Sine Synth.")
    class SynthBank:
        @staticmethod
        def gen_vocal_tone(midi, duration=0.15):
            sr = 44100
            t = np.linspace(0, duration, int(sr * duration), False)
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            env = np.exp(-5 * t)
            wave = 0.5 * np.sin(2 * np.pi * freq * t) * env
            return (wave * 32767).astype(np.int16)

def f0_to_midi_robust(f0, audio, hop_length, sr, min_dur_sec=0.06):
    """
    V3: Adds Hysteresis to solve 'Interwoven Noise' / Vibrato Jitter.
    """
    frames = len(f0)
    
    # 1. ENERGY GATE (Volume Check)
    # Using 0.12 (Stronger gate based on your feedback)
    frame_len = 2048
    rms_hop = int(len(audio) / frames)
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=rms_hop, center=True)[0]
    rms = librosa.util.fix_length(rms, size=frames)
    if rms.max() > 0: rms /= rms.max()
    
    # Kill anything below 12% volume (removes background hiss)
    f0[rms < 0.12] = 0

    # 2. CONVERT TO MIDI FLOAT
    # Filter realistic range 45 (A1) to 90 (F#6) - Standard Pop Range
    f0_clean = np.where((f0 > 45) & (f0 < 1200), f0, np.nan)
    midi_float = 12 * np.log2(f0_clean / 440.0) + 69
    midi_float = np.nan_to_num(midi_float, nan=0.0)

    # 3. HYSTERESIS QUANTIZATION (The Vibrato Fix)
    # Instead of round(), we use a sticky logic.
    midi_stable = np.zeros_like(midi_float, dtype=int)
    last_pitch = 0
    
    for i in range(frames):
        val = midi_float[i]
        if val <= 0:
            last_pitch = 0
            midi_stable[i] = 0
            continue
            
        if last_pitch == 0:
            # New note start: just round it
            curr = int(round(val))
        else:
            # Sustain: Only change if we drift > 0.6 semitones away
            # This allows vibrato (±0.4) to stay on the same note
            diff = abs(val - last_pitch)
            if diff > 0.6:
                curr = int(round(val))
            else:
                curr = last_pitch
        
        midi_stable[i] = curr
        last_pitch = curr

    # 4. GLITCH MERGING (Post-Processing)
    # If we have pattern AAAAA BBB AAAAA where B is < 60ms, turn B into A.
    min_frames = int(min_dur_sec * sr / hop_length)
    
    # First pass: Build segments
    segments = []
    if len(midi_stable) > 0:
        curr_p = midi_stable[0]
        curr_start = 0
        for i in range(1, frames):
            if midi_stable[i] != curr_p:
                segments.append({"p": curr_p, "s": curr_start, "e": i})
                curr_p = midi_stable[i]
                curr_start = i
        segments.append({"p": curr_p, "s": curr_start, "e": frames})

    # Filter Segments
    final_notes = []
    for i, seg in enumerate(segments):
        dur = (seg["e"] - seg["s"]) * hop_length / sr
        
        # If it's a valid note (pitch > 0) and long enough
        if seg["p"] > 0 and dur >= min_dur_sec:
            final_notes.append({
                "midi": seg["p"],
                "start": seg["s"] * hop_length / sr,
                "dur": dur
            })
        
        # Heuristic: If it's a SHORT note (noise) sandwiched between identical notes?
        # (Optional complexity omitted for stability, simple duration gate is usually enough with hysteresis)

    return final_notes

def run_pipeline(input_path, output_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- FCPE V3: Anti-Vibrato Pipeline ({device.upper()}) ---")
    
    sr_model = 16000
    print(f"[1/4] Loading: {input_path}")
    audio, _ = librosa.load(input_path, sr=sr_model)
    audio_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0).unsqueeze(-1)
    
    print(f"[2/4] Extracting F0...")
    model = spawn_bundled_infer_model(device=device)
    
    # TWEAKED PARAMS
    # Threshold 0.04 (Stricter)
    # f0_max 880 (A5) - safer for pop
    f0_tensor = model.infer(
        audio_tensor, 
        sr=sr_model, 
        decoder_mode="local_argmax", 
        threshold=0.1, 
        f0_min=55, 
        f0_max=1000
    )
    
    f0 = f0_tensor.squeeze().cpu().numpy()
    hop_length = len(audio) / len(f0)
    
    print(f"[3/4] Hysteresis Quantization...")
    notes = f0_to_midi_robust(f0, audio, hop_length, sr_model)
    print(f"  > Extracted {len(notes)} Stable Notes")

    print(f"[4/4] Synthesizing...")
    final_sr = 44100
    if not notes:
        print("[WARN] No notes detected!")
        return

    total_dur = notes[-1]["start"] + notes[-1]["dur"] + 2.0
    out_len = int(total_dur * final_sr)
    output_buffer = np.zeros((out_len, 2), dtype=np.float32)
    
    for n in notes:
        midi = int(n["midi"])
        dur = n["dur"]
        start_t = n["start"]
        
        raw_audio = SynthBank.gen_vocal_tone(midi, duration=dur)
        
        wave_float = raw_audio.astype(np.float32) / 32768.0
        if wave_float.ndim == 1:
            wave_float = np.column_stack((wave_float, wave_float))
            
        start_idx = int(start_t * final_sr)
        end_idx = start_idx + len(wave_float)
        
        if end_idx > out_len:
            wave_float = wave_float[:out_len - start_idx]
            end_idx = out_len
            
        output_buffer[start_idx:end_idx] += wave_float

    peak = np.max(np.abs(output_buffer))
    if peak > 0:
        output_buffer = (output_buffer / peak) * 0.95

    sf.write(output_path, output_buffer, final_sr)
    print(f"[DONE] Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Vocal Stem (wav)")
    parser.add_argument("output", help="Output Synth (wav)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output, args.device)