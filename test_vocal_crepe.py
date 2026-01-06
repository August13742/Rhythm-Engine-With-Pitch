import argparse
import os
import torch
import librosa
import numpy as np
import soundfile as sf
import scipy.signal
import torchcrepe

# Try importing your synth
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

def f0_to_midi_crepe_robust(f0, confidence, audio, hop_length, sr, min_dur_sec=0.06):
    """
    Adapted V3 Logic for Crepe (Uses Confidence + RMS + Hysteresis).
    """
    frames = len(f0)
    
    # 1. GATING (Confidence + Energy)
    
    # A. Confidence Gate (Crepe Specific)
    # 0.4 is standard for clean vocals. 0.6 is strict.
    conf_mask = confidence < 0.4
    f0[conf_mask] = 0
    
    # B. RMS Energy Gate (Silence Removal)
    frame_len = 2048
    rms_hop = int(len(audio) / frames)
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=rms_hop, center=True)[0]
    rms = librosa.util.fix_length(rms, size=frames)
    if rms.max() > 0: rms /= rms.max()
    
    # Kill silence (even if Crepe is 'confident' about background noise)
    f0[rms < 0.12] = 0

    # 2. CONVERT TO MIDI FLOAT
    # Crepe is smoother than FCPE, but we still clamp range
    f0_clean = np.where((f0 > 45) & (f0 < 1200), f0, np.nan)
    midi_float = 12 * np.log2(f0_clean / 440.0) + 69
    midi_float = np.nan_to_num(midi_float, nan=0.0)

    # 3. HYSTERESIS QUANTIZATION (Sticky Pitch)
    midi_stable = np.zeros_like(midi_float, dtype=int)
    last_pitch = 0
    
    for i in range(frames):
        val = midi_float[i]
        if val <= 0:
            last_pitch = 0
            midi_stable[i] = 0
            continue
            
        if last_pitch == 0:
            curr = int(round(val))
        else:
            # Vibrato tolerance (0.6 semitones)
            diff = abs(val - last_pitch)
            if diff > 0.6:
                curr = int(round(val))
            else:
                curr = last_pitch
        
        midi_stable[i] = curr
        last_pitch = curr

    # 4. GLITCH MERGING
    min_frames = int(min_dur_sec * sr / hop_length)
    
    # Build segments
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
        
        if seg["p"] > 0 and dur >= min_dur_sec:
            final_notes.append({
                "midi": seg["p"],
                "start": seg["s"] * hop_length / sr,
                "dur": dur
            })

    return final_notes

def run_pipeline(input_path, output_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- CREPE Robust Pipeline ({device.upper()}) ---")
    
    # 1. SETUP
    # Crepe standard is 16k
    sr_model = 16000
    hop_length = 160 # 10ms at 16k
    
    print(f"[1/4] Loading: {input_path}")
    audio, _ = librosa.load(input_path, sr=sr_model)
    
    # Crepe expects (1, samples)
    audio_tensor = torch.tensor(audio, device=device).unsqueeze(0)
    
    # 2. CREPE INFERENCE
    print(f"[2/4] Running TorchCrepe (Full Model)...")
    # Batch size 2048 is safe for 5090
    f0, confidence = torchcrepe.predict(
        audio_tensor, 
        sr_model, 
        hop_length=hop_length, 
        fmin=50, 
        fmax=880, 
        model='full', 
        batch_size=2048, 
        device=device,
        return_periodicity=True
    )
    
    # Extract from tensor
    f0 = f0.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()
    
    print(f"[3/4] Hysteresis & Gating...")
    # Pass 'audio' (numpy) for RMS calculation
    notes = f0_to_midi_crepe_robust(f0, confidence, audio, hop_length, sr_model)
    print(f"  > Extracted {len(notes)} Notes")

    # 4. SYNTHESIS
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