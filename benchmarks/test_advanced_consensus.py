
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from typing import List

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatmap import NoteEvent
from transcribe.basic_pitch import BasicPitchTranscriber
from benchmarks.benchmark_choir import ChoirBenchmark

def test_advanced_consensus():
    # Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, "TestData", "kiritan_singing")
    
    wav_file = os.path.join(test_data_dir, "wav", "01.wav")
    midi_file = os.path.join(test_data_dir, "midi_label", "01.mid")
    
    bench = ChoirBenchmark()
    truth = bench.verify_single_track(wav_file, midi_file)
    print(f"Ground Truth: {len(truth)} notes")

    bp = BasicPitchTranscriber()
    
    # PARAMETER SWEEP
    # We want to test very low thresholds + Consensus Voting
    thresholds = [
        (0.35, 0.30), # Current Best
        (0.25, 0.20), # Aggressive (Noise heavy)
        (0.20, 0.15)  # Very Aggressive
    ]
    
    # Calibration (Re-use previous logic)
    print("\n--- Calibration ---")
    notes_calib = bp.transcribe(wav_file, "vocals", onset_threshold=0.35, frame_threshold=0.30)
    best_calib = bench.auto_align_and_evaluate(notes_calib, truth) # Prints debug
    # Manual search because I keep forgetting to update the class...
    offset_search = np.linspace(-0.3, 0.3, 13)
    best_offset = 0.0
    best_f1_calib = -1
    for t_off in offset_search:
        test_notes = [NoteEvent(n.time + t_off, n.duration, n.pitch, n.velocity, n.source) for n in notes_calib]
        res = bench.evaluate(test_notes, truth, tolerance_time=0.1)
        if res["f1"] > best_f1_calib:
            best_f1_calib = res["f1"]
            best_offset = t_off
    print(f"Calibrated Offset: {best_offset:.3f}s")

    for onset, frame in thresholds:
        print(f"\n\n=== Testing Thresholds: Onset={onset}, Frame={frame} ===")
        
        # --- PREPARE PASSES ---
        passes = []
        shifts = [0, 12, -12] # Base, High Octave, Low Octave
        
        for semitones in shifts:
            print(f"  Running Shift {semitones}st...")
            if semitones == 0:
                audio_path = wav_file
                cleanup = False
            else:
                y, sr = librosa.load(wav_file, sr=None)
                y_s = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
                audio_path = f"benchmarks/temp/test_shift_{semitones}.wav"
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                sf.write(audio_path, y_s, sr)
                cleanup = True
                
            # Transcribe
            notes_raw = bp.transcribe(audio_path, "vocals", onset_threshold=onset, frame_threshold=frame)
            
            # Normalize
            for n in notes_raw:
                n.pitch -= semitones # Shift back
                n.time += best_offset # Apply calibration
            
            passes.append(notes_raw)
            
            if cleanup: os.remove(audio_path)
            
        # --- VOTING LOGIC ---
        # 2/3 Consensus
        # A note is kept if it has a Match in at least 1 OTHER pass.
        # (i.e. It exists in 2 out of 3 lists)
        
        # Flatten all notes to find candidates? 
        # Or iterate through Pass 1 (Base) and confirm with others?
        # Iterating Base is biased towards Base.
        # Ideally we pool all notes and cluster them.
        
        print("  Computing Consensus (2/3 Votes)...")
        
        all_notes = []
        for i, p_list in enumerate(passes):
            for n in p_list:
                n.source = f"pass_{i}" # Tag source
                all_notes.append(n)
                
        # Simple Clustering
        # Sort by time
        all_notes.sort(key=lambda x: x.time)
        
        clusters = []
        used = [False] * len(all_notes)
        
        for i in range(len(all_notes)):
            if used[i]: continue
            
            # Start cluster
            cluster = [all_notes[i]]
            used[i] = True
            
            # Find neighbors
            for j in range(i+1, len(all_notes)):
                if used[j]: continue
                
                n1 = all_notes[i]
                n2 = all_notes[j]
                
                if abs(n2.time - n1.time) > 0.1: # Outside time window
                    break
                    
                if abs(n2.pitch - n1.pitch) < 0.5: # Same pitch
                    cluster.append(n2)
                    used[j] = True
            
            clusters.append(cluster)
            
        # Vote
        final_notes = []
        min_votes = 2
        
        for c in clusters:
            # Check unique sources in cluster
            sources = set(n.source for n in c)
            if len(sources) >= min_votes:
                # Average the candidate? Or just take the first one?
                # Taking the one from Pass 0 (Base) is safest if available, else average.
                
                proto = c[0]
                # Try find Pass 0
                base_note = next((n for n in c if n.source == "pass_0"), None)
                if base_note:
                    proto = base_note
                
                final_notes.append(proto)
                
        # Evaluate
        print(f"  Consensus Idea: {len(final_notes)} notes.")
        res = bench.evaluate(final_notes, truth)
        print(f"  RESULTS -> F1: {res['f1']:.3f}, Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}")
        
        # Compare with Single Pass (Pass 0)
        res_p0 = bench.evaluate(passes[0], truth)
        print(f"  BASELINE (Single Pass) -> F1: {res_p0['f1']:.3f}, Precision: {res_p0['precision']:.3f}, Recall: {res_p0['recall']:.3f}")

if __name__ == "__main__":
    test_advanced_consensus()
