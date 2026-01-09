
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

def test_multipass():
    # Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, "TestData", "kiritan_singing")
    
    wav_file = os.path.join(test_data_dir, "wav", "01.wav")
    midi_file = os.path.join(test_data_dir, "midi_label", "01.mid")
    
    bench = ChoirBenchmark()
    truth = bench.verify_single_track(wav_file, midi_file)
    print(f"Ground Truth: {len(truth)} notes")

    bp = BasicPitchTranscriber()
    
    # CALIBRATION
    print("\n--- Calibration ---")
    # Run a quick pass to find offset
    notes_calib = bp.transcribe(wav_file, "vocals", onset_threshold=0.35, frame_threshold=0.30)
    best_calib = bench.auto_align_and_evaluate(notes_calib, truth)
    best_offset = best_calib.get("shift_time", 0.0)
    # Note: auto_align_and_evaluate in benchmark_choir doesn't return the shift config directly in valid format sometimes?
    # Let's check benchmark_choir.py... ah, it prints it but returns 'best_res' which might not have 'shift_time'.
    # In the benchmark script we modified (Step 472), we manually searched.
    # Let's do the manual search here too.
    
    offset_search = np.linspace(-0.3, 0.3, 13)
    best_offset = 0.0
    best_f1_calib = -1
    for t_off in offset_search:
        test_notes = [NoteEvent(n.time + t_off, n.duration, n.pitch, n.velocity, n.source) for n in notes_calib]
        res = bench.evaluate(test_notes, truth, tolerance_time=0.1)
        if res["f1"] > best_f1_calib:
            best_f1_calib = res["f1"]
            best_offset = t_off
    print(f"Calibrated Offset: {best_offset:.3f}s (F1: {best_f1_calib:.3f})")
    
    # PASS 1: Original
    print("\n--- Pass 1: Original ---")
    # Low thresholds for max recall (calibration: 0.35/0.30)
    notes_p1 = bp.transcribe(wav_file, "vocals", onset_threshold=0.35, frame_threshold=0.30)
    # Apply Calibration Offset
    for n in notes_p1: n.time += best_offset
    
    res_p1 = bench.evaluate(notes_p1, truth)
    print(f"Pass 1 Results: F1={res_p1['f1']:.3f}, Pre={res_p1['precision']:.3f}, Rec={res_p1['recall']:.3f}")
    
    # PASS 2: Transposed (+12st)
    print("\n--- Pass 2: Transposed (+12st) ---")
    temp_wav = "benchmarks/temp/temp_shifted_plus12.wav"
    os.makedirs(os.path.dirname(temp_wav), exist_ok=True)
    
    # Load and Shift
    y, sr = librosa.load(wav_file, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=12)
    sf.write(temp_wav, y_shifted, sr)
    
    notes_p2_raw = bp.transcribe(temp_wav, "vocals", onset_threshold=0.35, frame_threshold=0.30)
    
    # Shift Back (-12st)
    notes_p2 = []
    for n in notes_p2_raw:
        n.pitch -= 12
        n.time += best_offset # Apply calibrated offset
        notes_p2.append(n)
        
    res_p2 = bench.evaluate(notes_p2, truth)
    print(f"Pass 2 Results: F1={res_p2['f1']:.3f}, Pre={res_p2['precision']:.3f}, Rec={res_p2['recall']:.3f}")
    
    # Consensus
    print("\n--- Consensus (Intersection) ---")
    consensus_notes = []
    
    # Optimistic Joining
    # If a note exists in P1, check if it exists in P2 (within tolerance)
    # If yes, keep it.
    
    # Sort
    notes_p1.sort(key=lambda x: x.time)
    notes_p2.sort(key=lambda x: x.time)
    
    matches = 0
    
    for n1 in notes_p1:
        found_match = False
        for n2 in notes_p2:
            if abs(n1.time - n2.time) < 0.1: # 100ms tolerance
                if abs(n1.pitch - n2.pitch) < 0.5: # Same pitch
                    found_match = True
                    break
        
        if found_match:
            consensus_notes.append(n1)
            matches += 1
            
    print(f"Consensus matched {matches} notes (Discarded {len(notes_p1) - matches})")
    
    res_con = bench.evaluate(consensus_notes, truth)
    print(f"Consensus Results: F1={res_con['f1']:.3f}, Pre={res_con['precision']:.3f}, Rec={res_con['recall']:.3f}")
    
    # Cleanup
    if os.path.exists(temp_wav): os.remove(temp_wav)

if __name__ == "__main__":
    test_multipass()
