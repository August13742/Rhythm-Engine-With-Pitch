
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
import argparse
from typing import List, Dict, Tuple

# Add parent directory to path to allow importing rhythm_engine modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatmap import NoteEvent
from transcribe.council import CouncilV2
from transcribe.basic_pitch import BasicPitchTranscriber

class ChoirBenchmark:
    def __init__(self, temp_dir: str = "benchmarks/temp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def generate_synthetic_data(self, source_wav: str, source_midi: str, interval: int = 4) -> Tuple[str, List[NoteEvent]]:
        """
        Creates a 'Choir' mix: Original + Original shifted by `interval` semitones.
        Returns: (path_to_mixed_wav, list_of_ground_truth_events)
        """
        print(f"[Benchmark] Generating synthetic data from {os.path.basename(source_wav)} (Shift: +{interval}st)...")
        
        # 1. Load Audio (Full processing)
        y, sr = librosa.load(source_wav, sr=None)

        # 2. Shift Audio
        # Use simple pitch shift (time-stretch agnostic? basic pitch shift changes speed if resampling, 
        # but librosa.effects.pitch_shift preserves duration).
        # It's slow but fine for benchmark.
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=interval)
        
        # 3. Mix
        # Normalize to avoid clipping
        y_mix = (y + y_shifted) / 2.0
        
        out_path = os.path.join(self.temp_dir, f"choir_mix_{interval}st.wav")
        sf.write(out_path, y_mix, sr, subtype='PCM_16')
        
        # 4. Process MIDI Ground Truth
        truth_events = []
        
        try:
            pm = pretty_midi.PrettyMIDI(source_midi)
            
            # Combine all tracks? Usually vocal implies specific track, but let's take all notes.
            original_notes = []
            for instrument in pm.instruments:
                # Filter drum tracks?
                if instrument.is_drum: continue
                original_notes.extend(instrument.notes)
                
            # Convert to NoteEvents
            for note in original_notes:
                # Original
                truth_events.append(NoteEvent(
                    time=note.start,
                    duration=note.end - note.start,
                    pitch=note.pitch,
                    velocity=note.velocity / 127.0,
                    source="vocal_main"
                ))
                
                # Shifted
                truth_events.append(NoteEvent(
                    time=note.start,
                    duration=note.end - note.start,
                    pitch=note.pitch + interval,
                    velocity=note.velocity / 127.0,
                    source="vocal_harmony"
                ))
                
        except Exception as e:
            print(f"[Benchmark] MIDI Error: {e}")
            return out_path, []
            
        print(f"[Benchmark] Created mix with {len(truth_events)} ground truth notes.")
        return out_path, truth_events

    def evaluate(self, predicted: List[NoteEvent], ground_truth: List[NoteEvent], tolerance_time: float = 0.1, tolerance_pitch: float = 0.5) -> Dict:
        """
        Compares prediction against ground truth.
        """
        # Sort both
        predicted.sort(key=lambda x: x.time)
        ground_truth.sort(key=lambda x: x.time)
        
        tp = 0
        fp = 0
        fn = 0
        
        # Simple greedy matching
        # Mark matched truth to avoid double counting
        matched_truth_indices = set()
        
        for p in predicted:
            best_match = -1
            min_dist = float('inf')
            
            # Search candidate truths
            # Optimize: Window search?
            for i, t in enumerate(ground_truth):
                if i in matched_truth_indices: continue
                
                t_diff = abs(p.time - t.time)
                if t_diff > tolerance_time: continue
                
                p_diff = abs(p.pitch - t.pitch)
                if p_diff > tolerance_pitch: continue
                
                # Found candidate
                dist = t_diff + p_diff # Simple metric
                if dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match != -1:
                tp += 1
                matched_truth_indices.add(best_match)
            else:
                fp += 1
                
        fn = len(ground_truth) - len(matched_truth_indices)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "shift": 0.0
        }

    def verify_single_track(self, source_wav: str, source_midi: str) -> List[NoteEvent]:
        """Loads ground truth from MIDI for a single track (no mixing)."""
        truth_events = []
        try:
            pm = pretty_midi.PrettyMIDI(source_midi)
            for instrument in pm.instruments:
                if instrument.is_drum: continue
                for note in instrument.notes:
                    truth_events.append(NoteEvent(
                        time=note.start,
                        duration=note.end - note.start,
                        pitch=note.pitch,
                        velocity=note.velocity / 127.0,
                        source="vocals"
                    ))
        except Exception as e:
            print(f"[Benchmark] MIDI Error: {e}")
            
        return truth_events

    def auto_align_and_evaluate(self, predicted: List[NoteEvent], ground_truth: List[NoteEvent]) -> Dict:
        """
        Tries to find the best time offset (within -0.5s to +0.5s) AND Pitch Shift (Octaves) to maximize F1.
        """
        best_res = {"f1": -1}
        best_cfg = {"time": 0, "pitch": 0}
        
        # Coarse Search
        time_shifts = np.linspace(-0.5, 0.5, 21) # 50ms steps
        pitch_shifts = [0, 12, -12, 1, -1] # Check Octaves and Semitone errors
        
        p_base = [NoteEvent(p.time, p.duration, p.pitch, p.velocity, p.source) for p in predicted]
        
        for p_shift in pitch_shifts:
            for t_shift in time_shifts:
                # Apply shifts
                p_test = []
                for p in p_base:
                    new_n = NoteEvent(p.time + t_shift, p.duration, p.pitch + p_shift, p.velocity, p.source)
                    p_test.append(new_n)
                    
                res = self.evaluate(p_test, ground_truth, tolerance_time=0.1, tolerance_pitch=0.6)
                if res["f1"] > best_res["f1"]:
                    best_res = res
                    best_cfg = {"time": t_shift, "pitch": p_shift}
        
        print(f"[AutoAlign] Best: Time={best_cfg['time']:.3f}s, Pitch={best_cfg['pitch']}st -> F1: {best_res['f1']:.3f} (P:{best_res['precision']:.2f}, R:{best_res['recall']:.2f})")
        
        # DEBUG: Print Failure Cases for Best Match
        print("\n--- Failure Analysis (Best Match) ---")
        # Apply best shifts
        final_preds = []
        for p in p_base:
            final_preds.append(NoteEvent(p.time + best_cfg['time'], p.duration, p.pitch + best_cfg['pitch'], p.velocity, p.source))
            
        # Analyze False Negatives (Missed Truths)
        final_preds.sort(key=lambda x: x.time)
        ground_truth.sort(key=lambda x: x.time)
        
        matched_truth = set()
        for p in final_preds:
            for i, t in enumerate(ground_truth):
                if i in matched_truth: continue
                if abs(p.time - t.time) < 0.1 and abs(p.pitch - t.pitch) < 0.6:
                    matched_truth.add(i)
                    break
        
        missed_indices = [i for i in range(len(ground_truth)) if i not in matched_truth]
        print(f"Total Missed Truths (FN): {len(missed_indices)}")
        print("Scrutining first 5 misses:")
        for idx in missed_indices[:5]:
            t = ground_truth[idx]
            print(f"  MISS: Truth(t={t.time:.3f}, p={t.pitch:.1f})")
            # Find closest predicted neighbor
            nearest = min(final_preds, key=lambda x: abs(x.time - t.time))
            print(f"        Closest Pred: t={nearest.time:.3f}, p={nearest.pitch:.1f}, dist_t={nearest.time - t.time:.3f}, dist_p={nearest.pitch - t.pitch:.1f}")

        return best_res

def run_tests():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, "TestData", "kiritan_singing")
    
    # Use first file
    wav_file = os.path.join(test_data_dir, "wav", "01.wav")
    midi_file = os.path.join(test_data_dir, "midi_label", "01.mid")
    
    if not os.path.exists(wav_file):
        print(f"Test data not found at {wav_file}")
        return

    bench = ChoirBenchmark()
    
    # Single Track Ground Truth
    print(f"Loading Ground Truth from {os.path.basename(midi_file)}...")
    truth_single = bench.verify_single_track(wav_file, midi_file)
    print(f"Ground Truth Notes: {len(truth_single)}")
    
    print("\n=== CALIBRATION (Finding Optimal Offset) ===")
    from transcribe.basic_pitch import BasicPitchTranscriber
    bp = BasicPitchTranscriber()
    
    # Run with default params to find offset
    notes_calib = bp.transcribe(wav_file, "vocals", onset_threshold=0.5, frame_threshold=0.3)
    best_calib = bench.auto_align_and_evaluate(notes_calib, truth_single)
    
    # Store calibration offsets
    calib_time_shift = 0.0
    calib_pitch_shift = 0
    
    # Hacky way to extract best params from auto_align since it returns a Dict with F1, but prints the outcome
    # We need to modify auto_align to return the cfg
    # OR better: iterate here or trust the print? 
    # Let's trust the user to read the print, but for code we need the value.
    # Bench.auto_align_and_evaluate returns best_res (metrics).
    # I should have modified it to return the cfg too.
    # Let's just hardcode a wider search in evaluate() for the sweep? No, that's slow.
    
    # Let's modify auto_align in the class briefly? No, let's just do a mini-search here.
    
    # Mini-search for offset
    offset_search = np.linspace(-0.3, 0.3, 13) # 50ms steps
    best_offset = 0.0
    best_f1_calib = -1
    
    for t_off in offset_search:
        test_notes = [NoteEvent(n.time + t_off, n.duration, n.pitch, n.velocity, n.source) for n in notes_calib]
        res = bench.evaluate(test_notes, truth_single, tolerance_time=0.1)
        if res["f1"] > best_f1_calib:
            best_f1_calib = res["f1"]
            best_offset = t_off
            
    print(f"\n[Calibration] Applied Global Time Offset: {best_offset:.3f}s (F1: {best_f1_calib:.3f})")

    print("\n=== STARTING BASIC PITCH PARAMETER SWEEP ===")
    
    # Sweep Ranges
    onsets = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    frames = [0.2, 0.3, 0.4] # Removed 0.1 (Too noisy based on previous run)
    
    best_f1 = -1
    results = []
    
    for o in onsets:
        for f in frames:
            if f > o: continue 
            print(f"Testing Onset={o}, Frame={f}...", end="\r")
            try:
                notes = bp.transcribe(wav_file, "vocals", onset_threshold=o, frame_threshold=f)
                
                # Apply Calibration Offset
                for n in notes: n.time += best_offset
                
                res = bench.evaluate(notes, truth_single, tolerance_time=0.1, tolerance_pitch=0.6)
                res["onset"] = o
                res["frame"] = f
                results.append(res)
            except Exception as e:
                pass

    print("\n\n=== SWEEP RESULTS (Sorted by F1) ===")
    results.sort(key=lambda x: x["f1"], reverse=True)
    
    print(f"{'Onset':<6} | {'Frame':<6} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 65)
    for r in results:
        print(f"{r['onset']:<6.2f} | {r['frame']:<6.2f} | {r['f1']:<6.3f} | {r['precision']:<6.3f} | {r['recall']:<6.3f} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4}")

    print("\n=== RECOMMENDATION ===")
    valid = [r for r in results if r["precision"] > 0.4]
    if valid:
        valid.sort(key=lambda x: x["recall"], reverse=True)
        top = valid[0]
        print(f"Best High-Recall Config: Onset={top['onset']}, Frame={top['frame']} (Recall: {top['recall']:.3f})")
    else:
        print("No config met Precision > 0.4")

if __name__ == "__main__":
    run_tests()
