import os
import sys
import numpy as np
import argparse
from typing import List, Dict

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transcribe.council import CouncilV2
from transcribe.basic_pitch import BasicPitchTranscriber as BP_Torch
try:
    from transcribe.basic_pitch_onnx import BasicPitchTranscriber as BP_ONNX
except ImportError:
    BP_ONNX = None

from beatmap import NoteEvent

def note_list_to_roll(notes: List[NoteEvent], duration: float, dt=0.01):
    """Converts note list to a piano roll grid for comparison."""
    steps = int(duration / dt)
    roll = np.zeros(steps)
    
    for n in notes:
        start_idx = int(n.time / dt)
        end_idx = int((n.time + n.duration) / dt)
        roll[start_idx:end_idx] = n.pitch
        
    return roll

def plot_comparison(results: Dict[str, List[NoteEvent]], duration: float, title="Council Meeting"):
    print("[Plotting skipped: matplotlib not found]")
    return

def run_council(audio_path):
    print(f" convening Council for {os.path.basename(audio_path)}...")
    
    council = CouncilV2()
    results = {}
    
    # 1. Monophonic Models
    try:
        results["FCPE"] = council.transcribe(audio_path, model_type="fcpe")
    except Exception as e:
        print(f"FCPE Failed: {e}")
        
    try:
        results["RMVPE"] = council.transcribe(audio_path, model_type="rmvpe")
    except Exception as e:
        print(f"RMVPE Failed: {e}")
        
    # 2. Polyphonic Models (Torch)
    try:
        bp_torch = BP_Torch()
        results["BP_Torch"] = bp_torch.transcribe(audio_path, "vocals")
    except Exception as e:
         print(f"BP_Torch Failed: {e}")

    # 3. Polyphonic Models (ONNX)
    if BP_ONNX:
        try:
            bp_onnx = BP_ONNX()
            results["BP_ONNX"] = bp_onnx.transcribe(audio_path, "vocals")
        except Exception as e:
            print(f"BP_ONNX Failed: {e}")
    else:
        print("BP_ONNX not available.")
        
    # Analysis
    print("\n--- COUNCIL MINUTES ---")
    for name, notes in results.items():
        count = len(notes)
        if count > 0:
            avg_pitch = np.mean([n.pitch for n in notes])
            avg_vel = np.mean([n.velocity for n in notes])
            print(f"{name}: {count} notes, Avg Pitch: {avg_pitch:.1f}, Avg Vel: {avg_vel:.2f}")
        else:
            print(f"{name}: 0 notes")
            
    # Need to verify if notes align?
    # Simple overlap check
    # Let's say FCPE is truth (for monophonic). How many BP notes overlap with FCPE?
    
    # Analysis: Check Intervals of "Extra" Notes
    if "FCPE" in results and "BP_Torch" in results:
        fcpe_notes = sorted(results["FCPE"], key=lambda x: x.time)
        bp_notes = sorted(results["BP_Torch"], key=lambda x: x.time)
        
        hits = 0
        harmonics = [] # intervals
        noise = 0
        
        for bp_n in bp_notes:
            # Find temporally overlapping FCPE notes
            overlaps = [f for f in fcpe_notes if (bp_n.time < f.time + f.duration) and (bp_n.time + bp_n.duration > f.time)]
            
            if not overlaps:
                # BP found a note where FCPE is silent. Could be a solo backing line or noise.
                # Check for "Is it a hold tail?" or just noise.
                noise += 1
                continue
                
            # Check pitch relation with overlaps
            is_match = False
            for f in overlaps:
                diff = abs(bp_n.pitch - f.pitch)
                if diff < 1.0:
                    is_match = True
                    break
                else:
                    # It's overlapping but different pitch -> Harmony Candidate
                    harmonics.append(diff)
            
            if is_match:
                hits += 1
                
        total_bp = len(bp_notes)
        print(f"\nAnalysis of BasicPitch Notes ({total_bp}):")
        print(f"  Matches Lead (FCPE): {hits} ({hits/total_bp*100:.1f}%)")
        print(f"  Harmonic Candidates: {len(harmonics)} ({len(harmonics)/total_bp*100:.1f}%)")
        print(f"  Standalone/Noise:    {noise} ({noise/total_bp*100:.1f}%)")
        
        # Analyze Harmonic Intervals
        if harmonics:
            # Round to nearest seminote
            intervals = [round(x) for x in harmonics]
            counts = {}
            for i in intervals:
                counts[i] = counts.get(i, 0) + 1
            
            # Sort by frequency
            print("\n  Top Intervals (Semitones distance from Lead):")
            sorted_intervals = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for iv, count in sorted_intervals[:8]:
                desc = "?"
                if iv == 12: desc = "Octave"
                elif iv == 7: desc = "Perfect 5th"
                elif iv == 5: desc = "Perfect 4th"
                elif iv == 4: desc = "Major 3rd"
                elif iv == 3: desc = "Minor 3rd"
                print(f"    {iv}: {count} notes ({desc})")
        
    # Plot
    import librosa
    dur = librosa.get_duration(path=audio_path)
    plot_comparison(results, dur, title=f"Comparison: {os.path.basename(audio_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file")
    args = parser.parse_args()
    
    run_council(args.file)
