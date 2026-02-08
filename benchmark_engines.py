import os
import sys
import glob
import numpy as np
import mido
import time
from typing import List, Dict
from beatmap import NoteEvent
from transcribe.council import CouncilV2
from transcribe.smoother import VocalSmoother

# Configuration
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "TestData", "kiritan_singing")
WAV_DIR = os.path.join(TEST_DATA_DIR, "wav")
MIDI_DIR = os.path.join(TEST_DATA_DIR, "midi_label")
MAX_FILES = 5 # Limit for speed, set to None for full run

def load_ground_truth(midi_path: str) -> List[NoteEvent]:
    mid = mido.MidiFile(midi_path)
    events = []
    
    # Flatten tracks
    # We assume vocal melody is in the first track with notes
    # Kiritan MIDI might have multiple tracks?
    # Usually track 1.
    
    for track in mid.tracks:
        curr_time = 0.0
        active_notes = {} # note_num -> start_time
        
        has_notes = False
        track_events = []
        
        for msg in track:
            # Convert ticks to seconds
            # Standard MIDI file timing... complicated without tempo map.
            # mido.MidiFile(..., clip=True) handles headers.
            # actually mid.length is available.
            # We need to use 'tick2second'.
            pass
        
        # Simpler approach: verify track.
        # Check delta times.
        pass

    # Mido's iterating over messages yields absolute time if using 'play()' or manual calc.
    # But 'mid.print_tracks()' shows deltas.
    # We iterate with time accumulation.
    
    # Tempo handling is tricky. typical Kiritan datset is standard.
    # Let's assume 120bpm default or read meta?
    # Actually, mido has keys.
    
    # Use mido's built-in tick-to-second converter if available or generic.
    # Actually, iterate messages.
    
    tempo = 500000 # Default 120 BOM
    ticks_per_beat = mid.ticks_per_beat
    
    merged_events = []
    
    for track in mid.tracks:
        abs_time = 0.0
        active = {} # pitch -> start_time
        track_notes = []
        
        for msg in track:
            # Update time
            # time in seconds = tick * (tempo / 1000000) / ticks_per_beat
            # msg.time is in ticks.
            
            # Correction: mido msg.time is delta time in seconds IF iterating via 'play()', 
            # but delta ticks if raw.
            # Let's use simple tick accumulation and convert.
            
            dt_seconds = mido.tick2second(msg.time, ticks_per_beat, tempo)
            abs_time += dt_seconds
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = abs_time
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    start = active.pop(msg.note)
                    dur = abs_time - start
                    if dur > 0.05:
                        track_notes.append(NoteEvent(time=start, duration=dur, pitch=msg.note, velocity=1.0, source="truth"))
        
        if len(track_notes) > 0:
            merged_events.extend(track_notes)
            
    # Sort
    merged_events.sort(key=lambda x: x.time)
    return merged_events

def align_sequence(pred: List[NoteEvent], truth: List[NoteEvent], max_offset=2.0, step=0.05) -> float:
    """
    Finds best time offset to align pred to truth.
    Positive offset means Pred is LATER than Truth (Pred - Truth = Offset)
    So we need to Subtract offset from Pred to match Truth.
    """
    if not pred or not truth: return 0.0
    
    # Heuristic: Match first 10 notes?
    # Or just try shifts and maximize matches.
    
    best_offset = 0.0
    max_matches = 0
    
    # Coarse search
    offsets = np.arange(-max_offset, max_offset, step)
    
    p_times = np.array([p.time for p in pred])
    t_times = np.array([t.time for t in truth])
    
    for off in offsets:
        # Shift pred by -off (Sequence shift)
        # Check overlaps
        # Simple count of P within 0.1s of T
        matches = 0
        # Vectorized is hard with differing lengths.
        # fast loop
        
        # Shifted P
        p_shifted = p_times - off
        
        # For every t, is there a p close?
        # This is n*m. 
        # Optimize: 
        # Calculate diff matrix? Too big.
        
        # greedy match count
        matched_indices = set()
        for t in t_times:
            # Find closest p
            closest_idx = (np.abs(p_shifted - t)).argmin()
            dist = abs(p_shifted[closest_idx] - t)
            if dist < 0.1:
                matches += 1
        
        if matches > max_matches:
            max_matches = matches
            best_offset = off
            
    return best_offset

def evaluate(pred: List[NoteEvent], truth: List[NoteEvent], tol_time=0.1, tol_pitch=0.6, align=True):
    # Greedy matching
    # Sort both
    pred.sort(key=lambda x: x.time)
    truth.sort(key=lambda x: x.time)
    
    if align:
        offset = align_sequence(pred, truth)
        if abs(offset) > 0.01:
            # print(f"    (Auto-Align: {offset*1000:.0f}ms)")
            for p in pred:
                p.time -= offset
    
    matched_truth = set()
    matches = 0
    pitch_errors = []
    
    # For each predicted note, try to find a truth note
    for p in pred:
        best_t = None
        min_dist = float('inf')
        
        for i, t in enumerate(truth):
            if i in matched_truth: continue
            
            # Time check
            if abs(p.time - t.time) < tol_time:
                # Pitch check
                p_diff = abs(p.pitch - t.pitch)
                if p_diff < tol_pitch:
                    # Found candidate
                    dist = abs(p.time - t.time)
                    if dist < min_dist:
                        min_dist = dist
                        best_t = i
            
            if t.time > p.time + tol_time:
                break
        
        if best_t is not None:
            matched_truth.add(best_t)
            matches += 1
            pitch_dev = abs(p.pitch - truth[best_t].pitch)
            pitch_errors.append(pitch_dev)
            
    precision = matches / len(pred) if len(pred) > 0 else 0
    recall = matches / len(truth) if len(truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_pitch_err = np.mean(pitch_errors) if pitch_errors else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pitch_err": mean_pitch_err,
        "n_pred": len(pred),
        "n_truth": len(truth)
    }

def run_benchmark():
    files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    if not files:
        print("No WAV files found in TestData!")
        return

    if MAX_FILES: files = files[:MAX_FILES]
    
    council = CouncilV2()
    
    # Engines to test
    # (Name, ModelType, SmoothLevel)
    engines = [
        ("FCPE (Smooth 0.4)", "fcpe", 0.4),
        ("RMVPE (Raw)", "rmvpe", 0.0),
        ("RMVPE (Smooth 0.4)", "rmvpe", 0.4),
        ("CREPE (Raw)", "crepe", 0.0),
        ("CREPE (Smooth 0.4)", "crepe", 0.4),
        ("BasicPitch (Raw)", "basic_pitch", 0.0),
        ("Hybrid (Raw)", "hybrid", 0.0),
    ]
    
    results = {name: [] for name, _, _ in engines}
    
    print(f"Starting Benchmark on {len(files)} files...")
    print(f"{'Engine':<20} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'PitchErr':<8} | {'Notes':<6}")
    print("-" * 80)
    
    for f in files:
        basename = os.path.basename(f)
        mid_name = basename.replace(".wav", ".mid")
        mid_path = os.path.join(MIDI_DIR, mid_name)
        
        if not os.path.exists(mid_path):
            print(f"Skipping {basename}: No MIDI found.")
            continue
            
        try:
            truth = load_ground_truth(mid_path)
        except Exception as e:
            print(f"Error loading MIDI {mid_name}: {e}")
            continue
            
        if not truth:
            print(f"Skipping {basename}: Empty MIDI truth.")
            continue
            
        print(f"Benchmarking {basename} (Truth: {len(truth)} notes)...")
        # DEBUG ALIGNMENT
        print("  Truth[0:5]: ", [(round(n.time, 2), round(n.pitch, 1)) for n in truth[:5]])
        
        for name, mtype, slevel in engines:
            try:
                # Transcribe
                notes = council.transcribe(f, model_type=mtype)
                
                # Smooth
                if slevel > 0:
                    notes = VocalSmoother.smooth(notes, level=slevel)
                
                # DEBUG FIRST ENGINE ONLY
                if results[name] == []: 
                     print(f"  Pred({name})[0:5]: ", [(round(n.time, 2), round(n.pitch, 1)) for n in notes[:5]])
                     
                metrics = evaluate(notes, truth)
                results[name].append(metrics)
                
                print(f"  > {name:<16}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}, Err={metrics['pitch_err']:.2f}, N={metrics['n_pred']}")
                
            except Exception as e:
                print(f"  > {name:<16}: FAILED ({e})")
                
    print("\n--- AGGREGATE RESULTS ---")
    print(f"{'Engine':<20} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'PitchErr':<8} | {'AvgNotes':<8}")
    
    for name, _ in results.items():
        mets = results[name]
        if not mets: continue
        
        avg_p = np.mean([m['precision'] for m in mets])
        avg_r = np.mean([m['recall'] for m in mets])
        avg_f1 = np.mean([m['f1'] for m in mets])
        avg_err = np.mean([m['pitch_err'] for m in mets])
        avg_n = np.mean([m['n_pred'] for m in mets])
        
        print(f"{name:<20} | {avg_p:.3f}  | {avg_r:.3f}  | {avg_f1:.3f}  | {avg_err:.3f}    | {avg_n:.1f}")

if __name__ == "__main__":
    run_benchmark()
