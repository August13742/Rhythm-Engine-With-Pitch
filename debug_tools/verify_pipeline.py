
import os
import sys
import argparse
from typing import List

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import RhythmEngine, NoteEvent

def main():
    parser = argparse.ArgumentParser(description="Debug Rhythm Engine Pipeline")
    parser.add_argument("folder", help="Path to stems folder")
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Folder not found: {args.folder}")
        return

    print(f"--- Debugging Rhythm Engine on {args.folder} ---")
    
    # Initialize Engine
    engine = RhythmEngine(args.folder)
    print(f"BPM: {engine.bpm}")
    
    # Helper to print stats
    def print_stats(name, notes: List[NoteEvent]):
        if not notes:
            print(f"[{name}] No notes found.")
            return
            
        print(f"[{name}] {len(notes)} notes.")
        # Calc avg density (NPS)
        duration = notes[-1].time - notes[0].time if len(notes) > 1 else 1.0
        nps = len(notes) / duration
        print(f"  > NPS: {nps:.2f}")
        # Pitch range
        pitches = [n.pitch for n in notes]
        print(f"  > Pitch: {min(pitches):.1f} - {max(pitches):.1f}")
        
    # Run specific stems debugging
    # We can invoke internal methods or parts of run()
    
    # 1. Drums
    d_path = os.path.join(args.folder, "drums.wav")
    if os.path.exists(d_path):
        print("\n--- Testing Drums (Onset Decimation) ---")
        try:
            d_notes = engine._transcribe_drums_onset(d_path)
            print_stats("Drums", d_notes)
            
            # Check quantization roughly
            # histograms of interactions
        except Exception as e:
            print(f"Error processing drums: {e}")
            
    # 2. Piano (Multi-Pass)
    p_path = os.path.join(args.folder, "piano.wav")
    if os.path.exists(p_path):
        print("\n--- Testing Piano (BasicPitch + MultiPass) ---")
        try:
            # Manually invoke with config
            from generator import STEM_TRANSCRIBE_CONFIG
            cfg = STEM_TRANSCRIBE_CONFIG["piano"]
            print(f"Config: {cfg}")
            
            p_notes = engine.bp_transcriber.transcribe(
                p_path, 
                instrument_name="piano",
                onset_threshold=cfg["onset"],
                frame_threshold=cfg["frame"],
                multipass_consensus=cfg["multipass"],
                minimum_note_length=cfg["min_len"]
            )
            print_stats("Piano", p_notes)
        except Exception as e:
            print(f"Error processing piano: {e}")
            
    # 3. Bass
    b_path = os.path.join(args.folder, "bass.wav")
    if os.path.exists(b_path):
        print("\n--- Testing Bass ---")
        try:
             cfg = STEM_TRANSCRIBE_CONFIG["bass"]
             b_notes = engine.bp_transcriber.transcribe(
                b_path, 
                instrument_name="bass", 
                onset_threshold=cfg["onset"],
                frame_threshold=cfg["frame"],
                multipass_consensus=cfg["multipass"],
                minimum_note_length=cfg["min_len"]
            )
             print_stats("Bass", b_notes)
        except Exception as e:
             print(f"Error: {e}")

if __name__ == "__main__":
    main()
