
import sys
import os
import json
import pickle

# Fix import path (rhythm_engine root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import ChartGenerator, NoteEvent, GENERATOR_CONFIG, DIFF_CONFIGS, STEM_TRANSCRIBE_CONFIG, StemSelector, EventFilter, Quantizer
from debug_tools.verify_constraints import verify_constraints
from debug_tools.analyze_beatmap import analyze_beatmap

def fast_test(song_name, difficulty="HARD"):
    cache_path = os.path.join("stems", song_name, "events_cache.pkl")
    manifest_path = os.path.join("stems", song_name, "stems_manifest.json")
    
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        print("Run main.py once to generate it.")
        return

    print(f"Loading cached events from {cache_path}...")
    with open(cache_path, 'rb') as f:
        all_events = pickle.load(f)
        
    print(f"Loaded {len(all_events)} events.")
    
    # Load Manifest
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
    # Init Generator
    gen = ChartGenerator(bpm=120.0) # BPM doesn't matter much for events, but used in quantization
    # We might need the real BPM. It's usually in metadata or we can guess.
    # Hack: Read BPM from existing easy map if available or just hardcode for testing
    # Actually, let's just use a default or try to read it from manifest/metadata if saved.
    
    # Generate
    print(f"Generating {difficulty}...")
    chart_data = gen.generate(list(all_events), difficulty, manifest=manifest)
    
    # Save
    out_dir = os.path.join("stems", song_name, "beatmap")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{difficulty}.json")
    
    with open(out_path, 'w') as f:
        json.dump(chart_data, f, indent=2)
        
    print(f"Saved to {out_path}")
    
    # Verify
    print("\n--- VERIFICATION ---")
    verify_constraints(out_path, difficulty)
    analyze_beatmap(out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_tools/fast_test.py <SongName> [Difficulty]")
        sys.exit(1)
        
    song = sys.argv[1]
    diff = sys.argv[2] if len(sys.argv) > 2 else "HARD"
    fast_test(song, diff)
