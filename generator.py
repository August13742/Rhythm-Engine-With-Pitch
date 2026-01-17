import os
import argparse
import sys

# Add project root to path if needed (e.g. if running directly as script)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import RhythmEngine 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rhythm Engine V300")
    parser.add_argument("folder", help="Path to the folder containing separated stems")
    parser.add_argument("--beatmap_folder", help="Optional override for output beatmap folder", default=None)
    parser.add_argument("--latency", help="Audio Engine Latency Offset (seconds)", type=float, default=-0.02)
    args = parser.parse_args()
    
    stems_folder = args.folder
    if args.beatmap_folder:
        out_folder = args.beatmap_folder
    else:
        # Infer from stems folder
        # stems_folder usually: .../stems/SongName/
        # want: .../Beatmaps/SongName/
        
        # Get Song Name
        # If folder ends in slash, dirname is same.
        norm_path = os.path.normpath(stems_folder)
        song_name = os.path.basename(norm_path)
        parent = os.path.dirname(norm_path) # .../stems
        root = os.path.dirname(parent) # .../
        
        # New structure
        out_folder = os.path.join(root, "Beatmaps", song_name)
        
    print(f"Beatmap Output: {out_folder}")
    
    engine = RhythmEngine(stems_folder, out_folder, audio_engine_latency_offset=args.latency)
    engine.run()
