'''main.py'''
import argparse
import os
import sys
from visualizer import Visualizer
from engine import RhythmEngine
from separator import separate_audio

# Suppress CUDA compatibility warnings for newer GPUs
import warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: separate audio stems and visualize beatmap")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--skip-separation", action="store_true", help="Skip audio separation (use existing folder)")
    parser.add_argument("--rebake", action="store_true", help="Force regenerate beatmaps (skip loading from files)")
    parser.add_argument("--generate-only", action="store_true", help="Generate beatmaps only, do not launch visualizer")
    parser.add_argument("--rechart", action="store_true", help="Skip extraction and only re-run charting (requires previous run)")
    parser.add_argument("--lanes", type=int, default=4, help="Target lane count (Default: 4)")
    parser.add_argument("--profile", type=str, default="STANDARD", choices=["STANDARD", "DRAFT", "RAW"], help="Chart generation profile: STANDARD, DRAFT, or RAW")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # --- PATH DEFINITIONS ---
    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    
    # 1. Stems Path: stems/{base_name}
    stems_dir = os.path.join("stems", base_name)
    
    # 2. Beatmaps Path: Beatmaps/{base_name}/
    beatmap_root = os.path.join("Beatmaps", base_name)
    
    # Ensure roots exist
    os.makedirs("stems", exist_ok=True)
    os.makedirs(beatmap_root, exist_ok=True)
    
    print(f"[Main] Processing: {base_name}")
    print(f"  > Stems: {stems_dir}")
    print(f"  > Beatmaps: {beatmap_root}")
    
    # --- STEP 1: SEPARATION ---
    # Check if stems exist
    required_stems = [
        os.path.join(stems_dir, "vocals.wav"),
        os.path.join(stems_dir, "other.wav"),
        os.path.join(stems_dir, "bass.wav"),
        os.path.join(stems_dir, "drums.wav"),
        # Piano/Guitar might be silent/missing, but check core 4 or logic from before
    ]
    # Simple check: Does directory exist and have some wavs
    stems_exist = os.path.isdir(stems_dir) and any(f.endswith(".wav") for f in os.listdir(stems_dir))
    
    if args.skip_separation:
        if not stems_exist:
             print("[ERROR] --skip-separation set but stems not found.")
             sys.exit(1)
        print("[Main] Skipping separation (User Requested)...")
    elif stems_exist:
        print("[Main] Stems already exist, skipping separation...")
    else:
        print("[Main] Separating audio...")
        separate_audio(args.audio_file, output_path=stems_dir)
        
    # --- STEP 2: BEATMAP GENERATION ---
    # Check if beatmaps exist
    difficulties = ["EASY", "NORMAL", "HARD", "ALT_HARD"]
    items = os.listdir(beatmap_root) if os.path.exists(beatmap_root) else []
    
    def check_exists(diff):
        # Check for ANY file starting with {diff} and ending with .json
        # This covers {diff}.json and {diff}_4k.json etc.
        for f in items:
            if f.startswith(diff) and f.endswith(".json"):
                return True
        return False
        
    beatmaps_exist = all(check_exists(d) for d in difficulties)
    
    should_generate = False
    if args.rebake:
        print("[Main] Rebake requested. Forcing generation.")
        should_generate = True
    elif args.rechart:
        print("[Main] Rechart requested. Regenerating charts from cached events.")
        should_generate = True
    elif not beatmaps_exist:
        print("[Main] Beatmaps missing. Generating...")
        should_generate = True
    else:
        print("[Main] Beatmaps already exist.")
        
    if should_generate:
        print(f"[Main] Launching RhythmEngine -> {beatmap_root}")
        engine = RhythmEngine(stems_dir, beatmap_root)
        engine.run(rechart=args.rechart, force_lanes=args.lanes, chart_profile=args.profile)
        print("[Main] Generation Complete.")
        
    if args.generate_only:
        print("[Main] Generate-only mode. Exiting.")
        return

    # --- STEP 3: VISUALIZER ---
    print("[Main] Launching Visualizer...")
    visualizer = Visualizer(args.audio_file, stems_dir, beatmap_root, target_lanes=args.lanes)
    visualizer.run()

if __name__ == "__main__":
    main()
