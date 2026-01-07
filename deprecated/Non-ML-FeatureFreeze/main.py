'''main.py'''
import argparse
import os
import sys
from visualizer import Visualizer
from generator import MapGenerator
from separator import separate_audio

# Suppress CUDA compatibility warnings for newer GPUs
import warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: separate audio stems and visualize beatmap")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--skip-separation", action="store_true", help="Skip audio separation (use existing folder)")
    parser.add_argument("--rebake", action="store_true", help="Force regenerate beatmaps (skip loading from files)")
    parser.add_argument("--mode", choices=["high", "medium", "low"], default="high", help="Separation quality: high (3-pass SOTA), medium (2-pass), low (legacy)")
    parser.add_argument("--generate-only", action="store_true", help="Generate beatmaps only, do not launch visualizer")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Step 1: Separate audio into stems (unless skipped or already exists)
    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    folder_path = os.path.join("stems", base_name)
    
    # Check if stem files already exist
    required_stems = [
        os.path.join(folder_path, "vocals.wav"),
        os.path.join(folder_path, "other.wav"),
        os.path.join(folder_path, "bass.wav"),
        os.path.join(folder_path, "drums.wav"),
        os.path.join(folder_path, "piano.wav"),
        os.path.join(folder_path, "guitar.wav")
    ]
    
    stems_exist = all(os.path.exists(stem) for stem in required_stems)
    
    if args.skip_separation:
        if not stems_exist:
            print("[ERROR] Folder or stems not found. Run without --skip-separation first.")
            sys.exit(1)
        print("[VIS] Using existing stems...")
    elif stems_exist:
        print("[VIS] Stems already exist, skipping separation...")
    else:
        folder_path = separate_audio(args.audio_file, mode=args.mode)
    
    # Step 2: Check for generated beatmaps (unless rebake flag is set)
    if not args.rebake:
        beatmap_files = [
            os.path.join(folder_path, f"{base_name}_EASY.json"),
            os.path.join(folder_path, f"{base_name}_NORMAL.json"),
            os.path.join(folder_path, f"{base_name}_HARD.json"),
            os.path.join(folder_path, f"{base_name}_INSANE.json")
        ]
        
        if all(os.path.exists(bm) for bm in beatmap_files):
            print("[VIS] Generated beatmaps found, skipping generation...")
            if args.generate_only:
                print("[VIS] Generate-only mode, exiting...")
                return
            print("[VIS] Launching visualizer...")
            visualizer = Visualizer(args.audio_file, folder_path)
            visualizer.run()
            return
    
    if args.rebake:
        print("[VIS] Rebake flag set, forcing beatmap regeneration...")
    
    # Step 3: Generate beatmaps
    print("\n[VIS] Generating beatmaps...")
    stems_dict = {
        "vocals": os.path.join(folder_path, "vocals.wav"),
        "other":  os.path.join(folder_path, "other.wav"),
        "bass":   os.path.join(folder_path, "bass.wav"),
        "drums":  os.path.join(folder_path, "drums.wav"),
        "piano":  os.path.join(folder_path, "piano.wav"),
        "guitar": os.path.join(folder_path, "guitar.wav")
    }
    
    gen = MapGenerator(stems_dict, use_holds=True)
    gen.generate_all()
    print("[VIS] Beatmaps generated and saved!")
    
    if args.generate_only:
        return
    
    # Step 4: Launch visualizer (reads from generated files)
    print("\n[VIS] Launching visualizer...")
    visualizer = Visualizer(args.audio_file, folder_path)
    visualizer.run()

if __name__ == "__main__":
    main()
