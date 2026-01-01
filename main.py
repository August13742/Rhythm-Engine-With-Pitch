'''main.py'''
import argparse
import os
import sys
from visualizer import Visualizer
from separator import separate_audio

# Suppress CUDA compatibility warnings for newer GPUs
import warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: separate audio stems and visualize beatmap")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--skip-separation", action="store_true", help="Skip audio separation (use existing folder)")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Step 1: Separate audio into stems (unless skipped or already exists)
    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    folder_path = base_name
    
    # Check if stem files already exist
    required_stems = [
        os.path.join(folder_path, f"{base_name}_vocals.wav"),
        os.path.join(folder_path, f"{base_name}_other.wav"),
        os.path.join(folder_path, f"{base_name}_rhythm.wav")
    ]
    
    stems_exist = all(os.path.exists(stem) for stem in required_stems)
    
    if args.skip_separation:
        if not stems_exist:
            print("[ERROR] Folder or stems not found. Run without --skip-separation first.")
            sys.exit(1)
        print("[VIS] Using existing stems...")
    elif stems_exist:
        print("[VIS] Stems already exist, skipping Demucs...")
    else:
        folder_path = separate_audio(args.audio_file)
    
    # Step 2: Visualize with stems
    print("\n[VIS] Launching visualizer...")
    visualizer = Visualizer(args.audio_file, folder_path)
    visualizer.run()

if __name__ == "__main__":
    main()
