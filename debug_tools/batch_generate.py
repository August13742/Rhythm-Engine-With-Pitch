'''batch_generate.py - Generate beatmaps for all songs in a directory'''
import argparse
import os
import sys
from pathlib import Path
from generator import MapGenerator
from separator import separate_audio

# Suppress CUDA compatibility warnings for newer GPUs
import warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

def get_audio_files(directory):
    """Get all audio files from directory"""
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    audio_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory, file))
    
    return sorted(audio_files)

def process_song(audio_file, mode="high", skip_separation=False, rebake=False):
    """Process a single song: separate stems and generate beatmaps"""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = os.path.join("stems", base_name)
    
    print(f"\n{'='*60}")
    print(f"[BATCH] Processing: {base_name}")
    print(f"{'='*60}")
    
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
    
    # Step 1: Audio separation
    if skip_separation:
        if not stems_exist:
            print(f"[BATCH] ERROR: Stems not found for {base_name}, skipping...")
            return False
        print("[BATCH] Using existing stems...")
    elif stems_exist:
        print("[BATCH] Stems already exist, skipping separation...")
    else:
        print("[BATCH] Separating audio stems...")
        try:
            folder_path = separate_audio(audio_file, mode=mode)
        except Exception as e:
            print(f"[BATCH] ERROR: Separation failed for {base_name}: {e}")
            return False
    
    # Step 2: Check for existing beatmaps
    if not rebake:
        beatmap_files = [
            os.path.join(folder_path, f"{base_name}_EASY.json"),
            os.path.join(folder_path, f"{base_name}_NORMAL.json"),
            os.path.join(folder_path, f"{base_name}_HARD.json"),
            os.path.join(folder_path, f"{base_name}_INSANE.json")
        ]
        
        if all(os.path.exists(bm) for bm in beatmap_files):
            print("[BATCH] Beatmaps already exist, skipping generation...")
            return True
    
    if rebake:
        print("[BATCH] Rebake flag set, forcing beatmap regeneration...")
    
    # Step 3: Generate beatmaps
    print("[BATCH] Generating beatmaps for all difficulties...")
    stems_dict = {
        "vocals": os.path.join(folder_path, "vocals.wav"),
        "other":  os.path.join(folder_path, "other.wav"),
        "bass":   os.path.join(folder_path, "bass.wav"),
        "drums":  os.path.join(folder_path, "drums.wav"),
        "piano":  os.path.join(folder_path, "piano.wav"),
        "guitar": os.path.join(folder_path, "guitar.wav")
    }
    
    try:
        gen = MapGenerator(stems_dict, use_holds=True)
        gen.generate_all()
        print(f"[BATCH] ✓ Successfully generated beatmaps for {base_name}")
        return True
    except Exception as e:
        print(f"[BATCH] ERROR: Beatmap generation failed for {base_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate beatmaps for all songs in a directory"
    )
    parser.add_argument(
        "directory", 
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--skip-separation", 
        action="store_true", 
        help="Skip audio separation (use existing stems)"
    )
    parser.add_argument(
        "--rebake", 
        action="store_true", 
        help="Force regenerate beatmaps even if they exist"
    )
    parser.add_argument(
        "--mode", 
        choices=["high", "medium", "low"], 
        default="high", 
        help="Separation quality: high (3-pass SOTA), medium (2-pass), low (legacy)"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.directory):
        print(f"[BATCH] ERROR: Directory not found: {args.directory}")
        sys.exit(1)
    
    if not os.path.isdir(args.directory):
        print(f"[BATCH] ERROR: Not a directory: {args.directory}")
        sys.exit(1)
    
    # Get all audio files
    audio_files = get_audio_files(args.directory)
    
    if not audio_files:
        print(f"[BATCH] No audio files found in: {args.directory}")
        sys.exit(1)
    
    print(f"\n[BATCH] Found {len(audio_files)} audio file(s)")
    print("[BATCH] Files to process:")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Process each song
    print(f"\n[BATCH] Starting batch generation...")
    successful = []
    failed = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[BATCH] Progress: {i}/{len(audio_files)}")
        
        success = process_song(
            audio_file, 
            mode=args.mode,
            skip_separation=args.skip_separation,
            rebake=args.rebake
        )
        
        if success:
            successful.append(os.path.basename(audio_file))
        else:
            failed.append(os.path.basename(audio_file))
    
    # Summary
    print(f"\n{'='*60}")
    print("[BATCH] BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(audio_files)}")
    if successful:
        print("\nSuccessfully processed:")
        for name in successful:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(audio_files)}")
        print("Failed to process:")
        for name in failed:
            print(f"  ✗ {name}")
    
    print(f"\n[BATCH] Done! Check the stems/ directory for generated beatmaps.")

if __name__ == "__main__":
    main()
