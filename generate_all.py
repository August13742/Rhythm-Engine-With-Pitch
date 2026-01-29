'''
generate_all.py - Batch generate beatmaps for all songs in the Music folder.
Version: V300 (Layered Generation)
'''
import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

# Suppress CUDA compatibility warnings for newer GPUs
import warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

def get_audio_files(directory):
    """Get all audio files from directory"""
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    audio_files = []
    
    if not os.path.exists(directory):
        return []
        
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory, file))
    
    return sorted(audio_files)

def process_song(audio_file, rebake=False, rechart=False, lanes=4, profile="STANDARD", skip_separation=False):
    """Process a single song by calling main.py"""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    print(f"\n{'='*80}")
    print(f"[BATCH] Processing: {base_name}")
    print(f"{'='*80}")
    
    cmd = [sys.executable, "main.py", audio_file, "--generate-only"]
    if rebake:
        cmd.append("--rebake")
    if rechart:
        cmd.append("--rechart")
    if skip_separation:
        cmd.append("--skip-separation")
    if lanes != 4:
        cmd.extend(["--lanes", str(lanes)])
    if profile != "STANDARD":
        cmd.extend(["--profile", profile])
        
    try:
        # Run main.py as a separate process to keep memory clean and rely on its logic
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"[BATCH] Success: {base_name}")
            return True
        else:
            print(f"[BATCH] Failed: {base_name} (Exit Code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"[BATCH] Execution Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch generate beatmaps for all songs in the Music folder")
    parser.add_argument("--music-dir", default="Music", help="Directory containing audio files")
    parser.add_argument("--rebake", action="store_true", help="Force regenerate beatmaps even if they exist")
    parser.add_argument("--rechart", action="store_true", help="Skip extraction and only re-run charting")
    parser.add_argument("--skip-separation", action="store_true", help="Skip audio separation")
    parser.add_argument("--lanes", type=int, default=4, help="Target lane count")
    parser.add_argument("--profile", type=str, default="STANDARD", choices=["STANDARD", "DRAFT", "RAW"], help="Chart profile")
    args = parser.parse_args()
    
    audio_files = get_audio_files(args.music_dir)
    if not audio_files:
        print(f"[BATCH] No audio files found in {args.music_dir}")
        return

    print(f"[BATCH] Found {len(audio_files)} songs to process.")
    
    success_count = 0
    start_all = time.time()
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[BATCH] Song {i}/{len(audio_files)}")
        if process_song(
            audio_path, 
            rebake=args.rebake, 
            rechart=args.rechart, 
            lanes=args.lanes, 
            profile=args.profile, 
            skip_separation=args.skip_separation
        ):
            success_count += 1
            
    duration = time.time() - start_all
    print(f"\n{'='*80}")
    print(f"[BATCH] Finished! Processed {success_count}/{len(audio_files)} songs successfully.")
    print(f"[BATCH] Total time: {duration/60:.1f} minutes.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
