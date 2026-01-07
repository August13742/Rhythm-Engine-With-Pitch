
import os
import sys

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from generator import RhythmEngine

def verify_song(song_name):
    stems_dir = os.path.join("stems", song_name)
    if not os.path.exists(stems_dir):
        print(f"Stems not found for {song_name}")
        return
        
    print(f"Verifying {song_name}...")
    engine = RhythmEngine(stems_dir)
    engine.run()
    print("Done.")

if __name__ == "__main__":
    verify_song("NamelessMartyr")
