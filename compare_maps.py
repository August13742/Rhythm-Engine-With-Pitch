
import json
import os

def compare_counts(song_name, difficulty="HARD"):
    backup_path = f"Beatmaps/{song_name}/{difficulty}_backup.json"
    new_path = f"Beatmaps/{song_name}/{difficulty}.json"
    
    if not os.path.exists(backup_path):
        print(f"Backup not found: {backup_path}")
        return
    if not os.path.exists(new_path):
        print(f"New file not found: {new_path}")
        return
        
    with open(backup_path, 'r') as f:
        old_data = json.load(f)
    with open(new_path, 'r') as f:
        new_data = json.load(f)
        
    print(f"--- Note Count Comparison for {difficulty} ---")
    print(f"Old count: {len(old_data['notes'])}")
    print(f"New count: {len(new_data['notes'])}")
    print(f"Difference: {len(new_data['notes']) - len(old_data['notes'])}")

if __name__ == "__main__":
    compare_counts("UnfinishedJourney")
