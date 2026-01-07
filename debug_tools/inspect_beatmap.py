
import json
import sys

def inspect_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        
    counts = {}
    for n in data["notes"]:
        s = n.get("source", "unknown")
        counts[s] = counts.get(s, 0) + 1
        
    print(f"Stats for {path}:")
    print(counts)

if __name__ == "__main__":
    inspect_json("stems/NamelessMartyr/beatmap/ALT_HARD.json")
