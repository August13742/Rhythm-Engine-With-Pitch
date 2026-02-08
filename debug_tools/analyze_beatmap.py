
import json
import sys
import numpy as np

def analyze_beatmap(path):
    print(f"Analyzing {path}...")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    notes = data.get("notes", [])
    print(f"Total Notes: {len(notes)}")
    
    # Counts by Source
    sources = {}
    types = {}
    lanes = {}
    
    for n in notes:
        s = n.get("source", "unknown")
        t = n.get("type", "tap")
        l = n.get("lane", 0)
        
        sources[s] = sources.get(s, 0) + 1
        types[t] = types.get(t, 0) + 1
        lanes[l] = lanes.get(l, 0) + 1
        
    print("\n--- Sources ---")
    for k, v in sources.items():
        print(f"  {k}: {v}")
        
    print("\n--- Types ---")
    for k, v in types.items():
        print(f"  {k}: {v}")
        
    print("\n--- Lanes ---")
    for k, v in sorted(lanes.items()):
        print(f"  Lane {k}: {v}")

    # Gaps Histogram
    notes.sort(key=lambda x: x["time"])
    gaps = []
    unique_times = sorted(list(set(round(n["time"], 4) for n in notes)))
    last_t = unique_times[0]
    for t in unique_times[1:]:
        dt = t - last_t
        if dt < 0.2: # Only care about small gaps
            gaps.append(dt)
        last_t = t
            
    print("\n--- Small Gaps (< 0.2s) ---")
    # Bin limits
    bins = [0.0, 0.01, 0.05, 0.08, 0.10, 0.16, 0.20]
    hist, _ = np.histogram(gaps, bins=bins)
    
    for i in range(len(hist)):
        print(f"  {bins[i]:.3f} - {bins[i+1]:.3f}: {hist[i]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: analyze_beatmap.py <json_path>")
    else:
        analyze_beatmap(sys.argv[1])
