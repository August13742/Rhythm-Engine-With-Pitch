
import json
import sys
import numpy as np

def verify_constraints(json_path, difficulty="HARD"):
    print(f"Verifying {json_path} [{difficulty}]...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    notes = data["notes"]
    notes.sort(key=lambda x: x["time"])
    
    # Config (Mirroring Generator)
    limits = {
        "EASY": 0.25,
        "NORMAL": 0.16,
        "HARD": 0.08,
        "ALT_HARD": 0.07
    }
    min_int = limits.get(difficulty, 0.08)
    burst_int = 0.05
    
    violations = 0
    bursts = 0
    total_notes = len(notes)
    
    # 1. Check Global Intervals (Speed Limit)
    last_time = -10.0
    
    # Pre-cluster by time to handle chords vs rhythm
    unique_times = sorted(list(set(round(n["time"], 4) for n in notes)))
    
    for t in unique_times:
        dt = t - last_time
        if dt < min_int - 0.001: # Tolerance
            if dt < burst_int - 0.001:
                print(f"  [FAIL] Abs Limit Violation: {t:.3f}s (dt={dt:.3f}s)")
                violations += 1
            else:
                # Burst
                bursts += 1
        last_time = t
        
    # 2. Check Per-Lane Gaps
    lane_last_end = {}
    lane_violations = 0
    for n in notes:
        l = n["lane"]
        t = n["time"]
        prev = lane_last_end.get(l, -10.0)
        gap = t - prev
        if gap < 0.05 - 0.001: # 50ms absolute physical limit
             # print(f"  [FAIL] Lane {l} Gap Violation: {t:.3f}s (Gap={gap:.3f}s)")
             lane_violations += 1
        lane_last_end[l] = t + n.get("dur", 0)
        
    # 3. Check Holds
    bad_holds = 0
    for n in notes:
        if n["type"] == "hold":
            if n["dur"] < 0.75 - 0.001:
                 # print(f"  [FAIL] Short Hold: {n['time']:.3f}s (Dur={n['dur']:.3f}s)")
                 bad_holds += 1

    print("-" * 30)
    print(f"Total Notes: {total_notes}")
    print(f"Global Speed Failures: {violations}")
    print(f"Lane Gap Failures:     {lane_violations}")
    print(f"Bad Holds:             {bad_holds}")
    print(f"Allowed Bursts:        {bursts}")
    
    if violations == 0 and lane_violations == 0 and bad_holds == 0:
        print(">> PASSED")
        return True
    else:
        print(">> FAILED")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_constraints.py <json_file>")
        sys.exit(1)
    verify_constraints(sys.argv[1])
