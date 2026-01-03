import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse

# ==========================================
#        V106: "THE INFINITE HARVEST"
# ==========================================

# ==========================================
#        TUNING & CONFIGURATION
# ==========================================
TUNING = {
    "audio": {
        "sr": 44100,
        "hop_length": 512,
        "pyin_frame": 4096,
        "pyin_fmin": 40,    # Deep Bass E1
        "pyin_fmax": 1200,  # Soprano D6
    },
    "holds": {
        "min_dur": 0.5,
        "max_dur": 5.00,
        "energy_decay": 0.50,
        "gap_buffer": 0.10,
    },
    "mixing": {
        "vocal_mask": 0.7,
        "vocal_boost": 1.3,
        "stem_vol": {
            "vocals": 1.25, "drums": 0.9, "bass": 0.85,
            "piano": 0.9, "guitar": 0.85, "other": 0.6
        },
        "priorities": {
            "vocals": 1.4, "drums": 1.2, "bass": 1.1,
            "piano": 1.1, "guitar": 1.1, "other": 0.6
        }
    },
    # CRITICAL: HARVEST SENSITIVITY
    # Lower = More notes (including noise). 
    # We harvest LOOSELY here, then filter strictly in Diff Configs.
    "harvest": {
        "drums_sens": 0.10,      # Was 0.25. Now captures ghost notes/hi-hats.
        "bass_sens": 0.10,       # Was 0.20. Captures sub-bass rumbles.
        "piano_sens": 0.05,      # Was 0.10. Captures sustain tails/soft keys.
        "guitar_sens": 0.04,     # Was 0.10. Captures palm mutes.
        "vocal_sens": 0.03,      # Was 0.06. Captures breaths/whispers.
        "other_sens": 0.08,      # Was 0.12. Captures background synth.
        "vocal_gate": 0.45,      # Strict pitch gating for vocals
        "inst_gate": 0.15,       # Loose pitch gating for instruments
    },
    "quantization": {
        "snap_strength": 0.5,   # Slightly tighter snap
        "magnetic_radius": 0.07  # Wider grab radius (70ms)
    }
}

# ==========================================
#        DIFFICULTY DEFINITIONS
# ==========================================
# min_score: The gatekeeper. 
#   - EASY ignores everything below 0.5 (only loud hits).
#   - INSANE accepts 0.15 (background noise, ghost notes).
DIFF_CONFIGS = {
    "EASY": { 
        "lanes": 4, "grids": [4], 
        "poly": 1, "density_cap": 2.0, 
        "min_score": 0.50, "chaos": 0.0 
    },
    "NORMAL": { 
        "lanes": 4, "grids": [4, 8], 
        "poly": 2, "density_cap": 4.0, 
        "min_score": 0.35, "chaos": 0.0 
    },
    "HARD": { 
        "lanes": 4, "grids": [4, 8, 12, 16], 
        "poly": 3, "density_cap": 8.0, 
        "min_score": 0.25, "chaos": 0.1 
    },
    "INSANE": { 
        "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], 
        "poly": 4, "density_cap": 12.0, 
        "min_score": 0.15, "chaos": 0.2 
    },
}

VISUAL_RANGES = {
    "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
    "bass": (36, 60), "other": (48, 88)
}

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[GEN] V106 Initializing (Infinite Harvest Mode)...")
        self.stems_path = stems_path
        self.use_holds = use_holds
        self.cfg = TUNING
        
        self.manifest = self._load_manifest(stems_path)
        self.audio_data = self._smart_load_stems(stems_path)
        self.priorities = self._calculate_dynamic_priorities()
        
        self.envs = {}
        self.rms_curves = {} 
        self._preprocess_audio()

        self.rhythm_data = self._analyze_rhythm_composite()
        
        # HARVEST EVERYTHING. Filter later.
        raw_pool = self._harvest_all()
        self.master_pool = self._score_and_sort(raw_pool, self.rhythm_data["beat_times"])
        print(f"[GEN] Master Pool Size: {len(self.master_pool)} notes (Unfiltered)")

    def _load_manifest(self, stems_path):
        folder = os.path.dirname(stems_path["vocals"])
        manifest_path = os.path.join(folder, "stems_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f: return json.load(f)
        return {k: {"exists": True, "is_silent": False, "peak_energy": 1.0} for k in stems_path.keys()}

    def _smart_load_stems(self, paths):
        loaded = {}
        max_len = 0
        sr = self.cfg["audio"]["sr"]
        print("[GEN] Loading Audio...")
        
        for name, path in paths.items():
            info = self.manifest.get(name, {"is_silent": False})
            if os.path.exists(path) and not info.get("is_silent", False):
                y, _ = librosa.load(path, sr=sr, mono=True)
                loaded[name] = y
                if len(y) > max_len: max_len = len(y)
            else:
                loaded[name] = None
        
        if max_len == 0: raise ValueError("Error: No valid audio found.")

        final_audio = {}
        for name in paths.keys():
            data = loaded.get(name)
            if data is None: final_audio[name] = np.zeros(max_len, dtype=np.float32)
            elif len(data) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(data)] = data
                final_audio[name] = padded
            else: final_audio[name] = data
        return final_audio

    def _calculate_dynamic_priorities(self):
        p = self.cfg["mixing"]["priorities"].copy()
        m = self.manifest
        vocal_weak = not m["vocals"].get("exists", False) or m["vocals"].get("peak_energy", 0) < 0.1
        drums_weak = not m["drums"].get("exists", False) or m["drums"].get("peak_energy", 0) < 0.1
        
        if vocal_weak:
            p["piano"], p["guitar"], p["bass"] = 1.3, 1.3, 1.2 
        if drums_weak:
            p["bass"], p["piano"] = 1.3, 1.2
        return p

    def _preprocess_audio(self):
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        
        for name, y in self.audio_data.items():
            if not self.manifest.get(name, {}).get("exists", False):
                self.envs[name] = np.zeros(1)
                self.rms_curves[name] = np.zeros(1)
                continue

            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
            if rms.max() > 0: rms /= rms.max()
            self.rms_curves[name] = rms

            if name == "drums":
                y_h, y_p = librosa.effects.hpss(y)
                self.envs[name] = librosa.onset.onset_strength(y=y_p, sr=sr)
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=sr)
            
            if self.envs[name].max() > 0:
                self.envs[name] /= self.envs[name].max()

    def _analyze_rhythm_composite(self):
        m = self.manifest
        sr = self.cfg["audio"]["sr"]
        ref_track = self.audio_data["drums"] if len(self.audio_data["drums"]) > 0 else self.audio_data["vocals"]
        y_comp = np.zeros_like(ref_track)
        
        def add(name, weight):
            if m[name]["exists"] and m[name]["peak_energy"] > 0.05:
                y_comp[:] += self.audio_data[name] * weight

        add("drums", 1.0)
        add("bass", 0.9)
        if m["drums"].get("peak_energy", 0) < 0.2:
            add("piano", 1.2); add("guitar", 1.2)
        else:
            add("piano", 0.5); add("guitar", 0.5)

        onset_env = librosa.onset.onset_strength(y=y_comp, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        duration = librosa.get_duration(y=ref_track, sr=sr)
        return { "beat_times": beat_times, "duration": duration }

    def _harvest_all(self):
        pool = []
        m = self.manifest
        h_cfg = self.cfg["harvest"]
        
        def exists(key): return m.get(key, {}).get("exists", False) and not m.get(key, {}).get("is_silent", False)

        if exists("drums"):
            pool.extend(self._harvest_onsets("drums", 36, False, h_cfg["drums_sens"]))
        if exists("bass"):
            pool.extend(self._harvest_melodic("bass", (40, 400), h_cfg["bass_sens"], False))
        if exists("piano"):
            pool.extend(self._harvest_melodic("piano", (27, 4000), h_cfg["piano_sens"], False))
        if exists("guitar"):
            pool.extend(self._harvest_melodic("guitar", (80, 1200), h_cfg["guitar_sens"], False))
        if exists("vocals"):
            pool.extend(self._harvest_melodic("vocals", (50, 1000), h_cfg["vocal_sens"], True))
        if exists("other"):
            pool.extend(self._harvest_melodic("other", (100, 1500), h_cfg["other_sens"], True))

        return pool

    def _harvest_onsets(self, source, forced_midi, can_hold, sensitivity):
        if source not in self.envs: return []
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        # Use a very low 'delta' (sensitivity) to catch weak beats
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        
        notes = []
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr)
            dur = 0.0
            if can_hold and self.use_holds:
                dur = self._measure_signal_duration(source, f)
            notes.append({ "time": t, "midi": forced_midi, "dur": dur, "source": source, "score": env[f] })
        return notes

    def _harvest_melodic(self, source, freq_range, sensitivity, can_hold=False):
        if source not in self.envs: return []
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        pyin_len = self.cfg["audio"]["pyin_frame"]
        
        is_vocal = (source == "vocals")
        # Use wide range from config
        safe_fmin = 40 if is_vocal else freq_range[0]
        safe_fmax = 1200 if is_vocal else freq_range[1]

        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=5)
        
        notes = []
        last_midi = None
        last_time = -999.0
        PHRASE_THRESHOLD = 0.5 

        for f in frames:
            t = librosa.frames_to_time(f, sr=sr)
            if env[f] < sensitivity: continue

            offset_samples = int(0.030 * sr) if is_vocal else 0
            start_samp = int(f * hop) + offset_samples
            end_samp = start_samp + pyin_len
            if end_samp > len(y): break
            
            chunk = y[start_samp:end_samp]
            
            f0, _, voiced_prob = librosa.pyin(
                chunk, fmin=safe_fmin, fmax=safe_fmax, sr=sr, frame_length=pyin_len, fill_na=np.nan
            )
            
            gate = self.cfg["harvest"]["vocal_gate"] if is_vocal else self.cfg["harvest"]["inst_gate"]
            valid_f0 = f0[voiced_prob > gate]
            valid_f0 = valid_f0[~np.isnan(valid_f0)]
            if len(valid_f0) == 0: continue

            hz = np.median(valid_f0)
            midi = int(round(librosa.hz_to_midi(hz)))

            if last_midi is not None:
                dt = t - last_time
                if dt < PHRASE_THRESHOLD:
                    dist_raw = abs(midi - last_midi)
                    if dist_raw > 12: 
                        lower_oct = midi - 12
                        upper_oct = midi + 12
                        dist_lower = abs(lower_oct - last_midi)
                        dist_upper = abs(upper_oct - last_midi)
                        if dist_lower < 7 and dist_lower < dist_raw: midi = lower_oct
                        elif dist_upper < 7 and dist_upper < dist_raw: midi = upper_oct

            dur = 0.0
            if can_hold and self.use_holds:
                dur = self._measure_signal_duration(source, f)

            notes.append({
                "time": t, "midi": midi, "dur": dur, "source": source,
                "score": env[f] * (1.2 if can_hold else 1.0)
            })
            last_midi = midi
            last_time = t
            
        return notes

    def _measure_signal_duration(self, source, start_frame):
        if source not in self.rms_curves: return 0.0
        rms = self.rms_curves[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        h_cfg = self.cfg["holds"]
        
        if start_frame >= len(rms): return 0.0
        peak_energy = rms[start_frame]
        threshold = peak_energy * h_cfg["energy_decay"]
        
        current_frame = start_frame + 1
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        max_frame = min(len(rms), start_frame + max_dist)
        
        while current_frame < max_frame:
            if rms[current_frame] < threshold: break
            current_frame += 1
            
        dur_frames = current_frame - start_frame
        dur_sec = librosa.frames_to_time(dur_frames, sr=sr)
        if dur_sec < h_cfg["min_dur"]: return 0.0
        return dur_sec

    def _get_vocal_energy(self, t):
        if "vocals" not in self.rms_curves: return 0.0
        frame = int(t * self.cfg["audio"]["sr"] / self.cfg["audio"]["hop_length"])
        rms = self.rms_curves["vocals"]
        if frame < len(rms): return rms[frame]
        return 0.0

    def _score_and_sort(self, pool, beat_times):
        if not pool: return []
        beat_arr = np.array(beat_times)
        mix_cfg = self.cfg["mixing"]
        
        for n in pool:
            n["score"] *= self.priorities.get(n["source"], 1.0)
            
            if self.manifest["vocals"]["exists"]:
                voc_energy = self._get_vocal_energy(n["time"])
                if n["source"] in ["piano", "guitar", "other"]:
                    if voc_energy > 0.2:
                        n["score"] *= mix_cfg["vocal_mask"]
                    else:
                        n["score"] *= mix_cfg["vocal_boost"]

            if len(beat_arr) > 0:
                idx = (np.abs(beat_arr - n["time"])).argmin()
                dist = abs(n["time"] - beat_arr[idx])
                if dist < 0.05: n["score"] *= 1.3
                elif dist < 0.10: n["score"] *= 1.1
            
            if n["dur"] > 0: n["score"] *= 1.1

        pool.sort(key=lambda x: x["time"])
        return pool

    def generate(self, diff_name):
        print(f"[GEN] Generating {diff_name}...")
        diff_cfg = DIFF_CONFIGS[diff_name]
        
        # 1. Quantize
        quantized = self._quantize(self.master_pool, self.rhythm_data["beat_times"], diff_cfg["grids"])
        
        # 2. Sieve (Enforce Min Score & Density)
        filtered = self._apply_cluster_sieve(quantized, diff_cfg)
        
        # 3. Map Lanes
        mapped = self._allocate_lanes(filtered, diff_cfg)
        
        # 4. Resolve Overlaps
        cleaned = self._resolve_overlaps(mapped)
        
        # 5. Save
        self._save_beatmap(cleaned, diff_name)
        return cleaned

    def _quantize(self, pool, beats, grids):
        quantized = []
        q_cfg = self.cfg["quantization"]
        snap_strength = q_cfg["snap_strength"]
        mag_radius = q_cfg["magnetic_radius"]

        for n in pool:
            t = n["time"]
            if len(beats) < 2: 
                quantized.append(n); continue
            
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            
            if idx < len(beats) - 1: beat_dur = beats[idx+1] - beat_t
            elif idx > 0: beat_dur = beat_t - beats[idx-1]
            else: beat_dur = 0.5
            
            best_t = t
            min_err = 100.0
            
            for div in grids:
                step = beat_dur / (div / 4)
                offset = t - beat_t
                snapped_offset = round(offset / step) * step
                candidate = beat_t + snapped_offset
                err = abs(candidate - t)
                if err < min_err:
                    min_err = err
                    best_t = candidate
            
            if min_err < mag_radius:
                new_time = t + (best_t - t) * snap_strength
                n["time"] = new_time
                if n["dur"] > 0:
                    eighth = beat_dur / 2
                    n["dur"] = round(n["dur"] / eighth) * eighth
            
            quantized.append(n)
        
        quantized.sort(key=lambda x: x["time"])
        return quantized

    def _apply_cluster_sieve(self, pool, diff_cfg):
        # The logic here is now strict on score, loose on density for Insane
        final = []
        duration = self.rhythm_data["duration"]
        max_poly = diff_cfg["poly"]
        density_cap = diff_cfg["density_cap"]
        min_score = diff_cfg["min_score"]
        
        # Group by Time
        clusters = {}
        for n in pool:
            # DIFFICULTY GATING: This is the magic filter.
            if n["score"] < min_score: continue
            
            t = n["time"]
            if t not in clusters: clusters[t] = []
            clusters[t].append(n)
            
        sorted_times = sorted(clusters.keys())
        
        # Sliding Window
        window = 1.0
        accepted_clusters = []
        
        # 1. Prune Clusters
        pruned_clusters = []
        for t in sorted_times:
            stack = clusters[t]
            stack.sort(key=lambda x: x["score"], reverse=True)
            stack = stack[:max_poly]
            avg_score = sum(n["score"] for n in stack) / len(stack)
            pruned_clusters.append({ "time": t, "notes": stack, "score": avg_score })
            
        # 2. Density Sieve
        cursor = 0.0
        idx = 0
        while cursor < duration:
            candidates = []
            while idx < len(pruned_clusters) and pruned_clusters[idx]["time"] < cursor + window:
                if pruned_clusters[idx]["time"] >= cursor:
                    candidates.append(pruned_clusters[idx])
                idx += 1
                
            if candidates:
                candidates.sort(key=lambda x: x["score"], reverse=True)
                current_notes = 0
                for cluster in candidates:
                    count = len(cluster["notes"])
                    if current_notes + count <= density_cap:
                        accepted_clusters.append(cluster)
                        current_notes += count
            cursor += window
            
        final = []
        for c in accepted_clusters: final.extend(c["notes"])
        final.sort(key=lambda x: x["time"])
        return final

    def _allocate_lanes(self, notes, diff_cfg):
        lanes = diff_cfg["lanes"]
        chaos = diff_cfg["chaos"] # Jitter factor for higher diffs
        final_notes = []
        
        groups = {}
        for n in notes:
            t = n["time"]
            if t not in groups: groups[t] = []
            groups[t].append(n)
            
        last_lane = 0
        times = sorted(groups.keys())
        
        for t in times:
            stack = groups[t]
            stack.sort(key=lambda x: x["midi"]) 
            
            assigned = []
            count = len(stack)

            # --- Pattern Logic ---
            if count == 1:
                n = stack[0]
                src = n["source"]
                if src == "drums":
                    ideal = 1 if n["score"] > 0.8 else (0 if last_lane > 1 else 3)
                elif src == "other":
                    ideal = 0 if last_lane >= 2 else 3
                else:
                    r_min, r_max = VISUAL_RANGES.get(src, (48, 84))
                    norm = (n["midi"] - r_min) / (r_max - r_min)
                    norm = max(0.0, min(1.0, norm))
                    ideal = int(norm * (lanes - 1))
                
                # Apply Chaos (Randomness for Insane)
                if chaos > 0 and np.random.rand() < chaos:
                    ideal = np.random.randint(0, lanes)
                elif ideal == last_lane: 
                    ideal = (ideal + 1) % lanes
                assigned.append(ideal)
            else:
                # Chord Logic
                if count == 2:
                    assigned = [0, 3] if last_lane in [1, 2] else [1, 2]
                elif count == 3:
                    assigned = [0, 1, 3] if last_lane == 2 else [0, 2, 3]
                else:
                    assigned = [0, 1, 2, 3][:count]

            for i, n in enumerate(stack):
                lane = assigned[i] if i < len(assigned) else i % lanes
                base_vol = self.cfg["mixing"]["stem_vol"].get(n["source"], 0.8)
                vol = np.clip(base_vol * (0.7 + (n["score"] * 0.3)), 0.0, 1.0)
                
                final_notes.append({
                    "time": n["time"], "lane": lane, "dur": n["dur"],
                    "type": "hold" if n["dur"] > 0 else "tap",
                    "midi": n["midi"], "score": n["score"],
                    "source": n["source"], "vol": vol
                })
                last_lane = int(np.mean(assigned))
        return final_notes

    def _resolve_overlaps(self, notes):
        notes.sort(key=lambda x: x["time"])
        lane_queues = {i: [] for i in range(4)}
        for n in notes: lane_queues[n["lane"]].append(n)
        
        cleaned = []
        gap = self.cfg["holds"]["gap_buffer"]
        
        for lane, lane_notes in lane_queues.items():
            if not lane_notes: continue
            for i in range(len(lane_notes)):
                current = lane_notes[i]
                if current["type"] == "hold":
                    if i + 1 < len(lane_notes):
                        next_note = lane_notes[i+1]
                        hold_end = current["time"] + current["dur"]
                        limit = next_note["time"] - gap
                        if hold_end > limit:
                            new_dur = limit - current["time"]
                            if new_dur < 0.1: 
                                current["dur"] = 0; current["type"] = "tap"
                            else: current["dur"] = new_dur
                cleaned.append(current)
        cleaned.sort(key=lambda x: x["time"])
        return cleaned

    def _save_beatmap(self, notes, diff_name):
        vocals_path = self.stems_path["vocals"]
        base_dir = os.path.dirname(vocals_path)
        base_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"{base_name}_{diff_name}.json")
        
        out = []
        for n in notes:
            out.append({
                "time": float(f"{n['time']:.3f}"),
                "lane": int(n["lane"]),
                "dur": float(f"{n['dur']:.3f}"),
                "type": n["type"],
                "midi": int(n["midi"]),
                "score": float(f"{n['score']:.3f}"),
                "source": n["source"],
                "vol": float(f"{n['vol']:.3f}")
            })
            
        with open(output_file, 'w') as f: json.dump(out, f, indent=2)
        print(f"[GEN] Saved {len(out)} notes to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing stems")
    parser.add_argument("--holds", action="store_true", help="Enable hold notes")
    args = parser.parse_args()

    stems = {
        "vocals": os.path.join(args.folder, "vocals.wav"),
        "other":  os.path.join(args.folder, "other.wav"),
        "bass":   os.path.join(args.folder, "bass.wav"),
        "drums":  os.path.join(args.folder, "drums.wav"),
        "piano":  os.path.join(args.folder, "piano.wav"),
        "guitar": os.path.join(args.folder, "guitar.wav"),
    }
    
    try:
        gen = MapGenerator(stems, use_holds=args.holds)
        for diff in DIFF_CONFIGS.keys():
            gen.generate(diff)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GEN] Critical Error: {e}")