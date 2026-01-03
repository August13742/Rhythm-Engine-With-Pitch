import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse

# ==========================================
#        V105: "THE UNCHAINED SYMPHONY"
# ==========================================

# HOLD CONFIGURATION
MIN_HOLD_DURATION_SEC = 0.20   
MAX_HOLD_DURATION_SEC = 1.50   
HOLD_ENERGY_DECAY     = 0.50   
HOLD_GAP_BUFFER       = 0.10   # 100ms visual gap between Hold End and Next Note

# MIXING
VOCAL_MASKING_STRENGTH = 0.7 
VOCAL_SOLO_BOOST       = 1.3 

BASE_PRIORITIES = {
    "vocals": 1.4,  
    "drums":  1.2,
    "bass":   1.1,
    "piano":  1.1,
    "guitar": 1.1,
    "other":  0.6
}

STEM_VOLUMES = {
    "vocals": 1.25,
    "drums":  0.9,
    "bass":   0.85,
    "piano":  0.9,
    "guitar": 0.85,
    "other":  0.6
}

VISUAL_RANGES = {
    "vocals": (48, 84),
    "piano":  (48, 88),
    "guitar": (40, 76),
    "bass":   (36, 60),
    "other":  (48, 88)
}

# REFINED DIFFICULTY
# Polyphony = How many notes can happen at once (Chords)
DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "grids": [4],       "poly": 1, "density_cap": 2.0 },
    "NORMAL": { "lanes": 4, "grids": [4, 8],    "poly": 2, "density_cap": 4.0 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 16],"poly": 3, "density_cap": 7.0 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24], "poly": 4, "density_cap": 14.0 },
}

SR = 44100
PYIN_FRAME_LENGTH = 4096 
HOP_LENGTH = 512

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[GEN] V105 Initializing (Strict Holds: {use_holds})...")
        self.stems_path = stems_path
        self.use_holds = use_holds
        
        self.manifest = self._load_manifest(stems_path)
        self.audio_data = self._smart_load_stems(stems_path)
        self.priorities = self._calculate_dynamic_priorities()
        
        self.envs = {}
        self.rms_curves = {} 
        self._preprocess_audio()

        self.rhythm_data = self._analyze_rhythm_composite()
        
        raw_pool = self._harvest_all()
        self.master_pool = self._score_and_sort(raw_pool, self.rhythm_data["beat_times"])
        print(f"[GEN] Master Pool Ready: {len(self.master_pool)} notes")

    def _load_manifest(self, stems_path):
        folder = os.path.dirname(stems_path["vocals"])
        manifest_path = os.path.join(folder, "stems_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return {k: {"exists": True, "is_silent": False, "peak_energy": 1.0} for k in stems_path.keys()}

    def _smart_load_stems(self, paths):
        loaded = {}
        max_len = 0
        print("[GEN] Loading Audio into Memory...")
        
        for name, path in paths.items():
            info = self.manifest.get(name, {"is_silent": False})
            if os.path.exists(path) and not info.get("is_silent", False):
                y, _ = librosa.load(path, sr=SR, mono=True)
                loaded[name] = y
                if len(y) > max_len: max_len = len(y)
            else:
                loaded[name] = None
        
        if max_len == 0: raise ValueError("Error: No valid audio found in stems.")

        final_audio = {}
        for name in paths.keys():
            data = loaded.get(name)
            if data is None:
                final_audio[name] = np.zeros(max_len, dtype=np.float32)
            elif len(data) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(data)] = data
                final_audio[name] = padded
            else:
                final_audio[name] = data
        return final_audio

    def _calculate_dynamic_priorities(self):
        p = BASE_PRIORITIES.copy()
        m = self.manifest

        vocal_weak = not m["vocals"].get("exists", False) or m["vocals"].get("peak_energy", 0) < 0.1
        drums_weak = not m["drums"].get("exists", False) or m["drums"].get("peak_energy", 0) < 0.1
        
        if vocal_weak:
            p["piano"]  = 1.3
            p["guitar"] = 1.3
            p["bass"]   = 1.2 
        if drums_weak:
            p["bass"]  = 1.3
            p["piano"] = 1.2
        return p

    def _preprocess_audio(self):
        for name, y in self.audio_data.items():
            if not self.manifest.get(name, {}).get("exists", False):
                self.envs[name] = np.zeros(1)
                self.rms_curves[name] = np.zeros(1)
                continue

            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LENGTH)[0]
            if rms.max() > 0: rms /= rms.max()
            self.rms_curves[name] = rms

            if name in ["piano", "guitar"]:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=SR)
            elif name == "drums":
                y_h, y_p = librosa.effects.hpss(y)
                self.envs[name] = librosa.onset.onset_strength(y=y_p, sr=SR)
            elif name == "other":
                # WAS: self.envs[name] = scipy.ndimage.gaussian_filter1d(env, sigma=2)
                # FIX: Treat it like a lead instrument (sharp attacks)
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=SR)
                # Normalize
                if self.envs[name].max() > 0:
                     self.envs[name] /= self.envs[name].max()
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=SR)
            
            if self.envs[name].max() > 0:
                self.envs[name] /= self.envs[name].max()

    def _analyze_rhythm_composite(self):
        m = self.manifest
        ref_track = self.audio_data["drums"] if len(self.audio_data["drums"]) > 0 else self.audio_data["vocals"]
        y_comp = np.zeros_like(ref_track)
        
        def add_layer(name, weight):
            if m[name]["exists"] and m[name]["peak_energy"] > 0.05:
                y_comp[:] += self.audio_data[name] * weight

        add_layer("drums", 1.0)
        add_layer("bass", 0.9)
        
        if m["drums"].get("peak_energy", 0) < 0.2:
            add_layer("piano", 1.2)
            add_layer("guitar", 1.2)
        else:
            add_layer("piano", 0.5)
            add_layer("guitar", 0.5)

        onset_env = librosa.onset.onset_strength(y=y_comp, sr=SR)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR)
        beat_times = librosa.frames_to_time(beat_frames, sr=SR)
        
        duration = librosa.get_duration(y=ref_track, sr=SR)
        return { "beat_times": beat_times, "duration": duration }

    def _harvest_all(self):
        pool = []
        m = self.manifest
        def exists(key): return m.get(key, {}).get("exists", False) and not m.get(key, {}).get("is_silent", False)

        if exists("drums"):
            pool.extend(self._harvest_onsets("drums", 36, can_hold=False, sensitivity=0.25))
        if exists("bass"):
            pool.extend(self._harvest_melodic("bass", (40, 400), 0.2, can_hold=False))
        if exists("piano"):
            pool.extend(self._harvest_melodic("piano", (27, 4000), 0.15, can_hold=False))
        if exists("guitar"):
            pool.extend(self._harvest_melodic("guitar", (80, 1200), 0.15, can_hold=False))
        if exists("vocals"):
            pool.extend(self._harvest_melodic("vocals", (50, 1000), 0.08, can_hold=True))
        if exists("other"):
            # FIX: Melodic harvesting for Sax/Synth
            # Range (100, 1500) covers Tenor Sax low notes up to High Trumpet/Synth
            # Sensitivity 0.15 matches Piano/Guitar
            # can_hold=True because Saxophones sustain notes!
            pool.extend(self._harvest_melodic("other", (100, 1500), 0.15, can_hold=True))

        return pool

    def _harvest_onsets(self, source, forced_midi, can_hold, sensitivity):
        if source not in self.envs: return []
        env = self.envs[source]
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        
        notes = []
        for f in frames:
            t = librosa.frames_to_time(f, sr=SR)
            dur = 0.0
            if can_hold and self.use_holds:
                dur = self._measure_signal_duration(source, f)
            notes.append({ "time": t, "midi": forced_midi, "dur": dur, "source": source, "score": env[f] })
        return notes

    # def _harvest_melodic(self, source, freq_range, sensitivity, can_hold=False):
    #     if source not in self.envs: return []
    #     env = self.envs[source]
    #     y = self.audio_data[source]
        
    #     frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=5)
        
    #     notes = []
    #     for f in frames:
    #         t = librosa.frames_to_time(f, sr=SR)
    #         if env[f] < sensitivity: continue

    #         start_samp = int(f * HOP_LENGTH)
    #         end_samp = start_samp + PYIN_FRAME_LENGTH
    #         if end_samp > len(y): break
            
    #         chunk = y[start_samp:end_samp]
    #         f0, _, _ = librosa.pyin(chunk, fmin=freq_range[0], fmax=freq_range[1], sr=SR, frame_length=PYIN_FRAME_LENGTH)
    #         f0 = f0[~np.isnan(f0)]
            
    #         midi = 0
    #         if len(f0) > 0:
    #             midi = int(round(librosa.hz_to_midi(np.median(f0))))
    #         else:
    #             continue

    #         dur = 0.0
    #         if can_hold and self.use_holds:
    #             dur = self._measure_signal_duration(source, f)

    #         notes.append({
    #             "time": t, "midi": midi, "dur": dur, "source": source,
    #             "score": env[f] * (1.2 if can_hold else 1.0)
    #         })
    #     return notes
    def _harvest_melodic(self, source, freq_range, sensitivity, can_hold=False):
        if source not in self.envs: return []
        env = self.envs[source]
        y = self.audio_data[source]
        
        is_vocal = (source == "vocals")
        
        # 1. RANGE CONFIGURATION
        # Widen vocal range significantly (E1 to D6) to allow dynamics
        if is_vocal:
            safe_fmin = 40   # Deep Bass (E1)
            safe_fmax = 1200 # Soprano High (D6)
        else:
            safe_fmin = freq_range[0]
            safe_fmax = freq_range[1]

        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=5)
        
        notes = []
        last_midi = None
        last_time = -999.0
        
        # 2. PHRASE THRESHOLD
        # If notes are closer than this, we assume they are connected
        PHRASE_THRESHOLD = 0.5 

        for f in frames:
            t = librosa.frames_to_time(f, sr=SR)
            if env[f] < sensitivity: continue

            # === TRANSIENT OFFSET ===
            # Vocals still get the 30ms offset to skip consonants.
            # Instruments (Piano/Bass) keep 0ms for tight rhythmic accuracy.
            offset_samples = int(0.030 * SR) if is_vocal else 0
            
            start_samp = int(f * HOP_LENGTH) + offset_samples
            end_samp = start_samp + PYIN_FRAME_LENGTH
            if end_samp > len(y): break
            
            chunk = y[start_samp:end_samp]
            
            # === PITCH DETECTION ===
            f0, voiced_flag, voiced_prob = librosa.pyin(
                chunk, 
                fmin=safe_fmin, 
                fmax=safe_fmax, 
                sr=SR, 
                frame_length=PYIN_FRAME_LENGTH,
                fill_na=np.nan
            )
            
            # === GATING (Refined) ===
            # Vocals: Strict (0.45) to remove breath/noise.
            # Insts:  Mild (0.20) to filter faint harmonics but keep the notes.
            gate = 0.45 if is_vocal else 0.20
            valid_f0 = f0[voiced_prob > gate]
            
            valid_f0 = valid_f0[~np.isnan(valid_f0)]
            if len(valid_f0) == 0: continue

            hz = np.median(valid_f0)
            midi = int(round(librosa.hz_to_midi(hz)))

            # === UNIVERSAL TEMPORAL BIAS ===
            # Applies to ALL instruments now, but with a safer threshold.
            if last_midi is not None:
                dt = t - last_time
                if dt < PHRASE_THRESHOLD:
                    dist_raw = abs(midi - last_midi)
                    
                    # TRIGGER CONDITION: > 13 Semitones
                    # Old code was > 7, which flattened 1-octave jumps (12).
                    # Now we allow 1-octave jumps. We only correct massive 
                    # errors like +19 (Octave+5th) or +24.
                    if dist_raw > 12: 
                        lower_oct = midi - 12
                        upper_oct = midi + 12
                        dist_lower = abs(lower_oct - last_midi)
                        dist_upper = abs(upper_oct - last_midi)
                        
                        # Only shift if it brings us MUCH closer (smooth step)
                        if dist_lower < 7 and dist_lower < dist_raw:
                            midi = lower_oct
                        elif dist_upper < 7 and dist_upper < dist_raw:
                            midi = upper_oct

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
        if start_frame >= len(rms): return 0.0
        
        peak_energy = rms[start_frame]
        threshold = peak_energy * HOLD_ENERGY_DECAY
        
        current_frame = start_frame + 1
        max_dist = int(MAX_HOLD_DURATION_SEC * SR / HOP_LENGTH)
        max_frame = min(len(rms), start_frame + max_dist)
        
        while current_frame < max_frame:
            if rms[current_frame] < threshold:
                break
            current_frame += 1
            
        dur_frames = current_frame - start_frame
        dur_sec = librosa.frames_to_time(dur_frames, sr=SR)
        if dur_sec < MIN_HOLD_DURATION_SEC: return 0.0
        return dur_sec

    def _get_vocal_energy(self, t):
        if "vocals" not in self.rms_curves: return 0.0
        frame = int(t * SR / HOP_LENGTH)
        rms = self.rms_curves["vocals"]
        if frame < len(rms): return rms[frame]
        return 0.0

    def _score_and_sort(self, pool, beat_times):
        if not pool: return []
        beat_arr = np.array(beat_times)
        
        for n in pool:
            n["score"] *= self.priorities.get(n["source"], 1.0)
            
            if self.manifest["vocals"]["exists"]:
                voc_energy = self._get_vocal_energy(n["time"])
                if n["source"] in ["piano", "guitar", "other"]:
                    if voc_energy > 0.2:
                        n["score"] *= VOCAL_MASKING_STRENGTH
                    else:
                        n["score"] *= VOCAL_SOLO_BOOST

            if len(beat_arr) > 0:
                idx = (np.abs(beat_arr - n["time"])).argmin()
                nearest = beat_arr[idx]
                dist = abs(n["time"] - nearest)
                if dist < 0.05: n["score"] *= 1.3
                elif dist < 0.10: n["score"] *= 1.1
            
            if n["dur"] > 0: n["score"] *= 1.1

        pool.sort(key=lambda x: x["time"])
        return pool

    def generate(self, diff_name):
        print(f"[GEN] Generating {diff_name}...")
        cfg = DIFF_CONFIGS[diff_name]
        
        # 1. Quantize FIRST (Align chords)
        quantized = self._quantize(self.master_pool, self.rhythm_data["beat_times"], cfg["grids"])
        
        # 2. Cluster Sieve (Handle Chords & Density)
        filtered = self._apply_cluster_sieve(quantized, cfg)
        
        # 3. Map Lanes
        mapped = self._allocate_lanes(filtered, cfg)
        
        # 4. Resolve Overlaps (Hold Truncation)
        cleaned = self._resolve_overlaps(mapped)
        
        # 5. Save
        self._save_beatmap(cleaned, diff_name)
        return cleaned

    def _quantize(self, pool, beats, grids):
        quantized = []
        # SOFT SNAP STRENGTH: 0.0 = Raw Audio, 1.0 = Robotic Grid, 0.5 = Natural Correction
        SNAP_STRENGTH = 0.5 

        for n in pool:
            t = n["time"]
            if len(beats) < 2: 
                quantized.append(n)
                continue
            
            # Find nearest beat window
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            
            # Calculate beat duration (tempo) at this specific moment
            if idx < len(beats) - 1: beat_dur = beats[idx+1] - beat_t
            elif idx > 0: beat_dur = beat_t - beats[idx-1]
            else: beat_dur = 0.5
            
            best_t = t
            min_err = 100.0
            
            # Find best grid slot
            for div in grids:
                step = beat_dur / (div / 4)
                offset = t - beat_t
                snapped_offset = round(offset / step) * step
                candidate = beat_t + snapped_offset
                err = abs(candidate - t)
                if err < min_err:
                    min_err = err
                    best_t = candidate
            
            # LOGIC CHANGE: Magnetic Snap
            # Only snap if we are fairly close (<60ms), but don't snap 100%
            if min_err < 0.06:
                # Interpolate between Raw Time (t) and Grid Time (best_t)
                new_time = t + (best_t - t) * SNAP_STRENGTH
                n["time"] = new_time
                
                # Quantize duration to look clean, but keep start time semi-loose
                if n["dur"] > 0:
                    eighth = beat_dur / 2
                    n["dur"] = round(n["dur"] / eighth) * eighth
            
            quantized.append(n)
        
        quantized.sort(key=lambda x: x["time"])
        return quantized

    def _apply_cluster_sieve(self, pool, cfg):
        # Groups simultaneous notes into chords and enforces caps
        final = []
        duration = self.rhythm_data["duration"]
        max_poly = cfg["poly"]
        density_cap = cfg["density_cap"]
        
        # Group by Time (Quantized)
        clusters = {}
        for n in pool:
            t = n["time"]
            if t not in clusters: clusters[t] = []
            clusters[t].append(n)
            
        sorted_times = sorted(clusters.keys())
        
        # Sliding Window for Density Cap
        window = 1.0
        accepted_clusters = []
        
        # We assume clusters are points in time. 
        # We simple-sieve the clusters based on the highest score in that cluster.
        
        # 1. Prune Clusters (Polyphony Limit)
        pruned_clusters = []
        for t in sorted_times:
            stack = clusters[t]
            stack.sort(key=lambda x: x["score"], reverse=True)
            # Take top N notes for this chord
            stack = stack[:max_poly]
            # Average score of the chord
            avg_score = sum(n["score"] for n in stack) / len(stack)
            pruned_clusters.append({ "time": t, "notes": stack, "score": avg_score })
            
        # 2. Density Sieve on Clusters
        # We select which *beats* happen, not just individual notes.
        cursor = 0.0
        idx = 0
        while cursor < duration:
            candidates = []
            while idx < len(pruned_clusters) and pruned_clusters[idx]["time"] < cursor + window:
                if pruned_clusters[idx]["time"] >= cursor:
                    candidates.append(pruned_clusters[idx])
                idx += 1
                
            if candidates:
                # How many beats allowed in this second?
                # Use density_cap. If cap is 5, we allow 5 clusters (roughly).
                # (Actually, density_cap usually means NOTES per second. 
                #  If we have chords, 5 notes might be just 2 chords.)
                #  Let's stick to NOTES per second for the cap.
                
                candidates.sort(key=lambda x: x["score"], reverse=True)
                
                current_notes = 0
                for cluster in candidates:
                    count = len(cluster["notes"])
                    if current_notes + count <= density_cap:
                        accepted_clusters.append(cluster)
                        current_notes += count
                        
            cursor += window
            
        # Unpack clusters back to flat list
        final = []
        for c in accepted_clusters:
            final.extend(c["notes"])
            
        final.sort(key=lambda x: x["time"])
        return final

    def _allocate_lanes(self, notes, cfg):
        lanes = cfg["lanes"]
        final_notes = []
        
        # Regroup (since sieve output is flat)
        groups = {}
        for n in notes:
            t = n["time"]
            if t not in groups: groups[t] = []
            groups[t].append(n)
            
        last_lane = 0
        times = sorted(groups.keys())
        
        for t in times:
            stack = groups[t]
            # Polyphony already handled in sieve, but safety check
            stack.sort(key=lambda x: x["midi"]) # Low pitch left, High right
            
            assigned = []
            
            # Smart Assignment
            if len(stack) == 1:
                # Single Note Logic
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
                
                if ideal == last_lane: # jitter to avoid straight lines
                    ideal = (ideal + 1) % lanes
                assigned.append(ideal)
                
            else:
                # Chord Logic: Spread evenly
                # e.g. 2 notes -> 0, 3 or 1, 2
                count = len(stack)
                if count == 2:
                    assigned = [0, 3] if last_lane in [1, 2] else [1, 2]
                elif count == 3:
                    assigned = [0, 1, 3] if last_lane == 2 else [0, 2, 3]
                else:
                    assigned = [0, 1, 2, 3][:count]

            for i, n in enumerate(stack):
                lane = assigned[i] if i < len(assigned) else i % lanes
                
                base_vol = STEM_VOLUMES.get(n["source"], 0.8)
                vol = np.clip(base_vol * (0.7 + (n["score"] * 0.3)), 0.0, 1.0)
                
                final_notes.append({
                    "time": n["time"],
                    "lane": lane,
                    "dur": n["dur"],
                    "type": "hold" if n["dur"] > 0 else "tap",
                    "midi": n["midi"],
                    "score": n["score"],
                    "source": n["source"],
                    "vol": vol
                })
                last_lane = int(np.mean(assigned))
        return final_notes

    def _resolve_overlaps(self, notes):
        notes.sort(key=lambda x: x["time"])
        lane_queues = {i: [] for i in range(4)}
        for n in notes:
            lane_queues[n["lane"]].append(n)
            
        cleaned = []
        for lane, lane_notes in lane_queues.items():
            if not lane_notes: continue
            
            for i in range(len(lane_notes)):
                current = lane_notes[i]
                
                if current["type"] == "hold":
                    if i + 1 < len(lane_notes):
                        next_note = lane_notes[i+1]
                        hold_end = current["time"] + current["dur"]
                        
                        # Check collision with buffer
                        limit = next_note["time"] - HOLD_GAP_BUFFER
                        
                        if hold_end > limit:
                            # Truncate
                            new_dur = limit - current["time"]
                            if new_dur < 0.1: # Too short, make tap
                                current["dur"] = 0
                                current["type"] = "tap"
                            else:
                                current["dur"] = new_dur
                
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
            
        with open(output_file, 'w') as f:
            json.dump(out, f, indent=2)
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
        for diff in ["EASY", "NORMAL", "HARD", "INSANE"]:
            gen.generate(diff)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GEN] Critical Error: {e}")