"""
dependencies: 
pip install faster-whisper torchcrepe librosa soundfile numpy torch scipy
"""
import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse
import torch
import torchcrepe
from separator import generate_speech_notes

# ==========================================
#        V302: HYBRID ENGINE (SYNTHESIS)
#        ML Harvesting + Psychoacoustic Gameplay Logic
# ==========================================

# ==========================================
#        TUNING & CONFIGURATION
# ==========================================
TUNING = {
    "audio": {
        "sr": 44100, # Mixing SR
        "ai_sr": 16000, # AI Model SR
        "hop_length": 512,
    },
    "holds": {
        "min_dur": 0.15, 
        "max_dur": 5.00,
        "gap_buffer": 0.05, 
        "energy_decay": 0.50,
    },
    "mixing": {
        "vocal_mask": 0.8,
        "vocal_boost": 1.3,
        "stem_vol": {
            "vocals": 1.25, "drums": 0.9, "bass": 0.85,
            "piano": 0.9, "guitar": 0.85, "other": 0.6
        },
        "priorities": { # Base priorities before dynamic weighting
            "vocals": 2.0, "drums": 1.2, "bass": 1.0, 
            "piano": 1.1, "guitar": 1.1, "other": 0.6
        }
    },
    # V108: COINCIDENCE VOTING SYSTEM
    "coincidence": {
        "window": 0.05,          # 50ms perception window
        "min_support": 0.4,      # Score req from neighbors to trigger BOOST
        "boost_scale": 0.6,      # Multiplier intensity
        "mask_threshold": 0.4,   # Notes below this are vulnerable
        "mask_ratio": 2.5        # Neighbor must be Nx louder to kill
    },
    "harvest": {
        "drums_sens": 0.10,
        "bass_sens": 0.10,
        "piano_sens": 0.05,
        "guitar_sens": 0.04,
        "other_sens": 0.08,
        "vocal_sens": 0.05,
    }
}

VISUAL_RANGES = {
    "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
    "bass": (36, 60), "other": (48, 88)
}

# V108: HARD LIMIT COEFFICIENTS RESTORED
HARD_LIMIT_COEFF = 1.5 

DIFF_CONFIGS = {
    "EASY": { 
        "lanes": 4, "grids": [4], "poly": 1,
        "density_cap": 2.0, "min_score": 0.50, "chaos": 0.0 
    },
    "NORMAL": { 
        "lanes": 4, "grids": [4, 8], "poly": 2,
        "density_cap": 4.0, "min_score": 0.4, "chaos": 0.0 
    },
    "HARD": { 
        "lanes": 4, "grids": [4, 8, 12, 16], "poly": 3,
        "density_cap": 6.0, "min_score": 0.3, "chaos": 0.1 
    },
    "INSANE": { 
        "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], "poly": 4,
        "density_cap": 8.0, "min_score": 0.2, "chaos": 0.2 
    },
}

class MapGenerator:
    def __init__(self, stems_path, use_holds=True, speech_notes_path=None):
        print(f"[GEN] V302 HYBRID ENGINE INITIALIZING...")
        self.stems_path = stems_path
        self.use_holds = use_holds
        self.cfg = TUNING
        self.speech_notes_path = speech_notes_path
        
        # Load Manifest
        self.manifest = self._load_manifest(stems_path)
        
        # Load Audio (Standard 44.1k for DSP)
        self.audio_data = self._smart_load_stems(stems_path)
        
        # Precompute Envelopes
        self.envs = {}
        self.rms_curves = {} 
        self._preprocess_audio()

        # Rhythm Analysis
        self.rhythm_data = self._analyze_rhythm_composite()
        
        # V108: DYNAMIC WEIGHTING (Acoustic vs Band detection)
        self.vote_weights = self._calculate_dynamic_weights()
        
        # --- PHASE 1: HYBRID HARVEST ---
        raw_pool = self._harvest_all()
        print(f"[GEN] Total Candidates: {len(raw_pool)}")

        # --- PHASE 2: PSYCHOACOUSTIC VOTING (V108 RESTORED) ---
        # "The Banger Effect": Boost notes that happen across multiple stems simultaneously
        voted_pool = self._apply_coincidence_voting(raw_pool)

        # --- PHASE 3: SCORING & SORTING ---
        self.master_pool = self._score_and_sort(voted_pool, self.rhythm_data["beat_times"])

    # =========================================================
    #  V108 LOGIC: DYNAMICS & VOTING
    # =========================================================
    def _calculate_dynamic_weights(self):
        m = self.manifest
        weights = {
            "drums":  0.0, "bass":   0.0,
            "vocals": 0.0, "piano":  0.0,
            "guitar": 0.0, "other":  0.0
        }
        
        # Check presence
        if m.get("drums", {}).get("exists"): weights["drums"] = 1.0
        if m.get("bass", {}).get("exists"):  weights["bass"]  = 0.8
        
        rhythm_sum = weights["drums"] + weights["bass"]
        
        print("-" * 30)
        if rhythm_sum < 0.5:
            print("[GEN] >> MODE: ACOUSTIC (Melody Led)")
            weights["guitar"] = 1.0; weights["piano"] = 1.0; weights["vocals"] = 0.8 
        else:
            print("[GEN] >> MODE: BAND (Drum Led)")
            weights["guitar"] = 0.4; weights["piano"] = 0.4; weights["vocals"] = 0.5
            weights["other"] = 0.2
        print("-" * 30)
        return weights

    def _apply_coincidence_voting(self, pool):
        """
        Cross-Stem Coincidence Voting.
        If Guitar and Drums hit at the exact same time, both get a massive score boost.
        """
        print("[GEN] Applying Psychoacoustic Voting...")
        v_cfg = self.cfg["coincidence"]
        window = v_cfg["window"]
        pool.sort(key=lambda x: x["time"])
        
        stats = { "boosted": 0, "masked": 0, "unchanged": 0 }
        n_notes = len(pool)

        for i in range(n_notes):
            current = pool[i]
            t = current["time"]
            src = current["source"]
            
            support_score = 0.0
            masking_energy = 0.0
            
            # Scan Neighbors
            # Backward
            j = i - 1
            while j >= 0:
                neighbor = pool[j]
                if t - neighbor["time"] > window: break 
                if neighbor["source"] != src:
                    w = self.vote_weights.get(neighbor["source"], 0.2)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                j -= 1
            # Forward
            k = i + 1
            while k < n_notes:
                neighbor = pool[k]
                if neighbor["time"] - t > window: break 
                if neighbor["source"] != src:
                    w = self.vote_weights.get(neighbor["source"], 0.2)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                k += 1

            # Logic
            if support_score > v_cfg["min_support"]:
                boost = 1.0 + (support_score * v_cfg["boost_scale"])
                current["score"] *= boost
                stats["boosted"] += 1
            elif current["score"] < v_cfg["mask_threshold"]:
                # Vocals are immune to masking
                if src != "vocals" and masking_energy > (current["score"] * v_cfg["mask_ratio"]):
                    current["score"] *= 0.1
                    stats["masked"] += 1
                else: stats["unchanged"] += 1
            else: stats["unchanged"] += 1

        print(f"      > Boosted: {stats['boosted']} | Masked: {stats['masked']}")
        return pool

    # =========================================================
    #  CORE HARVESTING (V301 HYBRID)
    # =========================================================
    def _harvest_all(self):
        pool = []
        m = self.manifest
        h_cfg = self.cfg["harvest"]
        
        def exists(key): return m.get(key, {}).get("exists", False) and not m.get(key, {}).get("is_silent", False)

        # 1. SOTA VOCALS (Whisper + P-Center)
        if exists("vocals"):
            if self.speech_notes_path and os.path.exists(self.speech_notes_path):
                pool.extend(self._load_speech_notes())
            else:
                pool.extend(self._harvest_vocals_sota())

        # 2. LEGACY INSTRUMENTS (DSP)
        if exists("drums"):
            pool.extend(self._harvest_onsets("drums", 36, False, h_cfg["drums_sens"]))
        if exists("bass"):
            pool.extend(self._harvest_melodic_legacy("bass", (40, 400), h_cfg["bass_sens"], False))
        if exists("piano"):
            pool.extend(self._harvest_melodic_legacy("piano", (27, 4000), h_cfg["piano_sens"], False))
        if exists("guitar"):
            pool.extend(self._harvest_melodic_legacy("guitar", (80, 1200), h_cfg["guitar_sens"], False))
        if exists("other"):
            pool.extend(self._harvest_melodic_legacy("other", (100, 1500), h_cfg["other_sens"], True))

        return pool

    def _load_speech_notes(self):
        print(f"[GEN] Loading speech notes from file...")
        with open(self.speech_notes_path, 'r') as f:
            notes = json.load(f)
        notes = self._align_to_acoustic_onset(notes, "vocals")
        return notes

    def _harvest_vocals_sota(self):
        path = self.stems_path["vocals"]
        raw_notes = generate_speech_notes(path, use_holds=self.use_holds)
        refined_notes = self._align_to_acoustic_onset(raw_notes, "vocals")
        return refined_notes

    def _align_to_acoustic_onset(self, notes, stem_name="vocals"):
        """P-Center Correction (Look-ahead for vowels)"""
        print(f"[DSP] Aligning {stem_name} to Energy Peaks (Max Shift: 200ms)...")
        env = self.envs.get(stem_name)
        if env is None: return notes
        
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        search_fwd_frames = int(0.20 * sr / hop) 
        
        shifted_count = 0
        total_shift_ms = 0.0
        
        for n in notes:
            start_frame = int(n["time"] * sr / hop)
            end_frame = min(len(env), start_frame + search_fwd_frames)
            if end_frame <= start_frame: continue
            
            window = env[start_frame:end_frame]
            if len(window) == 0: continue
            
            peak_offset = np.argmax(window)
            peak_val = window[peak_offset]
            
            if peak_val > 0.15: 
                new_time = librosa.frames_to_time(start_frame + peak_offset, sr=sr)
                diff = new_time - n["time"]
                if diff > 0.01:
                    n["time"] = new_time
                    shifted_count += 1
                    total_shift_ms += (diff * 1000)

        if shifted_count > 0:
            avg = total_shift_ms / shifted_count
            print(f"      > Shifted {shifted_count}/{len(notes)} notes. Avg Delay: +{avg:.1f}ms")
        return notes

    def _harvest_onsets(self, source, forced_midi, can_hold, sensitivity):
        if source not in self.envs: return []
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        
        notes = []
        min_dur = self.cfg["holds"]["min_dur"]
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr)
            dur = 0.0
            note_type = "tap"
            if can_hold and self.use_holds:
                dur = self._measure_signal_duration(source, f)
                if dur >= min_dur: note_type = "hold"
            notes.append({ "time": t, "midi": forced_midi, "dur": dur, "source": source, "score": float(env[f]), "type": note_type })
        return notes

    def _harvest_melodic_legacy(self, source, freq_range, sensitivity, can_hold):
        if source not in self.envs: return []
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        pyin_len = 4096 
        
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=5)
        notes = []
        
        for f in frames:
            if env[f] < sensitivity: continue
            t = librosa.frames_to_time(f, sr=sr)
            
            start_samp = int(f * hop)
            end_samp = start_samp + pyin_len
            if end_samp > len(y): break
            chunk = y[start_samp:end_samp]
            
            f0, _, voiced_prob = librosa.pyin(chunk, fmin=freq_range[0], fmax=freq_range[1], sr=sr, frame_length=pyin_len)
            
            valid = f0[~np.isnan(f0)]
            if len(valid) == 0: continue
            
            hz = np.median(valid)
            midi = int(round(librosa.hz_to_midi(hz)))
            
            dur = 0.0
            if can_hold: dur = self._measure_signal_duration(source, f)
            
            notes.append({
                "time": t, "midi": midi, "dur": dur, "source": source,
                "score": float(env[f]), "type": "hold" if dur > 0 else "tap"
            })
        return notes

    # =========================================================
    #  HELPERS
    # =========================================================
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
        print("[GEN] Loading Audio Stems...")
        for name, path in paths.items():
            info = self.manifest.get(name, {"is_silent": False})
            if os.path.exists(path) and not info.get("is_silent", False):
                y, _ = librosa.load(path, sr=sr, mono=True)
                loaded[name] = y
                max_len = max(max_len, len(y))
            else: loaded[name] = None
        
        if max_len == 0: raise ValueError("No Audio Found")
        
        final = {}
        for name in paths.keys():
            data = loaded.get(name)
            if data is None: final[name] = np.zeros(max_len, dtype=np.float32)
            elif len(data) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(data)] = data
                final[name] = padded
            else: final[name] = data
        return final

    def _preprocess_audio(self):
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        for name, y in self.audio_data.items():
            if not self.manifest.get(name, {}).get("exists", False):
                self.envs[name] = np.zeros(1); self.rms_curves[name] = np.zeros(1)
                continue
            
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
            if rms.max() > 0: rms /= rms.max()
            self.rms_curves[name] = rms
            
            if name == "drums":
                y_h, y_p = librosa.effects.hpss(y)
                self.envs[name] = librosa.onset.onset_strength(y=y_p, sr=sr)
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=sr)
            
            if self.envs[name].max() > 0: self.envs[name] /= self.envs[name].max()

    def _measure_signal_duration(self, source, start_frame):
        if source not in self.rms_curves: return 0.0
        rms = self.rms_curves[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        h_cfg = self.cfg["holds"]
        
        threshold = rms[start_frame] * h_cfg["energy_decay"]
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        
        curr = start_frame + 1
        end = min(len(rms), start_frame + max_dist)
        while curr < end:
            if rms[curr] < threshold: break
            curr += 1
            
        dur = librosa.frames_to_time(curr - start_frame, sr=sr)
        return dur if dur > h_cfg["min_dur"] else 0.0

    def _get_vocal_energy(self, t):
        if "vocals" not in self.rms_curves: return 0.0
        frame = int(t * self.cfg["audio"]["sr"] / self.cfg["audio"]["hop_length"])
        rms = self.rms_curves["vocals"]
        if frame < len(rms): return rms[frame]
        return 0.0

    def _analyze_rhythm_composite(self):
        sr = self.cfg["audio"]["sr"]
        y_comp = self.audio_data["drums"] + (self.audio_data["bass"] * 0.8)
        onset_env = librosa.onset.onset_strength(y=y_comp, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return { "beat_times": librosa.frames_to_time(beat_frames, sr=sr), "duration": librosa.get_duration(y=y_comp, sr=sr) }

    def _score_and_sort(self, pool, beat_times):
        prio = self.cfg["mixing"]["priorities"]
        mix_cfg = self.cfg["mixing"]
        beat_arr = np.array(beat_times)
        
        for n in pool:
            n["score"] *= prio.get(n["source"], 1.0)
            
            # Vocal Masking (V108 logic integrated)
            if self.manifest["vocals"]["exists"]:
                voc_energy = self._get_vocal_energy(n["time"])
                if n["source"] in ["piano", "guitar", "other"]:
                    if voc_energy > 0.2:
                        n["score"] *= mix_cfg["vocal_mask"]
                    else:
                        n["score"] *= mix_cfg["vocal_boost"]

            # Beat Snap Bonus
            if len(beat_arr) > 0:
                idx = (np.abs(beat_arr - n["time"])).argmin()
                if abs(n["time"] - beat_arr[idx]) < 0.05: n["score"] *= 1.2
            
            if n.get("type") == "hold": n["score"] *= 1.1
        
        pool.sort(key=lambda x: x["time"])
        return pool

    # =========================================================
    #  GENERATION FLOW
    # =========================================================
    def generate(self, diff_name):
        print(f"[GEN] Generating {diff_name}")
        diff_cfg = DIFF_CONFIGS.get(diff_name, {})
        
        # 1. Quantize
        grids = diff_cfg.get("grids", [4])
        quantized = self._quantize(self.master_pool, self.rhythm_data["beat_times"], grids)
        
        # 2. Sieve (V108 Clustering + Hard Limits)
        filtered = self._apply_cluster_sieve_v108(quantized, diff_cfg)
        
        # 3. Lane Allocation (V108 Smart Mapping)
        mapped = self._allocate_lanes_v108(filtered, diff_cfg)
        
        # 4. Resolve Overlaps (V108 Flam Doctor)
        cleaned = self._resolve_overlaps_v108(mapped)
        
        self._save_beatmap(cleaned, diff_name)

    def generate_all(self):
        for diff in DIFF_CONFIGS.keys(): self.generate(diff)
        print("[GEN] All beatmaps generated and saved.")

    def _quantize(self, pool, beats, grids):
        out = []
        for n in pool:
            if len(beats) == 0: out.append(n); continue
            idx = (np.abs(beats - n["time"])).argmin()
            beat_t = beats[idx]
            beat_dur = 0.5 
            if idx < len(beats)-1: beat_dur = beats[idx+1] - beat_t
            
            best_t = n["time"]
            min_err = 100
            for div in grids:
                step = beat_dur / (div/4)
                target = round((n["time"] - beat_t) / step) * step + beat_t
                if abs(target - n["time"]) < min_err:
                    min_err = abs(target - n["time"])
                    best_t = target
            
            if min_err < 0.07: n["time"] = n["time"] + (best_t - n["time"]) * 0.5
            out.append(n)
        out.sort(key=lambda x: x["time"])
        return out

    # =========================================================
    #  V108 GAMEPLAY LOGIC (PORTED)
    # =========================================================
    def _apply_cluster_sieve_v108(self, pool, diff_cfg):
        # Uses V108's robust clustering to prevent impossibility
        final = []
        duration = self.rhythm_data["duration"]
        max_poly = diff_cfg["poly"]
        soft_limit = diff_cfg["density_cap"] 
        hard_limit = soft_limit * HARD_LIMIT_COEFF
        min_score = diff_cfg["min_score"]

        # 1. Group strictly by Time
        clusters = {}
        for n in pool:
            if n["score"] < min_score: continue
            t = n["time"]
            if t not in clusters: clusters[t] = []
            clusters[t].append(n)
            
        sorted_times = sorted(clusters.keys())
        pruned_clusters = []
        
        # 2. Prune Vertical Density
        for t in sorted_times:
            stack = clusters[t]
            stack.sort(key=lambda x: x["score"], reverse=True)
            stack = stack[:max_poly]
            avg_score = sum(n["score"] for n in stack) / len(stack)
            pruned_clusters.append({ "time": t, "notes": stack, "score": avg_score })

        # 3. Sliding Window Sieve
        window = 1.0; cursor = 0.0; idx = 0
        accepted_clusters = []
        
        while cursor < duration:
            candidates = []
            temp_idx = idx
            while temp_idx < len(pruned_clusters) and pruned_clusters[temp_idx]["time"] < cursor + window:
                if pruned_clusters[temp_idx]["time"] >= cursor:
                    candidates.append(pruned_clusters[temp_idx])
                temp_idx += 1
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            current_density = 0
            
            for cluster in candidates:
                count = len(cluster["notes"])
                if current_density + count <= soft_limit:
                    accepted_clusters.append(cluster)
                    current_density += count
                elif current_density + count <= hard_limit:
                    if cluster["score"] > 0.8: # Critical Hit only
                        accepted_clusters.append(cluster)
                        current_density += count
            
            cursor += 0.5
            while idx < len(pruned_clusters) and pruned_clusters[idx]["time"] < cursor: idx += 1

        # 4. Dedup
        unique_map = {}
        for c in accepted_clusters: unique_map[c["time"]] = c["notes"]
        for t in sorted(unique_map.keys()): final.extend(unique_map[t])
        
        return final

    def _allocate_lanes_v108(self, notes, diff_cfg):
        """
        V108 Allocation Logic.
        Uses Pitch to determine Vocal Lane (Lead vs Backing simulation).
        Handles Polyphony smartly.
        """
        lanes = diff_cfg["lanes"]
        chaos = diff_cfg["chaos"] 
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

            if count == 1:
                n = stack[0]
                src = n["source"]
                if src == "drums":
                    ideal = 1 if n["score"] > 0.8 else (0 if last_lane > 1 else 3)
                elif src == "vocals":
                    # Pitch-based biased spreading (V108 logic)
                    r_min, r_max = VISUAL_RANGES.get("vocals", (48, 84))
                    norm = (n["midi"] - r_min) / (r_max - r_min)
                    norm = max(0.0, min(1.0, norm))
                    # Map 0.0-1.0 to lanes 0-3 naturally
                    ideal = int(norm * 3.5)
                    if ideal > 3: ideal = 3
                else:
                    r_min, r_max = VISUAL_RANGES.get(src, (48, 84))
                    norm = (n["midi"] - r_min) / (r_max - r_min)
                    norm = max(0.0, min(1.0, norm))
                    ideal = int(norm * (lanes - 1))
                
                if chaos > 0 and np.random.rand() < chaos:
                    ideal = np.random.randint(0, lanes)
                elif ideal == last_lane: 
                    ideal = (ideal + 1) % lanes
                assigned.append(ideal)
            else:
                # Polyphony logic
                if count == 2:
                    assigned = [0, 3] if last_lane in [1, 2] else [1, 2]
                elif count == 3:
                    assigned = [0, 1, 3] if last_lane == 2 else [0, 2, 3]
                else:
                    assigned = [0, 1, 2, 3][:count]

            for i, n in enumerate(stack):
                lane = assigned[i] if i < len(assigned) else i % lanes
                n["lane"] = lane
                final_notes.append(n)
                last_lane = int(np.mean(assigned))
        return final_notes

    def _resolve_overlaps_v108(self, notes):
        # PHASE 1: THE FLAM DOCTOR (Fix Hidden Doubles)
        notes.sort(key=lambda x: x["time"])
        prio_map = {"vocals": 5, "drums": 4, "bass": 3, "piano": 3, "guitar": 2, "other": 1}
        accepted = []
        flam_window = 0.06 # 60ms
        
        for n in notes:
            collision_idx = -1
            for j in range(len(accepted) - 1, -1, -1):
                prev = accepted[j]
                if n["time"] - prev["time"] > flam_window: break 
                if prev["lane"] == n["lane"]:
                    collision_idx = j
                    break
            
            if collision_idx == -1:
                accepted.append(n); continue
                
            prev = accepted[collision_idx]
            p_prev = prio_map.get(prev["source"], 0)
            p_curr = prio_map.get(n["source"], 0)
            
            if p_curr > p_prev: victim, winner, victim_is_prev = prev, n, True
            else: victim, winner, victim_is_prev = n, prev, False
            
            # Try move victim to free lane (Create Chord)
            moved = False
            candidates = [l for l in range(4) if l != winner["lane"]]
            for cand_lane in candidates:
                is_free = True
                for check_n in accepted:
                    if abs(check_n["time"] - victim["time"]) < flam_window and check_n["lane"] == cand_lane:
                        is_free = False; break
                if is_free:
                    victim["lane"] = cand_lane; moved = True; break
            
            if moved:
                if not victim_is_prev: accepted.append(n)
            else:
                if victim_is_prev: accepted.pop(collision_idx); accepted.append(n)
        
        # PHASE 2: HOLD TRUNCATION
        accepted.sort(key=lambda x: x["time"])
        lane_queues = {i: [] for i in range(4)}
        for n in accepted: lane_queues[n["lane"]].append(n)
        
        cleaned = []
        gap = self.cfg["holds"]["gap_buffer"]
        min_dur = self.cfg["holds"]["min_dur"]
        
        for lane, lane_notes in lane_queues.items():
            if not lane_notes: continue
            for i in range(len(lane_notes)):
                current = lane_notes[i]
                if current.get("type") == "hold":
                    if i + 1 < len(lane_notes):
                        next_note = lane_notes[i+1]
                        hold_end = current["time"] + current["dur"]
                        limit = next_note["time"] - gap
                        if hold_end > limit:
                            new_dur = limit - current["time"]
                            if new_dur < min_dur: current["dur"] = 0; current["type"] = "tap"
                            else: current["dur"] = new_dur
                cleaned.append(current)
                
        cleaned.sort(key=lambda x: x["time"])
        return cleaned

    def _save_beatmap(self, notes, diff_name):
        vocals_path = self.stems_path["vocals"]
        base_dir = os.path.dirname(vocals_path)
        base_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"{base_name}_{diff_name}.json")
        serializable = []
        for n in notes:
            serializable.append({
                "time": round(n["time"], 3),
                "lane": int(n["lane"]),
                "dur": round(n["dur"], 3),
                "type": n["type"],
                "midi": int(n["midi"]),
                "source": n["source"]
            })
        with open(output_file, 'w') as f: json.dump(serializable, f, indent=2)
        print(f"[SAVE] {output_file} ({len(serializable)} notes)")

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
        gen.generate_all()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GEN] Critical Error: {e}")