import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse

# ==========================================
#        V108
# at least I tried. last version before switching to full Machine Learning. This is decent
# for a map generator, but my perfectionism is killing me from the midi track generation error rate.
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
        "vocal_mask": 0.8,
        "vocal_boost": 1.3,
        "stem_vol": {
            "vocals": 1.25, "drums": 0.9, "bass": 0.85,
            "piano": 0.9, "guitar": 0.85, "other": 0.6
        },
        "priorities": {
            "vocals": 1.4, "drums": 1.2, "bass": 1.0,
            "piano": 1.1, "guitar": 1.1, "other": 0.6
        }
    },
    # CRITICAL: HARVEST SENSITIVITY
    # Lower = More notes (including noise). 
    # We harvest LOOSELY here, then filter strictly in Diff Configs.
    "harvest": {
        "drums_sens": 0.10,
        "bass_sens": 0.10,
        "piano_sens": 0.05,
        "guitar_sens": 0.04,
        "vocal_sens": 0.05,
        "other_sens": 0.08,
        "vocal_gate": 0.30,      # Standard gate
        "vocal_min_gate": 0.10,  # [V109] Rescue gate for loud choruses
        "inst_gate": 0.15,
    },
    # PSYCHOACOUSTIC VOTING PARAMS
    "coincidence": {
        "window": 0.05,          # 50ms perception window (Standard is 30-50ms)
        "min_support": 0.4,      # Score required from neighbors to trigger a BOOST
        "boost_scale": 0.6,      # Multiplier intensity (1.0 + support * scale)
        "mask_threshold": 0.4,   # Notes below this score are vulnerable to masking
        "mask_ratio": 2.5        # Neighbor energy must be Nx louder than note to kill it
    },
    "quantization": {
        "snap_strength": 0.5,   # Slightly tighter snap
        "magnetic_radius": 0.07  # Wider grab radius (70ms)
    }
}

# ==========================================
#        DIFFICULTY DEFINITIONS
# ==========================================
HARD_LIMIT_COEFF = 1.5 
DIFF_CONFIGS = {
    "EASY": { 
        "lanes": 4, "grids": [4], 
        "poly": 1, "density_cap": 2.0, 
        "min_score": 0.50, "chaos": 0.0 
    },
    "NORMAL": { 
        "lanes": 4, "grids": [4, 8], 
        "poly": 2, "density_cap": 4.0, 
        "min_score": 0.4, "chaos": 0.0 
    },
    "HARD": { 
        "lanes": 4, "grids": [4, 8, 12, 16], 
        "poly": 3, "density_cap": 5.5, 
        "min_score": 0.3, "chaos": 0.1 
    },
    "INSANE": { 
        "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], 
        "poly": 4, "density_cap": 7.0, 
        "min_score": 0.2, "chaos": 0.2 
    },
}

VISUAL_RANGES = {
    "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
    "bass": (36, 60), "other": (48, 88)
}

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[GEN] V108 Initializing (Transparent Mode)...")
        self.stems_path = stems_path
        self.use_holds = use_holds
        self.cfg = TUNING
        
        self.manifest = self._load_manifest(stems_path)
        self.audio_data = self._smart_load_stems(stems_path)
        
        # 1. Standard Priorities (User Config)
        self.priorities = self._calculate_dynamic_priorities()
        # 2. Structural Weights (Who leads the rhythm?)
        self.vote_weights = self._calculate_dynamic_weights()
        
        self.envs = {}
        self.rms_curves = {} 
        self._preprocess_audio()

        self.rhythm_data = self._analyze_rhythm_composite()
        
        # --- PHASE 1: HARVEST ---
        raw_pool = self._harvest_all()
        print(f"[GEN] Raw Harvest: {len(raw_pool)} candidates")

        # --- PHASE 2: PSYCHOACOUSTIC VOTING (NEW) ---
        # We do this BEFORE scoring priorities, to judge pure audio coincidence.
        voted_pool = self._apply_coincidence_voting(raw_pool)

        # --- PHASE 3: SCORING & FILTERING ---
        self.master_pool = self._score_and_sort(voted_pool, self.rhythm_data["beat_times"])
        print(f"[GEN] Master Pool Size: {len(self.master_pool)} notes (Weighted)")

    def _calculate_dynamic_weights(self):
        """
        Determines the 'Rhythmic Leader' based on stem existence.
        Used for the Coincidence Voting system.
        """
        m = self.manifest
        # Base weights assuming standard rock/pop
        weights = {
            "drums":  0.0, "bass":   0.0,
            "vocals": 0.0, "piano":  0.0,
            "guitar": 0.0, "other":  0.0
        }
        
        # 1. Establish Rhythm Section
        if m["drums"]["exists"]: weights["drums"] = 1.0
        if m["bass"]["exists"]:  weights["bass"]  = 0.8
        
        # 2. Check for Acoustic/No-Drum scenarios
        rhythm_sum = weights["drums"] + weights["bass"]
        
        print("-" * 30)
        print(f"[GEN] Analyzing Rhythmic Hierarchy (Rhythm Sum: {rhythm_sum:.2f})")
        
        if rhythm_sum < 0.5:
            # Acoustic Mode: Melodic instruments become the grid
            print("[GEN] >> MODE: ACOUSTIC (Melody Led)")
            weights["guitar"] = 1.0
            weights["piano"]  = 1.0
            weights["vocals"] = 0.8 
        else:
            # Band Mode: Melodic instruments are support
            print("[GEN] >> MODE: BAND (Drum Led)")
            weights["guitar"] = 0.4
            weights["piano"]  = 0.4
            weights["vocals"] = 0.5
            weights["other"]  = 0.2
        
        print(f"[GEN] Voting Weights: {json.dumps(weights)}")
        print("-" * 30)
        return weights

    def _apply_coincidence_voting(self, pool):
        """
        V108: Cross-Stem Coincidence Voting with Configurable Parameters.
        """
        print("[GEN] Applying Psychoacoustic Voting...")
        
        # Config Params
        v_cfg = self.cfg["coincidence"]
        window = v_cfg["window"]
        min_support = v_cfg["min_support"]
        boost_scale = v_cfg["boost_scale"]
        mask_threshold = v_cfg["mask_threshold"]
        mask_ratio = v_cfg["mask_ratio"]

        # Sort by time is critical for the sliding window
        pool.sort(key=lambda x: x["time"])
        
        n_notes = len(pool)
        
        # Stats Counters
        stats = { "boosted": 0, "masked": 0, "unchanged": 0 }

        for i in range(n_notes):
            current = pool[i]
            t = current["time"]
            src = current["source"]
            
            support_score = 0.0
            masking_energy = 0.0
            
            # --- BACKWARD SCAN ---
            j = i - 1
            while j >= 0:
                neighbor = pool[j]
                dt = t - neighbor["time"]
                if dt > window: break 
                
                if neighbor["source"] != src:
                    w = self.vote_weights.get(neighbor["source"], 0.2)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                j -= 1
                
            # --- FORWARD SCAN ---
            k = i + 1
            while k < n_notes:
                neighbor = pool[k]
                dt = neighbor["time"] - t
                if dt > window: break 
                
                if neighbor["source"] != src:
                    w = self.vote_weights.get(neighbor["source"], 0.2)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                k += 1

            # === LOGIC 1: THE SYNC BOOST (Structure) ===
            if support_score > min_support:
                boost = 1.0 + (support_score * boost_scale)
                current["score"] *= boost
                stats["boosted"] += 1
            
            # === LOGIC 2: THE MASKING KILLER (Clarity) ===
            elif current["score"] < mask_threshold:
                # --- NEW: VOCAL IMMUNITY ---
                # Vocals are the "Soul". They can never be masked by the band.
                if src == "vocals":
                    pass 
                
                elif masking_energy > (current["score"] * mask_ratio):
                    current["score"] *= 0.1 # Soft delete
                    stats["masked"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                stats["unchanged"] += 1

        print(f"[GEN] Voting Results: Boosted {stats['boosted']} | Masked {stats['masked']} | Unchanged {stats['unchanged']}")
        return pool

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
            
            # [V109] ENVELOPE BLENDING FOR VOCALS
            elif name == "vocals":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                if onset_env.max() > 0: onset_env /= onset_env.max()
                
                # Blend: 50% Transient, 50% Raw Energy
                # This ensures "monotone screaming" (High RMS, Low Transient) creates peaks
                self.envs[name] = (onset_env * 0.5) + (rms * 0.5)

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
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        
        notes = []
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr)
            dur = 0.0
            if can_hold and self.use_holds:
                dur = self._measure_signal_duration(source, f)
            notes.append({ "time": t, "midi": forced_midi, "dur": dur, "source": source, "score": float(env[f]) })
        return notes

    def _harvest_melodic(self, source, freq_range, sensitivity, can_hold=False):
        if source not in self.envs: return []
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        pyin_len = self.cfg["audio"]["pyin_frame"]
        
        is_vocal = (source == "vocals")
        safe_fmin = 40 if is_vocal else freq_range[0]
        safe_fmax = 1200 if is_vocal else freq_range[1]

        wait_time = 2 if is_vocal else 5
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=wait_time)
        
        notes = []
        
        # [V121] VIRTUAL VOICE ANCHORS
        voices = [None, None] 
        last_times = [-999.0, -999.0]
        PHRASE_THRESHOLD = 0.25

        for f in frames:
            t = librosa.frames_to_time(f, sr=sr)
            if env[f] < sensitivity: continue

            offset_samples = int(0.030 * sr) if is_vocal else 0
            start_samp = int(f * hop) + offset_samples
            end_samp = start_samp + pyin_len
            if end_samp > len(y): break
            chunk = y[start_samp:end_samp]
            
            f0, _, voiced_prob = librosa.pyin(chunk, fmin=safe_fmin, fmax=safe_fmax, sr=sr, frame_length=pyin_len, fill_na=np.nan)
            
            standard_gate = self.cfg["harvest"]["vocal_gate"] if is_vocal else self.cfg["harvest"]["inst_gate"]
            mask_std = voiced_prob > standard_gate
            
            if is_vocal:
                current_rms = self.rms_curves[source][min(f, len(self.rms_curves[source])-1)]
                mask_rescue = (voiced_prob > 0.10) & (current_rms > 0.35)
                valid_mask = mask_std | mask_rescue
            else:
                valid_mask = mask_std

            all_valid_f0 = f0[valid_mask]
            all_valid_f0 = all_valid_f0[~np.isnan(all_valid_f0)]
            if len(all_valid_f0) == 0: continue
            
            hz = np.median(all_valid_f0)
            midi = int(round(librosa.hz_to_midi(hz)))

            # [V121] LOGIC: DIARIZATION WITH ACTIVE DEFENSE & PANIC PROOFING
            assigned_voice_idx = 0 
            
            if is_vocal:
                # 1. Active Defense: Dynamic Decay
                silence_0 = t - last_times[0]
                silence_1 = t - last_times[1]
                
                # If partner is singing (silence < 2s), hold door open for 10s (Duet Mode).
                # If both silent, close door in 4s (Instrumental Reset).
                limit_0 = 10.0 if (silence_1 < 2.0) else 4.0
                limit_1 = 10.0 if (silence_0 < 2.0) else 4.0
                
                stale_0 = silence_0 > limit_0
                stale_1 = silence_1 > limit_1
                
                # 2. Get Distances
                dist_0 = abs(midi - voices[0]) if voices[0] is not None else 999
                dist_1 = abs(midi - voices[1]) if voices[1] is not None else 999
                
                # 3. Decision Matrix (Panic-Proof)
                if (voices[0] is None and voices[1] is None) or (stale_0 and stale_1):
                    assigned_voice_idx = 0 
                
                elif not stale_0 and (stale_1 or voices[1] is None):
                    # Voice 0 Primary. Only switch if jump is BIG (>7) AND Target is closer/empty.
                    jump_is_big = dist_0 > 7
                    target_is_better = (voices[1] is None) or (dist_1 < dist_0)
                    if jump_is_big and target_is_better: assigned_voice_idx = 1
                    else: assigned_voice_idx = 0
                    
                elif not stale_1 and (stale_0 or voices[0] is None):
                    # Voice 1 Primary.
                    jump_is_big = dist_1 > 7
                    target_is_better = (voices[0] is None) or (dist_0 < dist_1)
                    if jump_is_big and target_is_better: assigned_voice_idx = 0
                    else: assigned_voice_idx = 1
                    
                else:
                    # Panic Fallback: If both jumps are huge (>12), stick to recent history (Inertia)
                    if dist_0 > 12 and dist_1 > 12:
                         assigned_voice_idx = 0 if last_times[0] > last_times[1] else 1
                    else:
                        assigned_voice_idx = 0 if dist_0 <= dist_1 else 1

            # Retrieve context
            last_midi = voices[assigned_voice_idx]
            last_time_anchor = last_times[assigned_voice_idx]

            # [V121] BULLETPROOF GUARDRAILS (Double Octave + Tight Tolerance)
            if last_midi is not None:
                dt = t - last_time_anchor
                if dt < PHRASE_THRESHOLD:
                    if env[f] > 0.6: 
                        pass # Attack Trust
                    else:
                        dist_raw = abs(midi - last_midi)
                        if dist_raw > 12: 
                            # Strict Tolerance: < 2 (0 or 1 semitone error only)
                            # Preserves 9ths (diff 2) and 10ths (diff 4)
                            if abs((midi - 12) - last_midi) < 2: midi -= 12
                            elif abs((midi + 12) - last_midi) < 2: midi += 12
                            
                            # Double Octave Check (common in bass/low-mid)
                            elif dist_raw > 24:
                                if abs((midi - 24) - last_midi) < 2: midi -= 24
                                elif abs((midi + 24) - last_midi) < 2: midi += 24

            # Update Anchor
            voices[assigned_voice_idx] = midi
            last_times[assigned_voice_idx] = t

            dur = 0.0
            if can_hold and self.use_holds: dur = self._measure_signal_duration(source, f)

            notes.append({
                "time": t, "midi": midi, "dur": dur, "source": source,
                "score": float(env[f]),
                "voice_id": assigned_voice_idx if is_vocal else 0 
            })
            
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

    # [V109.1] Helper method for pitch smoothing
    def smooth_pitch(self, f0, voiced_prob, gate, window_size=5):
        """
        Applies a median filter to f0 values that are above the gate.
        Returns the median of the smoothed segment.
        """
        valid_indices = np.where(voiced_prob > gate)[0]
        if len(valid_indices) < window_size:
            # Not enough data to smooth, return median of raw valid
            raw_valid_f0 = f0[voiced_prob > gate]
            raw_valid_f0 = raw_valid_f0[~np.isnan(raw_valid_f0)]
            return [np.median(raw_valid_f0)] if len(raw_valid_f0) > 0 else [None]
        
        # Pad f0 for median filter
        padded_f0 = np.pad(f0, (window_size // 2, window_size // 2), mode='median')
        
        # Apply median filter
        smoothed_f0_full = scipy.ndimage.median_filter1d(padded_f0, size=window_size)
        
        # Extract smoothed value corresponding to the original frame indices
        smoothed_f0_center = smoothed_f0_full[window_size // 2 : window_size // 2 + len(f0)]
        
        # Median filter only where voiced probability is above gate
        final_smoothed_f0 = np.where(voiced_prob > gate, smoothed_f0_center, np.nan)
        
        # Return the median of the smoothed segment
        final_valid_f0 = final_smoothed_f0[~np.isnan(final_smoothed_f0)]
        return [np.median(final_valid_f0)] if len(final_valid_f0) > 0 else [None]

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

    def generate_all(self):
        """Generate all difficulties and save to files (no return)."""
        for diff in DIFF_CONFIGS.keys():
            self.generate(diff)
        print("[GEN] All beatmaps generated and saved.")

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
        final = []
        duration = self.rhythm_data["duration"]
        max_poly = diff_cfg["poly"]
        
        # Soft Cap Tuning
        # "soft_limit": The NPS where we start getting strict.
        # "hard_limit": The absolute ceiling (prevent crashes/impossible patterns).
        soft_limit = diff_cfg["density_cap"] 
        hard_limit = soft_limit * HARD_LIMIT_COEFF
        min_score = diff_cfg["min_score"]
        
        # 1. Group by Time
        clusters = {}
        for n in pool:
            if n["score"] < min_score: continue
            t = n["time"]
            if t not in clusters: clusters[t] = []
            clusters[t].append(n)
            
        sorted_times = sorted(clusters.keys())
        
        # 2. Prune Vertical Density (Poly) FIRST
        pruned_clusters = []
        for t in sorted_times:
            stack = clusters[t]
            # Keep only the loudest notes at this exact millisecond
            stack.sort(key=lambda x: x["score"], reverse=True)
            stack = stack[:max_poly]
            avg_score = sum(n["score"] for n in stack) / len(stack)
            pruned_clusters.append({ "time": t, "notes": stack, "score": avg_score })
            
        # 3. Apply Sliding Window (Horizontal Density)
        window = 1.0 # 1 second window
        cursor = 0.0
        idx = 0
        accepted_clusters = []
        
        # Optimization: We iterate through the song in 1s chunks
        while cursor < duration:
            candidates = []
            # Gather all note clusters visible in this 1s window
            temp_idx = idx
            while temp_idx < len(pruned_clusters) and pruned_clusters[temp_idx]["time"] < cursor + window:
                if pruned_clusters[temp_idx]["time"] >= cursor:
                    candidates.append(pruned_clusters[temp_idx])
                temp_idx += 1
            
            # Sort candidates by SCORE (Quality > Quantity)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            current_density = 0
            for cluster in candidates:
                count = len(cluster["notes"])
                
                # --- SOFT CAP LOGIC ---
                if current_density + count <= soft_limit:
                    # Under the limit? Come on in.
                    accepted_clusters.append(cluster)
                    current_density += count
                elif current_density + count <= hard_limit:
                    # Over the limit? ONLY accept if it's a "Banger"
                    # Threshold: 0.8 (Critical Hit)
                    if cluster["score"] > 0.8: 
                        accepted_clusters.append(cluster)
                        current_density += count
                # If over hard_limit, we ignore it regardless of score.
            
            # Advance cursor
            cursor += 0.5 # Overlap windows by 0.5s to ensure smoothness
            
            # Advance main index (optimization to not re-scan passed notes)
            while idx < len(pruned_clusters) and pruned_clusters[idx]["time"] < cursor:
                idx += 1

        # 4. Deduplicate
        # Since windows overlap, we might add the same cluster twice.
        unique_map = {}
        for c in accepted_clusters:
            unique_map[c["time"]] = c["notes"]
            
        final = []
        for t in sorted(unique_map.keys()):
            final.extend(unique_map[t])
            
        return final

    def _allocate_lanes(self, notes, diff_cfg):
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
                elif src == "other":
                    ideal = 0 if last_lane >= 2 else 3
                
                # [V120] VOCAL SPLIT LOGIC
                elif src == "vocals":
                    vid = n.get("voice_id", 0)
                    r_min, r_max = VISUAL_RANGES.get("vocals", (48, 84))
                    
                    # Normalize pitch 0.0 to 1.0
                    norm = (n["midi"] - r_min) / (r_max - r_min)
                    norm = max(0.0, min(1.0, norm))
                    
                    if vid == 0:
                        # Lead Singer: Biased Left (Lanes 0, 1, 2)
                        # Maps 0.0-1.0 to lane 0-2
                        ideal = int(norm * 2.5) 
                    else:
                        # Backing Singer: Biased Right (Lanes 1, 2, 3)
                        # Maps 0.0-1.0 to lane 1-3
                        ideal = 1 + int(norm * 2.5)
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
        # ==========================================
        # PHASE 1: THE FLAM DOCTOR (Fix Hidden Doubles)
        # ==========================================
        # Detects notes on the same lane < 60ms apart.
        # Solution: Move lower-priority note to free lane (Make Chord).
        # Fallback: Delete lower-priority note (Keep One).
        
        notes.sort(key=lambda x: x["time"])
        prio_map = {"vocals": 5, "drums": 4, "bass": 3, "piano": 3, "guitar": 2, "other": 1}
        
        accepted = []
        flam_window = 0.06 # 60ms collision threshold
        
        for n in notes:
            collision_idx = -1
            
            # Check against recently accepted notes for collision
            for j in range(len(accepted) - 1, -1, -1):
                prev = accepted[j]
                if n["time"] - prev["time"] > flam_window: break # Safe distance
                
                if prev["lane"] == n["lane"]:
                    collision_idx = j
                    break
            
            if collision_idx == -1:
                accepted.append(n)
                continue
                
            # === COLLISION DETECTED ===
            prev = accepted[collision_idx]
            
            # 1. Determine Winner (Higher Priority wins)
            p_prev = prio_map.get(prev["source"], 0)
            p_curr = prio_map.get(n["source"], 0)
            
            if p_curr > p_prev:
                victim, winner = prev, n
                victim_is_prev = True
            else:
                victim, winner = n, prev
                victim_is_prev = False
            
            # 2. Try to Move Victim to a Free Lane (Create Chord)
            moved = False
            candidates = [l for l in range(4) if l != winner["lane"]]
            
            for cand_lane in candidates:
                is_free = True
                # Check if candidate lane is busy at this time
                for check_n in accepted:
                    if abs(check_n["time"] - victim["time"]) < flam_window and check_n["lane"] == cand_lane:
                        is_free = False; break
                
                if is_free:
                    victim["lane"] = cand_lane
                    moved = True
                    break
            
            # 3. Commit Changes
            if moved:
                # If we moved 'prev', update it in place. If 'n', append it.
                if not victim_is_prev: accepted.append(n)
                # If victim was prev, it's already in 'accepted', just mutated.
            else:
                # 4. Failed to Move -> Execute Victim (Delete)
                if victim_is_prev:
                    accepted.pop(collision_idx) # Kill prev
                    accepted.append(n)      # Add current
                else:
                    pass # Kill current (do nothing)

        # ==========================================
        # PHASE 2: HOLD TRUNCATION (Standard Logic)
        # ==========================================
        accepted.sort(key=lambda x: x["time"])
        lane_queues = {i: [] for i in range(4)}
        for n in accepted: lane_queues[n["lane"]].append(n)
        
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