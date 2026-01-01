import numpy as np
import librosa
import scipy.signal

# ==========================================
#       V56: CONFIGURATION & TUNING
# ==========================================

# --- 1. AUDIO ANALYSIS ---
ANALYSIS_FMIN = librosa.note_to_hz('C2') 
ANALYSIS_OCTAVES = 6

# --- 2. GLOBAL GATING ---
# Notes below this normalized volume (0.0-1.0) are ignored.
MIN_VOLUME_THRESHOLD = 0.10 

# --- 3. CHANNEL WEIGHTS (The Hierarchy) ---
WEIGHTS = { 
    "vocal":  1.15,  # Melody is King
    "other":  1.00,  # Instruments are the Foundation
    "rhythm": 0.85   # Drums are the Accent
}

# --- 4. HYSTERESIS (Sticky Focus) ---
# Prevents jittery switching. 
COOLDOWN_BEATS = 0.75   
SWITCH_PENALTY = 0.60   

# --- 5. GRID GRAVITY ---
# Penalizes snapping to complex grids to preserve musicality.
GRID_PENALTIES = {
    4: 1.0, 8: 1.0, 12: 2.0, 
    16: 1.2, 24: 2.5, 32: 3.0
}

# --- 6. FINGER BUDGET (Stamina System) ---
# soft_limit: How many fingers "comfortable" play requires.
# max_limit: The absolute maximum fingers allowed during a "Breakthrough" (climax).
# regen: How fast finger budget recovers per second. 
#        Higher = Sustained fast streams allowed. Lower = Bursts only.
DIFF_CONFIGS = {
    "EASY":   { 
        "lanes": 4, "grids": [4], "chords": False,
        "soft_fingers": 1.5, "max_fingers": 2.0, "finger_regen": 2.0 
    },
    "NORMAL": { 
        "lanes": 4, "grids": [4, 8], "chords": False,
        "soft_fingers": 2.0, "max_fingers": 2.5, "finger_regen": 3.0 
    },
    "HARD":   { 
        "lanes": 5, "grids": [4, 8, 12, 16], "chords": True,
        "soft_fingers": 2.0, "max_fingers": 3.0, "finger_regen": 5.0 
    },
    "INSANE": { 
        "lanes": 6, "grids": [4, 8, 12, 16, 24, 32], "chords": True,
        "soft_fingers": 2.0, "max_fingers": 4.0, "finger_regen": 8.0 
    },
}

# --- 7. SENSITIVITY ---
SENSITIVITY = {
    "EASY":   { "delta": 0.06, "wait": 4 }, 
    "NORMAL": { "delta": 0.05, "wait": 2 },
    "HARD":   { "delta": 0.04, "wait": 2 },
    "INSANE": { "delta": 0.03, "wait": 1 },
}

# ==========================================

class MapGenerator:
    def __init__(self, vocals_path, other_path, rhythm_path, is_instrumental=False):
        print("[GEN] Loading stems...")
        self.sr = 44100
        self.is_instrumental = is_instrumental
        
        self.y_voc, _ = librosa.load(vocals_path, sr=self.sr)
        self.y_oth, _ = librosa.load(other_path, sr=self.sr)
        self.y_rhy, _ = librosa.load(rhythm_path, sr=self.sr)
        
        self.env_voc = self._get_env(self.y_voc)
        self.env_oth = self._get_env(self.y_oth)
        self.env_rhy = self._get_env(self.y_rhy)

        self.y_voc_harm, _ = librosa.effects.hpss(self.y_voc, margin=3.0)
        self.y_oth_harm, _ = librosa.effects.hpss(self.y_oth, margin=3.0)
        
        self.data = self._analyze()

    def _get_env(self, y):
        env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if env.max() > 0: env /= env.max()
        return env

    def _analyze(self):
        print("[GEN] Extracting Rhythm Grid...")
        onset_rhy = librosa.onset.onset_strength(y=self.y_rhy, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_rhy, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        print("[GEN] Pitch Tracking...")
        if not self.is_instrumental:
            pitch_voc, mag_voc = librosa.piptrack(y=self.y_voc_harm, sr=self.sr, fmin=ANALYSIS_FMIN, threshold=0.05)
        else:
            pitch_voc, mag_voc = None, None
            
        pitch_oth, mag_oth = librosa.piptrack(y=self.y_oth_harm, sr=self.sr, fmin=ANALYSIS_FMIN, threshold=0.05)

        return {
            "beat_times": beat_times,
            "pitch_voc": pitch_voc, "mag_voc": mag_voc,
            "pitch_oth": pitch_oth, "mag_oth": mag_oth,
            "duration": librosa.get_duration(y=self.y_rhy, sr=self.sr)
        }

    def _get_candidates(self, env, pitch_grid, mag_grid, source_name, diff_name):
        sens = SENSITIVITY.get(diff_name, SENSITIVITY["NORMAL"])
        onset_frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sens["delta"], wait=sens["wait"])
        
        candidates = []
        n_bins, n_frames = mag_grid.shape if mag_grid is not None else (0, 0)

        for t in onset_frames:
            if env[t] < MIN_VOLUME_THRESHOLD: continue

            midi = 0
            if pitch_grid is not None:
                t_start = max(0, t-1)
                t_end = min(n_frames, t+3)
                local_mags = mag_grid[:, t_start:t_end]
                if local_mags.size > 0:
                    best_idx = np.unravel_index(np.argmax(local_mags), local_mags.shape)
                    if local_mags[best_idx] > 0.05:
                        freq = pitch_grid[best_idx[0], t_start + best_idx[1]]
                        if freq > 0: midi = librosa.hz_to_midi(freq)
            
            if midi == 0:
                if source_name == "rhythm": midi = 36 
                elif source_name == "vocal": continue 
                elif source_name == "other": continue 
            
            weight = WEIGHTS.get(source_name, 1.0)
            candidates.append({
                "time": librosa.frames_to_time(t, sr=self.sr),
                "midi": int(round(midi)),
                "score": env[t] * weight,
                "source": source_name,
                "raw_strength": env[t]
            })
        return candidates

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        
        # 1. EXTRACT
        vocs = [] if self.is_instrumental else self._get_candidates(self.env_voc, d["pitch_voc"], d["mag_voc"], "vocal", diff_name)
        oth = self._get_candidates(self.env_oth, d["pitch_oth"], d["mag_oth"], "other", diff_name)
        
        rhy_env = librosa.util.peak_pick(self.env_rhy, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=2)
        rhy = []
        for t in rhy_env:
            if self.env_rhy[t] >= MIN_VOLUME_THRESHOLD:
                rhy.append({ 
                    "time": librosa.frames_to_time(t, sr=self.sr), "midi": 36, 
                    "score": self.env_rhy[t] * WEIGHTS["rhythm"], 
                    "source": "rhythm", "raw_strength": self.env_rhy[t] 
                })

        # 2. MERGE & BUFFER CHORDS
        all_notes = vocs + oth + rhy
        all_notes.sort(key=lambda x: x["time"])
        
        merged = []
        if all_notes:
            curr = all_notes[0]
            chord_buffer = [curr]
            
            for next_n in all_notes[1:]:
                if next_n["time"] - curr["time"] < 0.06:
                    chord_buffer.append(next_n)
                else:
                    self._process_chord_buffer(merged, chord_buffer, cfg["chords"])
                    chord_buffer = [next_n]
                    curr = next_n
            self._process_chord_buffer(merged, chord_buffer, cfg["chords"])

        # 3. HYSTERESIS
        stable_notes = self._apply_hysteresis(merged, d["beat_times"])

        # 4. GAMEPLAY DIRECTOR (Physical Rules)
        final_notes = self._apply_gameplay_rules(stable_notes, d, cfg)
            
        return final_notes

    def _process_chord_buffer(self, output_list, buffer, allow_chords):
        buffer.sort(key=lambda x: x["score"], reverse=True)
        winner = buffer[0]
        output_list.append(winner)
        
        if allow_chords and len(buffer) > 1:
            loser = buffer[1]
            if loser["score"] > winner["score"] * 0.7:
                loser["time"] = winner["time"]
                output_list.append(loser)

    def _apply_hysteresis(self, notes, beat_times):
        if not notes: return []
        filtered = []
        last_source = notes[0]["source"]
        last_switch_time = notes[0]["time"]
        
        avg_beat = 0.5
        if len(beat_times) > 1: avg_beat = np.mean(np.diff(beat_times[:10]))
        cooldown_dur = avg_beat * COOLDOWN_BEATS
        
        for n in notes:
            if n["source"] != last_source:
                time_since_switch = n["time"] - last_switch_time
                if time_since_switch < cooldown_dur:
                    n["score"] *= SWITCH_PENALTY
                last_switch_time = n["time"]
                last_source = n["source"]
            
            if n["score"] > 0.15: 
                filtered.append(n)
        return filtered

    def _apply_gameplay_rules(self, notes, d, cfg):
        final_notes = []
        lanes = cfg["lanes"]
        
        # --- STATE TRACKING ---
        lane_free_time = [0.0] * lanes # When does this lane become playable? (Hold blocking)
        current_budget = 0.0 # How many "fingers" are tired?
        last_note_time = -999
        
        # Budget parameters
        SOFT_LIMIT = cfg["soft_fingers"]
        MAX_LIMIT = cfg["max_fingers"]
        REGEN_RATE = cfg["finger_regen"]
        
        # Pitch mapping setup
        melodic = [x["midi"] for x in notes if x["source"] != "rhythm"]
        if not melodic: melodic = [60]
        p_min, p_max = np.percentile(melodic, 5), np.percentile(melodic, 95)
        spread = max(1, p_max - p_min)
        
        last_lane = lanes // 2
        
        for n in notes:
            # 1. TIME SNAP
            snapped_t = self._weighted_smart_snap(n["time"], d["beat_times"], cfg["grids"])
            n["time"] = (snapped_t * 0.95) + (n["time"] * 0.05)
            
            # 2. FINGER BUDGET UPDATE
            dt = n["time"] - last_note_time
            if dt > 0:
                current_budget = max(0.0, current_budget - (dt * REGEN_RATE))
            
            # 3. BUDGET CHECK (The Gate)
            # Cost of a note = 1.0. 
            cost = 1.0
            
            # Can we afford this?
            # If we are under soft limit, yes.
            # If over soft limit, we need High Score to Break Through.
            is_breakthrough = False
            if current_budget + cost > SOFT_LIMIT:
                if current_budget + cost > MAX_LIMIT:
                    continue # Hard cap hit. Drop note.
                
                # Soft Cap Check: Only loud notes pass
                if n["score"] > 0.6: # High Score threshold
                    is_breakthrough = True
                else:
                    continue # Note too weak for current density
            
            # 4. DETERMINE DURATION (Holds)
            n["dur"] = 0.0
            if n["raw_strength"] > 0.85 and n["source"] != "rhythm":
                n["dur"] = 0.4 # Long press for strong melody
            
            # 5. LANE MAPPING & PHYSICS
            if n["source"] == "rhythm":
                target_lane = lanes // 2
                if n["score"] > 0.7: target_lane = 0 if last_lane >= lanes//2 else lanes-1
            else:
                midi = n["midi"]
                while midi < 48: midi += 12
                while midi > 84: midi -= 12
                pitch_norm = (midi - p_min) / spread
                ideal_lane = int(max(0, min(1, pitch_norm)) * (lanes - 1))
                target_lane = ideal_lane

            # 6. COLLISION & HOLD AVOIDANCE (Physics)
            # Try to place note in target_lane. If blocked by Hold, look nearby.
            final_lane = -1
            
            # Search order: Target -> Target+1 -> Target-1 -> ...
            search_offsets = [0, 1, -1, 2, -2]
            
            for offset in search_offsets:
                candidate = target_lane + offset
                if 0 <= candidate < lanes:
                    # Check 1: Is lane free? (Hold Blocking)
                    # Check 2: Is lane adjacent to a chord note at this EXACT time? (Visibility)
                    
                    is_blocked = n["time"] < lane_free_time[candidate]
                    
                    is_crowded = False
                    # Check concurrent notes (Chords)
                    if final_notes and abs(final_notes[-1]["time"] - n["time"]) < 0.01:
                        prev_lane = final_notes[-1]["lane"]
                        if abs(candidate - prev_lane) <= 1: # Adjacent or same
                            is_crowded = True
                            
                    if not is_blocked and not is_crowded:
                        final_lane = candidate
                        break
            
            if final_lane == -1:
                continue # Nowhere to place note (Physically impossible)
            
            # 7. COMMIT NOTE
            n["lane"] = final_lane
            
            # Update State
            lane_free_time[final_lane] = n["time"] + n["dur"] + 0.05 # Add tiny buffer
            current_budget += cost
            last_note_time = n["time"]
            last_lane = final_lane
            
            final_notes.append(n)
            
        return final_notes

    def _weighted_smart_snap(self, t, beats, allowed_grids):
        if len(beats) < 2: return t
        idx = (np.abs(beats - t)).argmin()
        closest_beat = beats[idx]
        if idx < len(beats)-1: beat_dur = beats[idx+1] - beats[idx]
        else: beat_dur = beats[idx] - beats[idx-1]
        
        best_time = t
        min_weighted_error = float('inf')
        for div in allowed_grids:
            grid_dur = beat_dur / (div / 4)
            steps = round((t - closest_beat) / grid_dur)
            candidate = closest_beat + (steps * grid_dur)
            raw_error = abs(candidate - t)
            penalty = GRID_PENALTIES.get(div, 2.0)
            weighted_error = raw_error * penalty
            if weighted_error < min_weighted_error:
                min_weighted_error = weighted_error
                best_time = candidate
        if min_weighted_error > 0.05: return t
        return best_time