import numpy as np
import librosa
import scipy.signal

# ==========================================
#       V58: CONFIGURATION & TUNING
# ==========================================

# --- 1. AUDIO ANALYSIS ---
ANALYSIS_FMIN = librosa.note_to_hz('C2') 
ANALYSIS_OCTAVES = 6

# --- 2. GLOBAL GATING ---
MIN_VOLUME_THRESHOLD = 0.15 

# --- 3. DYNAMIC MIXING (New in V58) ---
# Weights now vary by difficulty.
# Lower diffs = Crush the drums, boost the melody.
# Higher diffs = Let the drums punch through for density.
DIFF_WEIGHTS = {
    "EASY":   { "vocal": 1.30, "other": 1.20, "rhythm": 0.50 }, # Anti-Bass Mode
    "NORMAL": { "vocal": 1.20, "other": 1.10, "rhythm": 0.60 },
    "HARD":   { "vocal": 1.10, "other": 1.00, "rhythm": 0.70 },
    "INSANE": { "vocal": 1.05, "other": 1.00, "rhythm": 0.75 }  # Full Spectrum
}

# --- 4. HYSTERESIS ---
COOLDOWN_BEATS = 1.0   
SWITCH_PENALTY = 0.60   

# --- 5. GRID GRAVITY ---
GRID_PENALTIES = {
    4: 1.0, 8: 1.0, 12: 2.0, 
    16: 1.2, 24: 2.5, 32: 3.0
}

# --- 6. ERGONOMICS ---
MAX_CONSECUTIVE_JACKS = 3  
JACK_MIN_DELAY = 0.5      

# --- 7. DIFFICULTY SPECS ---
DIFF_CONFIGS = {
    "EASY":   { 
        "lanes": 4, "grids": [4], "chords": False, "min_dist": 0.40,
        "soft_fingers": 1.5, "max_fingers": 2.0, "finger_regen": 2.0 
    },
    "NORMAL": { 
        "lanes": 4, "grids": [4, 8], "chords": False, "min_dist": 0.25,
        "soft_fingers": 2.0, "max_fingers": 2.5, "finger_regen": 3.0 
    },
    "HARD":   { 
        "lanes": 5, "grids": [4, 8, 12, 16], "chords": True, "min_dist": 0.15,
        "soft_fingers": 2.0, "max_fingers": 3.0, "finger_regen": 5.0 
    },
    "INSANE": { 
        "lanes": 6, "grids": [4, 8, 12, 16, 24, 32], "chords": True, "min_dist": 0.10,
        "soft_fingers": 2.0, "max_fingers": 4.0, "finger_regen": 8.0 
    },
}

# --- 8. SENSITIVITY ---
SENSITIVITY = {
    "EASY":   { "delta": 0.06, "wait": 4 }, 
    "NORMAL": { "delta": 0.05, "wait": 3 },
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

    # UPDATE: Now accepts 'weights' dictionary
    def _get_candidates(self, env, pitch_grid, mag_grid, source_name, diff_name, weights):
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
            
            # Use Dynamic Weight
            weight = weights.get(source_name, 1.0)
            
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
        
        # 1. GET WEIGHTS FOR THIS DIFFICULTY
        current_weights = DIFF_WEIGHTS.get(diff_name, DIFF_WEIGHTS["NORMAL"])
        
        # 2. EXTRACT (Pass weights down)
        vocs = [] if self.is_instrumental else self._get_candidates(self.env_voc, d["pitch_voc"], d["mag_voc"], "vocal", diff_name, current_weights)
        oth = self._get_candidates(self.env_oth, d["pitch_oth"], d["mag_oth"], "other", diff_name, current_weights)
        
        rhy_env = librosa.util.peak_pick(self.env_rhy, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=2)
        rhy = []
        for t in rhy_env:
            if self.env_rhy[t] >= MIN_VOLUME_THRESHOLD:
                rhy.append({ 
                    "time": librosa.frames_to_time(t, sr=self.sr), "midi": 36, 
                    "score": self.env_rhy[t] * current_weights["rhythm"], 
                    "source": "rhythm", "raw_strength": self.env_rhy[t] 
                })

        # 3. MERGE & BUFFER CHORDS
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

        # 4. HYSTERESIS
        stable_notes = self._apply_hysteresis(merged, d["beat_times"])

        # 5. GAMEPLAY DIRECTOR
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
        if len(beat_times) > 1:
            avg_beat = np.mean(np.diff(beat_times[:10]))
            
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
        
        # State
        lane_free_time = [0.0] * lanes 
        current_budget = 0.0 
        last_note_time = -999
        lane_jack_count = [0] * lanes 
        last_lane_used = -1
        
        # Mapping
        melodic = [x["midi"] for x in notes if x["source"] != "rhythm"]
        if not melodic: melodic = [60]
        p_min, p_max = np.percentile(melodic, 5), np.percentile(melodic, 95)
        spread = max(1, p_max - p_min)
        last_lane_overall = lanes // 2
        
        notes.sort(key=lambda x: x["time"])

        for n in notes:
            snapped_t = self._weighted_smart_snap(n["time"], d["beat_times"], cfg["grids"])
            n["time"] = (snapped_t * 0.95) + (n["time"] * 0.05)
            
            # Density Check
            dt = n["time"] - last_note_time
            if dt > 0: current_budget = max(0.0, current_budget - (dt * cfg["finger_regen"]))
            
            if n["time"] != last_note_time:
                if dt < cfg["min_dist"]:
                    if final_notes and n["score"] > final_notes[-1]["score"] * 1.5: final_notes.pop()
                    else: continue
                
                cost = 1.0
                if current_budget + cost > cfg["soft_fingers"]:
                    if current_budget + cost > cfg["max_fingers"]: continue
                    if n["score"] < 0.6: continue
            
            # Holds
            n["dur"] = 0.0
            if n["raw_strength"] > 0.85 and n["source"] != "rhythm":
                n["dur"] = 0.4
            
            # Mapping
            if n["source"] == "rhythm":
                ideal_lane = lanes // 2
                if n["score"] > 0.7: ideal_lane = 0 if last_lane_overall >= lanes//2 else lanes-1
            else:
                midi = n["midi"]
                while midi < 48: midi += 12
                while midi > 84: midi -= 12
                pitch_norm = (midi - p_min) / spread
                ideal_lane = int(max(0, min(1, pitch_norm)) * (lanes - 1))

            # Physics
            final_lane = -1
            search_offsets = [0, 1, -1, 2, -2]
            for offset in search_offsets:
                candidate = ideal_lane + offset
                if 0 <= candidate < lanes:
                    if n["time"] < lane_free_time[candidate]: continue
                    
                    is_crowded = False
                    if final_notes and abs(final_notes[-1]["time"] - n["time"]) < 0.01:
                        prev_lane = final_notes[-1]["lane"]
                        if abs(candidate - prev_lane) <= 1: is_crowded = True
                    if is_crowded: continue

                    is_jack = (candidate == last_lane_used) and (n["time"] - last_note_time < JACK_MIN_DELAY)
                    if is_jack and lane_jack_count[candidate] >= MAX_CONSECUTIVE_JACKS:
                        continue 
                    
                    final_lane = candidate
                    break
            
            if final_lane == -1: continue
            
            n["lane"] = final_lane
            
            if final_lane == last_lane_used and (n["time"] - last_note_time < JACK_MIN_DELAY):
                lane_jack_count[final_lane] += 1
            else:
                lane_jack_count = [0] * lanes
                lane_jack_count[final_lane] = 1
                
            last_lane_used = final_lane
            last_lane_overall = final_lane
            lane_free_time[final_lane] = n["time"] + n["dur"] + 0.05 
            
            if n["time"] != last_note_time: 
                current_budget += 1.0
                last_note_time = n["time"]
            
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