import numpy as np
import librosa
import scipy.signal

# ==========================================
#       V63: CONFIGURATION
# ==========================================

# --- AUDIO ANALYSIS ---
ANALYSIS_FMIN = librosa.note_to_hz('C1')  # Minimum pitch frequency to track (range: C1-C3, lower = more bass notes)
ANALYSIS_OCTAVES = 6  # Octave range for pitch detection (range: 4-7, higher = wider pitch range)
MIN_VOLUME_THRESHOLD = 0.15  # Minimum volume to consider a note (range: 0.05-0.3, lower = more sensitive)

# --- HARMONIC/PERCUSSIVE SEPARATION ---
# Higher = Stricter separation. 
# Stricter Harmonic = Removes more noise/drums, but might lose fast notes.
# Stricter Percussive = Removes more tone, keeps only sharp clicks.
HPSS_CONFIG = {
    "vocal": 3.5,  # HIGH: Aggressively remove breath/sibilance to fix "jittery" pitch
    "other": 3.0,  # MED-HIGH: Clean up synth/piano chords
    "bass":  1.5,  # LOW: Bass needs body; too strict kills the fundamental freq
    "drums": 3.0   # HIGH (Percussive): Used to isolate sharp hits from cymbal wash
}
VOCAL_SMOOTH_FACTOR = 5  # Median filter for vocal envelope smoothing (range: 3-9, odd numbers only, higher = smoother)

# --- PITCH TRACKING THRESHOLDS ---
PITCH_THRESHOLD_DEFAULT = 0.05  # Magnitude threshold for pitch detection (range: 0.01-0.15, lower = more notes)
PITCH_THRESHOLD_BASS = 0.05  # Bass-specific pitch threshold (range: 0.01-0.15)
CANDIDATE_MAG_THRESHOLD = 0.05  # Minimum magnitude for valid pitch candidate (range: 0.01-0.2) 

# --- DYNAMIC MIXING ---
DIFF_WEIGHTS = {
    "EASY":   { "vocal": 1.35, "other": 1.15, "drums": 0.50, "bass": 0.40 },
    "NORMAL": { "vocal": 1.25, "other": 1.15, "drums": 0.65, "bass": 0.60 },
    "HARD":   { "vocal": 1.10, "other": 1.00, "drums": 0.80, "bass": 0.75 },
    "INSANE": { "vocal": 1.05, "other": 1.00, "drums": 0.80, "bass": 0.80 }
}

# --- PER-INSTRUMENT GATING (New in V63) ---
# Minimum score required to spawn a note.
# Prevents background bass/drums from flooding Easy mode.
SCORE_GATES = {
    "EASY":   { "vocal": 0.15, "other": 0.20, "drums": 0.45, "bass": 0.45 },
    "NORMAL": { "vocal": 0.15, "other": 0.15, "drums": 0.35, "bass": 0.35 },
    "HARD":   { "vocal": 0.10, "other": 0.10, "drums": 0.20, "bass": 0.20 },
    "INSANE": { "vocal": 0.05, "other": 0.05, "drums": 0.15, "bass": 0.15 }
}

# --- SOURCE SWITCHING BEHAVIOR ---
COOLDOWN_BEATS = 0.65  # Beats to wait before switching sources (range: 0.5-2.0, higher = less switching)
SWITCH_PENALTY = 0.75  # Score multiplier when switching sources (range: 0.5-0.9, lower = discourages switching)
HYSTERESIS_FINAL_GATE = 0.15  # Final score threshold after hysteresis (range: 0.05-0.3, higher = fewer notes)

# --- CHORD DETECTION ---
CHORD_TIME_WINDOW = 0.06  # Max time difference to group notes as chord (range: 0.03-0.1 seconds)
CHORD_SCORE_RATIO = 0.7  # Min score ratio for second note in chord (range: 0.5-0.9, higher = stricter chords)

# --- RHYTHM QUANTIZATION ---
GRID_PENALTIES = { 4: 1.0, 8: 1.0, 12: 2.0, 16: 1.2, 24: 2.5, 32: 3.0 }  # Snap preference (higher = less likely)
SNAP_BLEND_RATIO = 0.95  # How much to trust snapped time vs original (range: 0.8-1.0, 1.0 = full snap)
MAX_SNAP_ERROR = 0.05  # Max deviation before rejecting snap (range: 0.03-0.1 seconds)

# --- PATTERN LIMITS ---
MAX_CONSECUTIVE_JACKS = 3  # Max same-lane hits in a row (range: 2-5, higher = more jacks allowed)
JACK_MIN_DELAY = 0.35  # Min time between jacks in seconds (range: 0.3-0.8, lower = faster jacks)

# --- GAMEPLAY DENSITY CONTROL ---
MIN_NOTE_OVERRIDE_RATIO = 1.5  # Score ratio to override previous note (range: 1.2-2.0, higher = harder to override)
NON_MELODIC_BUDGET_THRESHOLD = 0.6  # Min score for drums/bass when over budget (range: 0.4-0.8)
CHORD_SAME_TIME_WINDOW = 0.01  # Max time to consider notes simultaneous (range: 0.005-0.02 seconds)
LANE_CROWDING_WINDOW = 0.01  # Time window to check lane crowding (range: 0.005-0.02 seconds)

# --- HOLD NOTE DURATIONS ---
BASS_HOLD_THRESHOLD = 0.6  # Min strength for bass hold notes (range: 0.4-0.8, higher = fewer holds)
BASS_HOLD_DURATION = 0.5  # Length of bass holds in seconds (range: 0.3-0.8)
VOCAL_HOLD_THRESHOLD = 0.8  # Min strength for vocal hold notes (range: 0.6-0.9, higher = fewer holds)
VOCAL_HOLD_DURATION = 0.4  # Length of vocal holds in seconds (range: 0.2-0.6)
HOLD_LANE_BUFFER = 0.05  # Extra time after hold before lane is free (range: 0.03-0.1 seconds)

# --- ONSET DETECTION ---
ONSET_PRE_MAX = 3   # Frames before peak for onset detection (range: 2-5)
ONSET_POST_MAX = 3  # Frames after peak for onset detection (range: 2-5)
ONSET_PRE_AVG = 3   # Frames before for averaging (range: 2-5)
ONSET_POST_AVG = 3  # Frames after for averaging (range: 2-5)
DRUM_ONSET_DELTA = 0.2  # Drum-specific onset sensitivity (range: 0.05-0.2, lower = more sensitive)
DRUM_ONSET_WAIT = 2      # Min frames between drum onsets (range: 1-3, must be Integer)

# --- PITCH-TO-LANE MAPPING ---
# Replaced static edge threshold with flow-based intensity threshold
DRUM_STRIKE_THRESHOLD = 0.7  # Score to force a "Jump" pattern vs a "Stream" pattern (range: 0.5-0.9)
BASS_PITCH_MIN = 36      # MIDI note for bass range start (range: 28-40)
BASS_PITCH_RANGE = 24    # MIDI semitones for bass spread (range: 12-36)
MELODIC_PITCH_MIN = 48   # MIDI note floor for melody (range: 40-52)
MELODIC_PITCH_MAX = 84   # MIDI note ceiling for melody (range: 80-96)

# --- DIFFICULTY CONFIGS ---
DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "grids": [4], "chords": False, "min_dist": 0.40, "soft_fingers": 1.5, "max_fingers": 2.0, "finger_regen": 3.0 },
    "NORMAL": { "lanes": 4, "grids": [4, 8], "chords": False, "min_dist": 0.25, "soft_fingers": 2.0, "max_fingers": 2.5, "finger_regen": 5.0 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 12, 16], "chords": True, "min_dist": 0.15, "soft_fingers": 2.0, "max_fingers": 3.0, "finger_regen": 7.0 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], "chords": True, "min_dist": 0.10, "soft_fingers": 2.0, "max_fingers": 4.0, "finger_regen": 9.0 },
}

SENSITIVITY = {
    "EASY":   { "delta": 0.06, "wait": 1 }, 
    "NORMAL": { "delta": 0.05, "wait": 1 },
    "HARD":   { "delta": 0.04, "wait": 1 },
    "INSANE": { "delta": 0.03, "wait": 1 },
}

# ==========================================

class MapGenerator:
    def __init__(self, stems_path, is_instrumental=False):
        print("[GEN] Loading stems...")
        self.sr = 44100
        self.is_instrumental = is_instrumental
        
        self.y_voc, _ = librosa.load(stems_path["vocals"], sr=self.sr)
        self.y_oth, _ = librosa.load(stems_path["other"], sr=self.sr)
        self.y_bass, _ = librosa.load(stems_path["bass"], sr=self.sr)
        self.y_drum, _ = librosa.load(stems_path["drums"], sr=self.sr)
        
        # --- PRE-PROCESSING ---
        self.env_voc = self._get_env(self.y_voc, smooth_factor=VOCAL_SMOOTH_FACTOR) 
        self.env_oth = self._get_env(self.y_oth)
        self.env_bass = self._get_env(self.y_bass)

        # --- SEPARATION ---
        print("[GEN] Separating Harmonics...")
        
        # Vocals: Strict harmonic separation to stabilize pitch
        self.y_voc_harm, _ = librosa.effects.hpss(self.y_voc, margin=HPSS_CONFIG["vocal"])
        
        # Other: Standard separation
        self.y_oth_harm, _ = librosa.effects.hpss(self.y_oth, margin=HPSS_CONFIG["other"])
        
        # Bass: Gentle separation to preserve low-end energy
        self.y_bass_harm, _ = librosa.effects.hpss(self.y_bass, margin=HPSS_CONFIG["bass"])
        
        # Drums: We want the PERCUSSIVE part, not the harmonic part
        # This removes cymbal 'hum' and reverb, leaving only sharp transients for the grid
        _, self.y_drum_perc = librosa.effects.hpss(self.y_drum, margin=HPSS_CONFIG["drums"])
        self.env_drum = self._get_env(self.y_drum_perc) # Update env to use clean percussion

        self.data = self._analyze()

    def _get_env(self, y, smooth_factor=0):
        env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if env.max() > 0: env /= env.max()
        
        # Optional Smoothing for jagged tracks (Vocals)
        if smooth_factor > 0:
            env = scipy.signal.medfilt(env, kernel_size=smooth_factor)
            
        return env

    def _analyze(self):
        print("[GEN] Extracting Rhythm Grid...")
        onset_env = librosa.onset.onset_strength(y=self.y_drum, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        print("[GEN] Pitch Tracking...")
        if not self.is_instrumental:
            pitch_voc, mag_voc = librosa.piptrack(y=self.y_voc_harm, sr=self.sr, fmin=ANALYSIS_FMIN, threshold=PITCH_THRESHOLD_DEFAULT)
        else:
            pitch_voc, mag_voc = None, None
            
        pitch_oth, mag_oth = librosa.piptrack(y=self.y_oth_harm, sr=self.sr, fmin=ANALYSIS_FMIN, threshold=PITCH_THRESHOLD_DEFAULT)
        pitch_bass, mag_bass = librosa.piptrack(y=self.y_bass_harm, sr=self.sr, fmin=librosa.note_to_hz('A0'), threshold=PITCH_THRESHOLD_BASS)

        return {
            "beat_times": beat_times,
            "pitch_voc": pitch_voc, "mag_voc": mag_voc,
            "pitch_oth": pitch_oth, "mag_oth": mag_oth,
            "pitch_bass": pitch_bass, "mag_bass": mag_bass,
            "duration": librosa.get_duration(y=self.y_drum, sr=self.sr)
        }

    def _get_candidates(self, env, pitch_grid, mag_grid, source_name, diff_name, weights, gates):
        sens = SENSITIVITY.get(diff_name, SENSITIVITY["NORMAL"])
        onset_frames = librosa.util.peak_pick(env, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                               pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                               delta=sens["delta"], wait=sens["wait"])
        
        candidates = []
        n_frames = mag_grid.shape[1] if mag_grid is not None else 0

        for t in onset_frames:
            # V63: Use Difficulty-Specific Gating
            score_gate = gates.get(source_name, 0.15)
            
            # Weighted Score Calculation
            weight = weights.get(source_name, 1.0)
            score = env[t] * weight
            
            # Gate Check
            if score < score_gate: continue

            midi = 0
            if pitch_grid is not None:
                t_start = max(0, t-1)
                t_end = min(n_frames, t+3)
                local_mags = mag_grid[:, t_start:t_end]
                if local_mags.size > 0:
                    best_idx = np.unravel_index(np.argmax(local_mags), local_mags.shape)
                    if local_mags[best_idx] > CANDIDATE_MAG_THRESHOLD:
                        freq = pitch_grid[best_idx[0], t_start + best_idx[1]]
                        if freq > 0: midi = librosa.hz_to_midi(freq)
            
            if midi == 0:
                if source_name == "drums": midi = 36 
                elif source_name == "bass": midi = 36 
                else: continue 
            
            candidates.append({
                "time": librosa.frames_to_time(t, sr=self.sr),
                "midi": int(round(midi)),
                "score": score,
                "source": source_name,
                "raw_strength": env[t]
            })
        return candidates

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        weights = DIFF_WEIGHTS.get(diff_name, DIFF_WEIGHTS["NORMAL"])
        gates = SCORE_GATES.get(diff_name, SCORE_GATES["NORMAL"])
        
        # 1. EXTRACT
        vocs = [] if self.is_instrumental else self._get_candidates(self.env_voc, d["pitch_voc"], d["mag_voc"], "vocal", diff_name, weights, gates)
        oth = self._get_candidates(self.env_oth, d["pitch_oth"], d["mag_oth"], "other", diff_name, weights, gates)
        bass = self._get_candidates(self.env_bass, d["pitch_bass"], d["mag_bass"], "bass", diff_name, weights, gates)
        
        drum_env = librosa.util.peak_pick(self.env_drum, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                          pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                          delta=DRUM_ONSET_DELTA, wait=DRUM_ONSET_WAIT)
        drums = []
        drum_gate = gates.get("drums", 0.2)
        for t in drum_env:
            score = self.env_drum[t] * weights["drums"]
            if score >= drum_gate:
                drums.append({ 
                    "time": librosa.frames_to_time(t, sr=self.sr), "midi": 36, 
                    "score": score, 
                    "source": "drums", "raw_strength": self.env_drum[t] 
                })

        # 2. MERGE
        all_notes = vocs + oth + bass + drums
        all_notes.sort(key=lambda x: x["time"])
        
        merged = []
        if all_notes:
            curr = all_notes[0]
            chord_buffer = [curr]
            
            for next_n in all_notes[1:]:
                if next_n["time"] - curr["time"] < CHORD_TIME_WINDOW:
                    chord_buffer.append(next_n)
                else:
                    self._process_chord_buffer(merged, chord_buffer, cfg["chords"])
                    chord_buffer = [next_n]
                    curr = next_n
            self._process_chord_buffer(merged, chord_buffer, cfg["chords"])

        # 3. HYSTERESIS
        stable_notes = self._apply_hysteresis(merged, d["beat_times"])

        # 4. GAMEPLAY
        final_notes = self._apply_gameplay_rules(stable_notes, d, cfg)
        return final_notes

    def _process_chord_buffer(self, output_list, buffer, allow_chords):
        buffer.sort(key=lambda x: x["score"], reverse=True)
        winner = buffer[0]
        output_list.append(winner)
        if allow_chords and len(buffer) > 1:
            loser = buffer[1]
            if loser["score"] > winner["score"] * CHORD_SCORE_RATIO:
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
            if n["score"] > HYSTERESIS_FINAL_GATE: filtered.append(n)
        return filtered

    def _apply_gameplay_rules(self, notes, d, cfg):
        final_notes = []
        lanes = cfg["lanes"]
        lane_free_time = [0.0] * lanes 
        current_budget = 0.0 
        last_note_time = -999
        lane_jack_count = [0] * lanes 
        last_lane_used = -1
        
        # Track general flow direction
        last_lane_overall = lanes // 2
        center_line = lanes / 2.0
        
        melodic = [x["midi"] for x in notes if x["source"] not in ["drums", "bass"]]
        if not melodic: melodic = [60]
        p_min, p_max = np.percentile(melodic, 5), np.percentile(melodic, 95)
        spread = max(1, p_max - p_min)

        notes.sort(key=lambda x: x["time"])

        for n in notes:
            snapped_t = self._weighted_smart_snap(n["time"], d["beat_times"], cfg["grids"])
            n["time"] = (snapped_t * SNAP_BLEND_RATIO) + (n["time"] * (1.0 - SNAP_BLEND_RATIO))
            
            dt = n["time"] - last_note_time
            if dt > 0: current_budget = max(0.0, current_budget - (dt * cfg["finger_regen"]))
            
            if n["time"] != last_note_time:
                if dt < cfg["min_dist"]:
                    if final_notes and n["score"] > final_notes[-1]["score"] * MIN_NOTE_OVERRIDE_RATIO: final_notes.pop()
                    else: continue
                cost = 1.0
                if current_budget + cost > cfg["soft_fingers"]:
                    if current_budget + cost > cfg["max_fingers"]: continue 
                    is_melodic = n["source"] in ["vocal", "other"]
                    if not is_melodic and n["score"] < NON_MELODIC_BUDGET_THRESHOLD: continue 
            
            n["dur"] = 0.0
            if n["source"] == "bass" and n["raw_strength"] > BASS_HOLD_THRESHOLD: n["dur"] = BASS_HOLD_DURATION
            elif n["source"] == "vocal" and n["raw_strength"] > VOCAL_HOLD_THRESHOLD: n["dur"] = VOCAL_HOLD_DURATION
            
            # --- START NEW LANE LOGIC ---
            ideal_lane = 0
            
            if n["source"] == "drums":
                # Check which side of the track we were last on
                is_left_side = last_lane_overall < center_line
                
                if n["score"] > DRUM_STRIKE_THRESHOLD:
                    # HARD STRIKE: Force a wide jump to the opposite outer edge
                    # If we were Left, jump to Far Right. If Right, jump to Far Left.
                    ideal_lane = (lanes - 1) if is_left_side else 0
                else:
                    # SOFT STRIKE: subtle alternation to the opposite inner lane
                    # 4 Lanes: Left(0,1) -> 2. Right(2,3) -> 1.
                    ideal_lane = int(center_line) if is_left_side else int(center_line) - 1
            
            elif n["source"] == "bass":
                pitch_norm = (n["midi"] - BASS_PITCH_MIN) / BASS_PITCH_RANGE
                ideal_lane = int(max(0, min(1, pitch_norm)) * (lanes - 1))
            else:
                midi = n["midi"]
                while midi < MELODIC_PITCH_MIN: midi += 12
                while midi > MELODIC_PITCH_MAX: midi -= 12
                pitch_norm = (midi - p_min) / spread
                ideal_lane = int(max(0, min(1, pitch_norm)) * (lanes - 1))
            # --- END NEW LANE LOGIC ---

            final_lane = -1
            search_offsets = [0, 1, -1, 2, -2]
            for offset in search_offsets:
                candidate = ideal_lane + offset
                if 0 <= candidate < lanes:
                    if n["time"] < lane_free_time[candidate]: continue
                    is_crowded = False
                    if final_notes and abs(final_notes[-1]["time"] - n["time"]) < CHORD_SAME_TIME_WINDOW:
                        prev_lane = final_notes[-1]["lane"]
                        if abs(candidate - prev_lane) <= 1: is_crowded = True
                    if is_crowded: continue
                    is_jack = (candidate == last_lane_used) and (n["time"] - last_note_time < JACK_MIN_DELAY)
                    if is_jack and lane_jack_count[candidate] >= MAX_CONSECUTIVE_JACKS: continue 
                    final_lane = candidate
                    break
            
            if final_lane == -1: continue
            n["lane"] = final_lane
            
            if final_lane == last_lane_used and (n["time"] - last_note_time < JACK_MIN_DELAY): lane_jack_count[final_lane] += 1
            else: 
                lane_jack_count = [0] * lanes
                lane_jack_count[final_lane] = 1
            
            last_lane_used = final_lane
            # Update overall flow tracker
            last_lane_overall = final_lane
            
            lane_free_time[final_lane] = n["time"] + n["dur"] + HOLD_LANE_BUFFER
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
        if min_weighted_error > MAX_SNAP_ERROR: return t
        return best_time