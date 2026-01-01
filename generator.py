import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import json
import os

# ==========================================
#       V70: CONFIGURATION (FINAL POLISH)
# ==========================================

# --- AUDIO ANALYSIS ---
ANALYSIS_FMIN = librosa.note_to_hz('C1')  # Minimum pitch frequency to track (range: C1-C3, lower = more bass notes)
ANALYSIS_OCTAVES = 6  # Octave range for pitch detection (range: 4-7, higher = wider pitch range)
MIN_VOLUME_THRESHOLD = 0.1  # Minimum volume to consider a note (range: 0.05-0.3, lower = more sensitive)

# --- HARMONIC/PERCUSSIVE SEPARATION ---
# V70: These are now BASE values. They scale dynamically.
# Higher = Stricter separation. 
# Stricter Harmonic = Removes more noise/drums, but might lose fast notes.
# Stricter Percussive = Removes more tone, keeps only sharp clicks.
HPSS_CONFIG = {
    "vocal": 3.0,  # Lowered slightly, dynamic logic handles the rest
    "other": 3.0,  # MED-HIGH: Clean up synth/piano chords
    "bass":  1.5,  # LOW: Bass needs body; too strict kills the fundamental freq
    "drums": 3.5   # HIGH (Percussive): Used to isolate sharp hits from cymbal wash
}
VOCAL_SMOOTH_FACTOR = 5  # Lowered for snappier vocals (range: 3-9, odd numbers only, higher = smoother)

# --- PITCH TRACKING THRESHOLDS ---
PITCH_THRESHOLD_DEFAULT = 0.05  # Magnitude threshold for pitch detection (range: 0.01-0.15, lower = more notes)
PITCH_THRESHOLD_BASS = 0.05  # Bass-specific pitch threshold (range: 0.01-0.15)
CANDIDATE_MAG_THRESHOLD = 0.05  # Minimum magnitude for valid pitch candidate (range: 0.01-0.2) 

# --- DYNAMIC MIXING ---
DIFF_WEIGHTS = {
    "EASY":   { "vocal": 1.33, "other": 1.15, "drums": 0.50, "bass": 0.40 },
    "NORMAL": { "vocal": 1.20, "other": 1.15, "drums": 0.65, "bass": 0.60 },
    "HARD":   { "vocal": 1.20, "other": 1.00, "drums": 0.80, "bass": 0.75 },
    "INSANE": { "vocal": 1.20, "other": 1.00, "drums": 0.80, "bass": 0.80 }
}

# --- PER-INSTRUMENT GATING (V64: ADAPTIVE BASE) ---
# These are now the "Floor" gates. 
# The actual gate will be: BASE + (TRACK_MEAN * ADAPTIVE_BIAS)
SCORE_GATES = {
    "EASY":   { "vocal": 0.15, "other": 0.20, "drums": 0.45, "bass": 0.45 },
    "NORMAL": { "vocal": 0.15, "other": 0.15, "drums": 0.35, "bass": 0.35 },
    "HARD":   { "vocal": 0.10, "other": 0.10, "drums": 0.20, "bass": 0.20 },
    "INSANE": { "vocal": 0.05, "other": 0.05, "drums": 0.15, "bass": 0.15 }
}
ADAPTIVE_GATE_BIAS = 0.4  # How much the average volume raises the detection threshold (0.0-1.0)
MIN_VOCAL_RATIO = 0.1 # minium vocal score required to not render the song as non-vocal
# --- SOURCE SWITCHING BEHAVIOR ---
COOLDOWN_BEATS = 0.70  # Beats to wait before switching sources (range: 0.5-2.0, higher = less switching)
SWITCH_PENALTY = 0.65  # Score multiplier when switching sources (range: 0.5-0.9, lower = discourages switching)
HYSTERESIS_FINAL_GATE = 0.15  # Slightly lowered since we are normalizing properly now

# --- CHORD DETECTION ---
CHORD_TIME_WINDOW = 0.05  # Max time difference to group notes as chord (range: 0.03-0.1 seconds)
CHORD_SCORE_RATIO = 0.75  # Min score ratio for second note in chord (range: 0.5-0.9, higher = stricter chords)

# --- RHYTHM QUANTIZATION ---
GRID_PENALTIES = { 4: 1.0, 8: 1.0, 12: 2.0, 16: 1.2, 24: 2.5, 32: 3.0 }  # Snap preference (higher = less likely)
SNAP_BLEND_RATIO = 0.95  # How much to trust snapped time vs original (range: 0.8-1.0, 1.0 = full snap)
MAX_SNAP_ERROR = 0.05  # Max deviation before rejecting snap (range: 0.03-0.1 seconds)

# --- PATTERN LIMITS ---
MAX_CONSECUTIVE_JACKS = 3  # Max same-lane hits in a row (range: 2-5, higher = more jacks allowed)
JACK_MIN_DELAY = 0.5  # Min time between jacks in seconds (range: 0.3-0.8, lower = faster jacks)

# --- GAMEPLAY DENSITY CONTROL ---
MIN_NOTE_OVERRIDE_RATIO = 1.5  # Score ratio to override previous note (range: 1.2-2.0, higher = harder to override)
CHORD_SAME_TIME_WINDOW = 0.01  # Max time to consider notes simultaneous (range: 0.005-0.02 seconds)

# --- HOLD NOTE DURATIONS ---
BASS_HOLD_THRESHOLD = 0.8  # Min strength for bass hold notes (range: 0.4-0.8, higher = fewer holds)
BASS_HOLD_DURATION = 0.5  # Length of bass holds in seconds (range: 0.3-0.8)
VOCAL_HOLD_THRESHOLD = 0.9  # Min strength for vocal hold notes (range: 0.6-0.9, higher = fewer holds)
VOCAL_HOLD_DURATION = 0.4  # Length of vocal holds in seconds (range: 0.2-0.6)
OTHER_HOLD_THRESHOLD = 0.85  # Min strength for melody holds (range: 0.5-0.9)
OTHER_HOLD_DURATION = 0.5   # Length of melody holds (range: 0.3-0.8)

HOLD_LANE_BUFFER = 0.1  # Extra time after hold before lane is free (range: 0.03-0.1 seconds)

# --- ONSET DETECTION ---
ONSET_PRE_MAX = 3   # Frames before peak for onset detection (range: 2-5)
ONSET_POST_MAX = 3  # Frames after peak for onset detection (range: 2-5)
ONSET_PRE_AVG = 3   # Frames before for averaging (range: 2-5)
ONSET_POST_AVG = 3  # Frames after for averaging (range: 2-5)
DRUM_ONSET_DELTA = 0.2  # Drum-specific onset sensitivity (range: 0.05-0.2, lower = more sensitive)
DRUM_ONSET_WAIT = 2      # Min frames between drum onsets (range: 1-3, must be Integer)

# --- PITCH-TO-LANE MAPPING ---
DRUM_STRIKE_THRESHOLD = 0.7  # Score to force a "Jump" pattern vs a "Stream" pattern (range: 0.5-0.9)
BASS_PITCH_MIN = 36      # MIDI note for bass range start (range: 28-40)
BASS_PITCH_RANGE = 24    # MIDI semitones for bass spread (range: 12-36)
MELODIC_PITCH_MIN = 48   # MIDI note floor for melody (range: 40-52)
MELODIC_PITCH_MAX = 84   # MIDI note ceiling for melody (range: 80-96)

# --- V70: POWER CHORDS ---
# If a note score exceeds this, we double it (Impact Chord)
POWER_CHORD_THRESHOLD = 0.85

# --- DIFFICULTY CONFIGS ---
DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "grids": [4], "chords": False, "min_dist": 0.40, "soft_fingers": 1.5, "max_fingers": 2.0, "finger_regen": 3.0 },
    "NORMAL": { "lanes": 4, "grids": [4, 8], "chords": True, "min_dist": 0.25, "soft_fingers": 2.0, "max_fingers": 2.5, "finger_regen": 5.0 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 12, 16], "chords": True, "min_dist": 0.15, "soft_fingers": 2.0, "max_fingers": 3.0, "finger_regen": 7.0 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], "chords": True, "min_dist": 0.10, "soft_fingers": 2.0, "max_fingers": 3.0, "finger_regen": 8.0 },
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
        self.voc_ratio = 0.0
        self.stems_path = stems_path
        
        self.y_voc, _ = librosa.load(stems_path["vocals"], sr=self.sr)
        self.y_oth, _ = librosa.load(stems_path["other"], sr=self.sr)
        self.y_bass, _ = librosa.load(stems_path["bass"], sr=self.sr)
        self.y_drum, _ = librosa.load(stems_path["drums"], sr=self.sr)
            
        # V83: Map for pYIN lookup during generation
        self.raw_stems = {
            "vocal": self.y_voc,
            "other": self.y_oth,
            "bass": self.y_bass,
            "drums": self.y_drum
        }
        self.stems_path = stems_path # Store for saving JSONs later

  
        # --- V67: AUTOMATIC INSTRUMENTAL DETECTION ---
        rms_voc = np.sqrt(np.mean(self.y_voc**2))
        rms_acc = np.sqrt(np.mean((self.y_drum + self.y_bass + self.y_oth)**2))
        self.voc_ratio = rms_voc / (rms_acc + 1e-6)
        print(f"[GEN] Vocal Energy Ratio: {self.voc_ratio:.4f}")
        
        if self.voc_ratio < MIN_VOCAL_RATIO and not is_instrumental:
            print("[GEN] Low vocal presence detected. Forcing INSTRUMENTAL mode.")
            self.is_instrumental = True

        # --- SEPARATION (V70: DYNAMIC HPSS) ---
        print("[GEN] Separating Harmonics...")
        
        # Dynamic Vocal Margin:
        # If Ratio is high (>0.8), use loose margin (1.5) to keep breath/texture.
        # If Ratio is low (<0.3), use strict margin (3.5) to kill bleed.
        # Linear interpolation between 0.3 and 0.8
        if self.voc_ratio > 0.8: voc_margin = 1.5
        elif self.voc_ratio < 0.3: voc_margin = 3.5
        else:
            # Lerp: 3.5 -> 1.5
            t = (self.voc_ratio - 0.3) / 0.5
            voc_margin = 3.5 - (t * 2.0)
            
        print(f"[GEN] Dynamic Vocal HPSS Margin: {voc_margin:.2f}")
        self.y_voc_harm, _ = librosa.effects.hpss(self.y_voc, margin=voc_margin)
        
        self.y_bass_harm, _ = librosa.effects.hpss(self.y_bass, margin=HPSS_CONFIG["bass"])
        _, self.y_drum_perc = librosa.effects.hpss(self.y_drum, margin=HPSS_CONFIG["drums"])
        
        # --- V68: DUAL-STREAM FOR 'OTHER' ---
        self.y_oth_harm, self.y_oth_perc = librosa.effects.hpss(self.y_oth, margin=HPSS_CONFIG["other"])
        
        # --- ENVELOPE PROCESSING ---
        self.env_drum = self._get_env(self.y_drum_perc) 
        self.env_voc = self._get_env(self.y_voc_harm, smooth_factor=VOCAL_SMOOTH_FACTOR) 
        self.env_bass = self._get_env(self.y_bass_harm)
        
        # SPECIAL HANDLING FOR "OTHER": FLUX + ATTACK
        S_oth = librosa.stft(self.y_oth_harm)
        env_oth_flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(np.abs(S_oth), ref=np.max), sr=self.sr)
        if env_oth_flux.max() > 0: env_oth_flux /= env_oth_flux.max()

        # V70: HI-HAT FILTER for Other Attack
        # Calculate Spectral Centroid. If Attack is high-freq only, it's bleed.
        cent = librosa.feature.spectral_centroid(y=self.y_oth_perc, sr=self.sr)[0]
        cent = np.interp(np.arange(len(env_oth_flux)), np.arange(len(cent)), cent)
        if cent.max() > 0: cent /= cent.max()
        
        env_oth_attack = self._get_env(self.y_oth_perc)
        
        # Suppress attacks where centroid is very high (> 0.7 normalized)
        # This kills hi-hat bleed while keeping guitars (mid-range)
        env_oth_attack = env_oth_attack * (1.0 - (cent * 0.8))

        self.env_oth = (env_oth_flux + env_oth_attack) / 2.0
        if self.env_oth.max() > 0: self.env_oth /= self.env_oth.max()

        self.data = self._analyze()

    def _get_env(self, y, smooth_factor=0):
        env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if env.max() > 0: env /= env.max()
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

    def _refine_pitches_robust(self, notes):
        """
        V83: BAKING STEP - pYIN
        Runs Probabilistic YIN on candidate notes to determine the TRUE fundamental frequency.
        This is slow but accurate.
        """
        # We only refine Melodic instruments
        targets = [n for n in notes if n["source"] in ["vocal", "bass", "other"]]
        count = len(targets)
        if count == 0: return notes

        print(f"      [pYIN] Refining {count} notes...")
        
        for i, n in enumerate(targets):
            # Sampling Window:
            # Shift 40ms forward to skip the attack transient/breath.
            # Analyze 100ms of audio (sufficient for fundamental).
            start_sample = int((n["time"] + 0.04) * self.sr)
            end_sample = start_sample + int(0.10 * self.sr)
            
            stem = self.raw_stems.get(n["source"], self.raw_stems["other"])
            if start_sample >= len(stem): continue
            
            slc = stem[start_sample:end_sample]
            
            # V83 FIX: Increased frame_length to 2048 to handle Bass frequencies (40Hz)
            # 1024 samples @ 44.1kHz is ~23ms, which is too short for 40Hz (25ms period).
            # 2048 samples is ~46ms, covering the low end safely.
            frame_len = 2048
            if len(slc) < frame_len: continue 
            
            # Constraints based on source to prevent Octave Errors
            if n["source"] == "bass":
                fmin, fmax = 40, 400
            elif n["source"] == "vocal":
                fmin, fmax = 100, 1000 
            else:
                fmin, fmax = 80, 2000
                
            try:
                # Run pYIN with larger frame_length
                f0, _, _ = librosa.pyin(slc, fmin=fmin, fmax=fmax, sr=self.sr, frame_length=frame_len)
                
                # Filter NaNs
                valid_f0 = f0[~np.isnan(f0)]
                
                if len(valid_f0) > 0:
                    # Use Median to find stable pitch (ignoring vibrato)
                    pitch_hz = np.median(valid_f0)
                    if pitch_hz > 0:
                        n["midi"] = int(round(librosa.hz_to_midi(pitch_hz)))
            except Exception:
                # Fallback to original pitch if pYIN fails
                continue
        
        return notes
    
    # V69: Added pitch_lookahead to fix "Off-Tune" detection on fast attacks
    def _get_candidates(self, env, pitch_grid, mag_grid, source_name, diff_name, weights, gates, debounce_time=0.0, pitch_lookahead=0):
        sens = SENSITIVITY.get(diff_name, SENSITIVITY["NORMAL"])
        onset_frames = librosa.util.peak_pick(env, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                              pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                              delta=sens["delta"], wait=sens["wait"])
        
        candidates = []
        n_frames = mag_grid.shape[1] if mag_grid is not None else 0
        last_accepted_time = -999.0

        # --- LOCAL CONTRAST GATING ---
        window_size = 86 
        local_floor = scipy.ndimage.uniform_filter1d(env, size=window_size)
        
        for t in onset_frames:
            current_time = librosa.frames_to_time(t, sr=self.sr)
            
            if current_time - last_accepted_time < debounce_time:
                continue

            base_gate = gates.get(source_name, 0.15)
            local_threshold = local_floor[t] + 0.10
            
            score = env[t] * weights.get(source_name, 1.0)
            
            if env[t] < base_gate: continue       
            if env[t] < local_threshold: continue 

            midi = 0
            if pitch_grid is not None:
                # V69 FIX: Shift window forward by 'pitch_lookahead' frames
                # This ensures we sample the 'Tone' (sustain), not the 'Click' (attack)
                search_t = t + pitch_lookahead
                t_start = max(0, search_t - 1)
                t_end = min(n_frames, search_t + 3)
                
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
            
            # V70: Check for Power Chords (Impacts)
            is_impact = False
            if source_name in ["drums", "bass", "vocal"]:
                if score > POWER_CHORD_THRESHOLD:
                    is_impact = True

            note_entry = {
                "time": current_time,
                "midi": int(round(midi)),
                "score": score,
                "source": source_name,
                "raw_strength": env[t],
                "is_impact": is_impact
            }
            candidates.append(note_entry)
            
            last_accepted_time = current_time
            
        return candidates

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        weights = DIFF_WEIGHTS.get(diff_name, DIFF_WEIGHTS["NORMAL"]).copy()
        gates = SCORE_GATES.get(diff_name, SCORE_GATES["NORMAL"])
        
        # --- DYNAMIC VOCAL WEIGHTS (Saturating Model) ---
        if not self.is_instrumental:
            saturated_ratio = np.tanh(self.voc_ratio * 2.0)
            weight_modifier = (saturated_ratio * 0.4) - 0.2
            weights["vocal"] = max(0.5, weights["vocal"] + weight_modifier)
            print(f"[GEN] Dynamic Vocal Weight: {weights['vocal']:.2f}")

        # 1. EXTRACT
        vocs = [] if self.is_instrumental else self._get_candidates(
            self.env_voc, d["pitch_voc"], d["mag_voc"], "vocal", diff_name, weights, gates, 
            debounce_time=0.10, pitch_lookahead=0
        )
        
        oth = self._get_candidates(
            self.env_oth, d["pitch_oth"], d["mag_oth"], "other", diff_name, weights, gates, 
            debounce_time=0.05, pitch_lookahead=2 
        )
        
        bass = self._get_candidates(
            self.env_bass, d["pitch_bass"], d["mag_bass"], "bass", diff_name, weights, gates, 
            debounce_time=0.10, pitch_lookahead=1
        )
        # --- V83: BAKE PITCHES ---
        # Refine pitches using pYIN before merging. 
        # This fixes "off-tune" notes by looking at the sustain body.
        if vocs: self._refine_pitches_robust(vocs)
        if oth: self._refine_pitches_robust(oth)
        if bass: self._refine_pitches_robust(bass)
        
        drum_env = librosa.util.peak_pick(self.env_drum, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                          pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                          delta=DRUM_ONSET_DELTA, wait=DRUM_ONSET_WAIT)
        drums = []
        drum_base_gate = gates.get("drums", 0.2)
        drum_track_mean = np.mean(self.env_drum)
        drum_adaptive_gate = drum_base_gate + (drum_track_mean * ADAPTIVE_GATE_BIAS)

        for t in drum_env:
            score = self.env_drum[t] * weights["drums"]
            if score >= drum_adaptive_gate:
                is_impact = score > POWER_CHORD_THRESHOLD
                drums.append({ 
                    "time": librosa.frames_to_time(t, sr=self.sr), "midi": 36, 
                    "score": score, 
                    "source": "drums", "raw_strength": self.env_drum[t],
                    "is_impact": is_impact
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
        
        # 5. SAVE TO FILE
        self._save_beatmap(final_notes, diff_name)
        
        return final_notes
    
    def _save_beatmap(self, notes, diff_name):
        """Save generated beatmap to JSON file"""
        vocals_path = self.stems_path["vocals"]
        base_dir = os.path.dirname(vocals_path)
        base_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"{base_name}_{diff_name}.json")
        
        # Serialization helper
        serializable_notes = []
        for note in notes:
            new_note = note.copy()
            # Ensure types are JSON compliant
            for k, v in new_note.items():
                if isinstance(v, (np.integer, np.int64, np.int32)): 
                    new_note[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)): 
                    new_note[k] = float(v)
                elif isinstance(v, (bool, np.bool_)): 
                    new_note[k] = bool(v)
            serializable_notes.append(new_note)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_notes, f, indent=2)
        print(f"[GEN] Saved {diff_name} beatmap to {output_file}")
        
        
    def _process_chord_buffer(self, output_list, buffer, allow_chords):
        buffer.sort(key=lambda x: x["score"], reverse=True)
        winner = buffer[0]
        output_list.append(winner)
        
        # V70: Allow "Impact Duplication"
        # If the winner is an IMPACT note (Power Chord), we manually add a duplicate
        # so the Gameplay processor sees two notes and tries to lane them differently.
        # This bypasses the "ghost filter" because it's intentional.
        if allow_chords and winner.get("is_impact", False):
            # Create a shallow copy
            dup = winner.copy()
            # Slight offset so it sorts correctly but treated as simultaneous
            dup["time"] += 0.001 
            # Mark duplicate so we don't infinitely recurse if we ran this again
            dup["is_impact"] = False 
            output_list.append(dup)
            return # Impact takes priority, no other notes in this chord

        if allow_chords and len(buffer) > 1:
            for loser in buffer[1:]:
                midi_diff = abs(winner["midi"] - loser["midi"])
                if midi_diff % 12 == 0:
                    continue 
                
                if loser["score"] > winner["score"] * CHORD_SCORE_RATIO:
                    loser["time"] = winner["time"]
                    output_list.append(loser)
                    break 

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
        
        last_lane_overall = lanes // 2
        center_line = lanes / 2.0
        
        active_holds = []

        melodic = [x["midi"] for x in notes if x["source"] not in ["drums", "bass"]]
        if not melodic: melodic = [60]
        p_min, p_max = np.percentile(melodic, 5), np.percentile(melodic, 95)
        spread = max(1, p_max - p_min)

        notes.sort(key=lambda x: x["time"])

        for n in notes:
            snapped_t = self._weighted_smart_snap(n["time"], d["beat_times"], cfg["grids"])
            n["time"] = (snapped_t * SNAP_BLEND_RATIO) + (n["time"] * (1.0 - SNAP_BLEND_RATIO))
            
            dt = n["time"] - last_note_time
            if dt > 0:
                current_budget = max(0.0, current_budget - (dt * cfg["finger_regen"]))
            
            active_holds = [end_t for end_t in active_holds if end_t > n["time"]]
            physical_load = current_budget + len(active_holds)

            if n["time"] != last_note_time:
                # Basic distance check
                if dt < cfg["min_dist"]:
                    if final_notes and n["score"] > final_notes[-1]["score"] * MIN_NOTE_OVERRIDE_RATIO:
                        popped = final_notes.pop()
                        if popped["dur"] > 0:
                            active_holds = active_holds[:-1] 
                    # V70: Allow Chords (Distance near zero)
                    elif dt < CHORD_SAME_TIME_WINDOW and cfg["chords"]:
                        pass # Allow it to proceed to allocation
                    else: 
                        continue

                if physical_load + 1.0 > cfg["max_fingers"]:
                    continue
            
            n["dur"] = 0.0
            if n["source"] == "bass" and n["raw_strength"] > BASS_HOLD_THRESHOLD: 
                n["dur"] = BASS_HOLD_DURATION
            elif n["source"] == "vocal" and n["raw_strength"] > VOCAL_HOLD_THRESHOLD: 
                n["dur"] = VOCAL_HOLD_DURATION
            elif n["source"] == "other" and n["raw_strength"] > OTHER_HOLD_THRESHOLD: 
                n["dur"] = OTHER_HOLD_DURATION
            
            ideal_lane = 0
            if n["source"] == "drums":
                is_left_side = last_lane_overall < center_line
                if n["score"] > DRUM_STRIKE_THRESHOLD:
                    ideal_lane = (lanes - 1) if is_left_side else 0
                else:
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

            final_lane = -1
            # V70: Randomize search direction for chords to prevent "Right-leaning" stacks
            search_offsets = [0, 1, -1, 2, -2]
            
            # If we are effectively simultaneous with the last note, ensure we don't pick the same lane
            if final_notes and abs(final_notes[-1]["time"] - n["time"]) < CHORD_SAME_TIME_WINDOW:
                # Try to pick a lane far from the previous note
                occupied = final_notes[-1]["lane"]
                if occupied == ideal_lane: 
                    # If ideal is taken, push outward
                    search_offsets = [1, -1, 2, -2, 3, -3] if occupied < center_line else [-1, 1, -2, 2, -3, 3]

            for offset in search_offsets:
                candidate = ideal_lane + offset
                if 0 <= candidate < lanes:
                    if n["time"] < lane_free_time[candidate]: 
                        continue
                    if final_notes and abs(final_notes[-1]["time"] - n["time"]) < CHORD_SAME_TIME_WINDOW:
                        if candidate == final_notes[-1]["lane"]: # Absolute collision
                            continue
                    if (candidate == last_lane_used) and (n["time"] - last_note_time < JACK_MIN_DELAY):
                        if lane_jack_count[candidate] >= MAX_CONSECUTIVE_JACKS: 
                            continue 
                    final_lane = candidate
                    break
            
            if final_lane == -1: 
                continue

            n["lane"] = final_lane
            if n["dur"] > 0:
                active_holds.append(n["time"] + n["dur"])
            
            lane_free_time[final_lane] = n["time"] + n["dur"] + HOLD_LANE_BUFFER
            
            if n["time"] != last_note_time: 
                current_budget += 1.0
                last_note_time = n["time"]
            
            last_lane_used = final_lane
            last_lane_overall = final_lane
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

