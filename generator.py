import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import json
import os
from numba import jit
from functools import lru_cache

# ==========================================
#        V88: THE DIRECTOR CONFIGURATION
# ==========================================

# --- HARMONIC/PERCUSSIVE SEPARATION ---
HPSS_CONFIG = {
    "vocal": 1.5,  # LOW: Critical for capturing 'T', 'K', 'S' sounds
    "other": 3.0,  # High: Clean harmonic separation for melody
    "bass":  1.5,  
    "drums": 3.5   
}
VOCAL_SMOOTH_FACTOR = 3 # Snappy vocals

# --- THRESHOLDS (RELAXED) ---
# We lower these so the Director has more candidates to choose from.
# We filter later based on Context, not just raw volume.
SCORE_GATES = {
    "EASY":   { "vocal": 0.10, "other": 0.15, "drums": 0.30, "bass": 0.30 },
    "NORMAL": { "vocal": 0.08, "other": 0.12, "drums": 0.25, "bass": 0.25 },
    "HARD":   { "vocal": 0.05, "other": 0.10, "drums": 0.20, "bass": 0.20 },
    "INSANE": { "vocal": 0.02, "other": 0.05, "drums": 0.15, "bass": 0.15 }
}

# --- DIRECTOR LOGIC ---
DIRECTOR_WINDOW_BEATS = 2.0  # Analyze energy every 2 beats (Half-Bar)
VOCAL_SENSITIVITY = 1.2      # Multiplier for Vocal RMS comparison (Higher = Vocals take lead easier)
MELODY_SENSITIVITY = 1.1     # Multiplier for Melody RMS comparison

# --- ANCHOR LOGIC ---
# When Vocals are leading, we still want Drums on strong beats (Anchors).
ANCHOR_VOLUME_THRESHOLD = 0.4 # Drums must be this loud to be considered Anchors

# --- RHYTHM QUANTIZATION ---
# Strict Grids: We remove 1/12, 1/24, 1/32 for lower difficulties to clean up the chart.
DIFF_CONFIGS = {
    # snap_threshold: Max seconds off-grid allowed. If exceeded, DELETE note.
    "EASY":   { "lanes": 4, "grids": [4], "chords": False, "finger_regen": 3.0, "snap_threshold": 0.05 },
    "NORMAL": { "lanes": 4, "grids": [4, 8], "chords": True, "finger_regen": 5.0, "snap_threshold": 0.08 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 12, 16], "chords": True, "finger_regen": 7.0, "snap_threshold": 0.10 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], "chords": True, "finger_regen": 8.0, "snap_threshold": 1.0 },
}

# (Keep other constants like ANALYSIS_FMIN, PITCH_THRESHOLDS from previous script)
ANALYSIS_FMIN = librosa.note_to_hz('C1')
PITCH_THRESHOLD_DEFAULT = 0.05
PITCH_THRESHOLD_BASS = 0.05
CANDIDATE_MAG_THRESHOLD = 0.05
POWER_CHORD_THRESHOLD = 0.85
ONSET_PRE_MAX = 3
ONSET_POST_MAX = 3
ONSET_PRE_AVG = 3
ONSET_POST_AVG = 3
DRUM_ONSET_DELTA = 0.2
DRUM_ONSET_WAIT = 2
MAX_CONSECUTIVE_JACKS = 3
JACK_MIN_DELAY = 0.5
HOLD_LANE_BUFFER = 0.1

# ==========================================

class MapGenerator:
    def __init__(self, stems_path, is_instrumental=False):
        print("[GEN] Loading stems...")
        self.sr = 44100
        self.stems_path = stems_path
        
        # Load Audio
        self.y_voc, _ = librosa.load(stems_path["vocals"], sr=self.sr)
        self.y_oth, _ = librosa.load(stems_path["other"], sr=self.sr)
        self.y_bass, _ = librosa.load(stems_path["bass"], sr=self.sr)
        self.y_drum, _ = librosa.load(stems_path["drums"], sr=self.sr)
            
        self.raw_stems = {
            "vocal": self.y_voc, "other": self.y_oth, 
            "bass": self.y_bass, "drums": self.y_drum
        }
  
        # --- SEPARATION ---
        print("[GEN] Separating Harmonics (V88 Low-Margin)...")
        # Use fixed low margin for vocals to capture articulation
        self.y_voc_harm, _ = librosa.effects.hpss(self.y_voc, margin=HPSS_CONFIG["vocal"])
        self.y_bass_harm, _ = librosa.effects.hpss(self.y_bass, margin=HPSS_CONFIG["bass"])
        _, self.y_drum_perc = librosa.effects.hpss(self.y_drum, margin=HPSS_CONFIG["drums"])
        self.y_oth_harm, self.y_oth_perc = librosa.effects.hpss(self.y_oth, margin=HPSS_CONFIG["other"])
        
        # --- ENVELOPES ---
        self.env_drum = self._get_env(self.y_drum_perc) 
        self.env_voc = self._get_env(self.y_voc_harm, smooth_factor=VOCAL_SMOOTH_FACTOR) 
        self.env_bass = self._get_env(self.y_bass_harm)
        
        # Melody Envelope (Flux + Attack)
        S_oth = librosa.stft(self.y_oth_harm)
        env_oth_flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(np.abs(S_oth), ref=np.max), sr=self.sr)
        if env_oth_flux.max() > 0: env_oth_flux /= env_oth_flux.max()
        self.env_oth = env_oth_flux # Simplified for V88

        self.data = self._analyze()

    @lru_cache(maxsize=32)
    def _get_env_cached(self, y_hash, smooth_factor=0):
        """Cache wrapper for envelope computation"""
        return self._get_env_impl(y_hash, smooth_factor)
    
    def _get_env(self, y, smooth_factor=0):
        """Compute onset envelope with optional caching"""
        # Try to use cache if array is hashable via bytes
        try:
            y_hash = hash(y.tobytes())
            return self._get_env_cached(y_hash, smooth_factor)
        except:
            # Fallback to direct computation if caching fails
            return self._get_env_impl(y, smooth_factor)
    
    def _get_env_impl(self, y, smooth_factor=0):
        """Actual envelope computation"""
        env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if env.max() > 0: env /= env.max()
        if smooth_factor > 0:
            env = scipy.signal.medfilt(env, kernel_size=smooth_factor)
        return env

    def _analyze(self):
        print("[GEN] Analyzing Rhythm & Pitch...")
        onset_env = librosa.onset.onset_strength(y=self.y_drum, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        # Pitch Tracking (Always track vocals now)
        pitch_voc, mag_voc = librosa.piptrack(y=self.y_voc_harm, sr=self.sr, fmin=ANALYSIS_FMIN, threshold=PITCH_THRESHOLD_DEFAULT)
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

    def _get_candidates(self, env, pitch_grid, mag_grid, source_name, diff_name, gates, weights={}):
        """
        Standard candidate extraction. Returns ALL valid onsets.
        We do NOT delete based on 'finger budget' here. We only check audio validity.
        """
        candidates = []
        
        # BPM-BASED SMOOTHING: Calculate debounce based on tempo
        tempo = 120
        if len(self.data["beat_times"]) > 1:
            tempo = 60.0 / np.mean(np.diff(self.data["beat_times"]))
        
        # Limit spam to max 16th notes
        # Fast songs (180bpm) = 0.08s, Slow songs (60bpm) = 0.25s
        sixteenth_note_dur = 60.0 / tempo / 4.0
        
        # 1. FIX VOCAL SPAM: Source-Specific Sensitivity + BPM-aware
        # Vocals need a larger 'wait' to merge vibrato/flutter
        wait = 1
        delta = 0.03
        if source_name == "vocal": 
            wait = int(max(6, sixteenth_note_dur * self.sr / 512))  # ~16th note duration
            delta = 0.05
        elif source_name == "drums": 
            delta = DRUM_ONSET_DELTA
        else:  # Bass & Other
            # Apply tempo-aware debounce
            wait = int(max(1, sixteenth_note_dur * self.sr / 512))
        
        onset_frames = librosa.util.peak_pick(env, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                              pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                              delta=delta, wait=wait)
        
        gate = gates.get(source_name, 0.1)
        
        for t in onset_frames:
            score = env[t]
            if score < gate: continue
            
            # Simple Pitch Lookup
            midi = 0
            if pitch_grid is not None:
                # Look at immediate frame for pitch
                mags = mag_grid[:, t]
                idx = np.argmax(mags)
                if mags[idx] > CANDIDATE_MAG_THRESHOLD:
                    freq = pitch_grid[idx, t]
                    if freq > 0: midi = librosa.hz_to_midi(freq)
            
            if midi == 0:
                if source_name in ["drums", "bass"]: midi = 36
                else: continue
            
            # 2. FIX HOWLING: Octave Folding
            # Force MIDI into a sane range (e.g., 48 [C3] to 84 [C6])
            if midi > 0 and source_name in ["vocal", "other"]:
                while midi > 84: midi -= 12
                while midi < 48: midi += 12
            elif midi > 0 and source_name == "bass":
                while midi > 60: midi -= 12 # Keep bass low
            
            candidates.append({
                "time": librosa.frames_to_time(t, sr=self.sr),
                "midi": int(round(midi)),
                "score": score,
                "source": source_name,
                "dur": 0.0
            })
            
        return candidates

    def _enforce_pitch_continuity(self, notes):
        """
        Detects unnatural large pitch jumps (>12 semitones) and corrects octave errors.
        Keeps notes close to the previous note in the same source.
        """
        if not notes: return []
        
        # Sort by time
        notes.sort(key=lambda x: x["time"])
        
        # Track the "running average" pitch of the melodic line
        last_pitch = -1
        last_source = ""
        
        for n in notes:
            if n["source"] not in ["vocal", "other"]: continue
            
            current_midi = n["midi"]
            
            # If we have a history for this source within reasonable time
            if last_pitch > 0 and n["source"] == last_source:
                diff = abs(current_midi - last_pitch)
                
                # If the jump is unnatural (> 12 semitones), it's likely an octave error
                if diff > 12:
                    # Force it down/up octaves until it's close to the last note
                    while current_midi > last_pitch + 6: current_midi -= 12
                    while current_midi < last_pitch - 6: current_midi += 12
                    n["midi"] = current_midi
                    
            last_pitch = n["midi"]
            last_source = n["source"]
            
        return notes

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        gates = SCORE_GATES.get(diff_name, SCORE_GATES["NORMAL"])
        
        # 1. HARVEST (Get everything with low gates)
        print(f"[GEN] Harvesting candidates for {diff_name}...")
        
        # We process vocals regardless of ratio. Cheap operation.
        vocs = self._get_candidates(self.env_voc, d["pitch_voc"], d["mag_voc"], "vocal", diff_name, gates)
        oth  = self._get_candidates(self.env_oth, d["pitch_oth"], d["mag_oth"], "other", diff_name, gates)
        bass = self._get_candidates(self.env_bass, d["pitch_bass"], d["mag_bass"], "bass", diff_name, gates)
        drums = self._get_candidates(self.env_drum, None, None, "drums", diff_name, gates)
        
        # Organize into a dictionary for the Director
        pool = { "vocal": vocs, "other": oth, "bass": bass, "drums": drums }

        # 2. THE DIRECTOR (Compose the chart)
        print("[GEN] The Director is composing...")
        structured_notes = self._compose_chart(pool, d["beat_times"], cfg, diff_name)

        # 2.5. REFINE PITCHES (pYIN)
        print("[GEN] Refining pitches with pYIN...")
        structured_notes = self._refine_pitches_robust(structured_notes)
        
        # 2.6. ENFORCE PITCH CONTINUITY
        print("[GEN] Enforcing pitch continuity...")
        structured_notes = self._enforce_pitch_continuity(structured_notes)
        
        # 3. FINALIZE (Quantize & Lane)
        print("[GEN] Finalizing...")
        final_notes = self._finalize_chart(structured_notes, d["beat_times"], cfg, diff_name)
        
        # 4. SAVE
        self._save_beatmap(final_notes, diff_name)
        return final_notes

    def _compose_chart(self, pool, beat_times, cfg, diff_name="NORMAL"):
        """
        The Core V89 Logic with Mixer Permissions:
        Iterates through the song in chunks, decides a 'Leader',
        and layers instruments based on difficulty-aware permissions.
        """
        timeline = []
        
        # Difficulty Permissions
        # WHO is allowed to speak when someone else is leading?
        allow_secondary = False  # Allow melody when vocals are singing?
        allow_tertiary  = False  # Allow drums when melody is playing?
        
        if diff_name in ["HARD"]:
            allow_secondary = True 
        elif diff_name in ["INSANE"]:
            allow_secondary = True
            allow_tertiary = True
        
        # Define Chunks (e.g. 2 beats)
        chunk_duration = 0.5 # Default fallback
        if len(beat_times) > 1:
            chunk_duration = np.mean(np.diff(beat_times)) * DIRECTOR_WINDOW_BEATS
            
        # Iterate through song in chunks
        max_time = self.data["duration"]
        cursor = 0.0
        
        while cursor < max_time:
            end_t = cursor + chunk_duration
            
            # --- A. DECIDE LEADER ---
            # Extract RMS for this chunk
            rms_voc = self._get_rms(self.y_voc, cursor, end_t)
            rms_oth = self._get_rms(self.y_oth, cursor, end_t)
            rms_drum = self._get_rms(self.y_drum, cursor, end_t)
            
            # The Logic
            leader = "DRUMS" # Default
            
            # 1. Vocal Priority
            # If vocals are present AND significant relative to drums
            if rms_voc > 0.02 and rms_voc > (rms_drum * 0.4) * VOCAL_SENSITIVITY:
                leader = "VOCAL"
            # 2. Melody Priority
            elif rms_oth > 0.02 and rms_oth > (rms_drum * 0.8) * MELODY_SENSITIVITY:
                leader = "OTHER"
            
            # --- B. LAYER ASSEMBLY (with Mixer Permissions) ---
            
            # Helper to get notes in this window
            def get_in_window(source_name):
                return [n for n in pool[source_name] if cursor <= n["time"] < end_t]
            
            chunk_notes = []
            
            if leader == "VOCAL":
                # 1. Primary Layer
                vocals = get_in_window("vocal")
                chunk_notes.extend(vocals)
                
                # 2. THE VOID CHECK (Fix for humming/silence sections)
                # If vocals are leading but empty, force Melody/Drums to take over
                if len(vocals) == 0:
                    chunk_notes.extend(get_in_window("other")) # Fill with melody
                    chunk_notes.extend(get_in_window("drums")) # Fill with drums
                else:
                    # Normal behavior: vocals are present
                    if allow_secondary:  # Hard/Insane gets melody backing
                        chunk_notes.extend(get_in_window("other"))
                    
                    # Always get Anchors (Kicks)
                    dr = get_in_window("drums")
                    anchors = [d for d in dr if d["score"] > ANCHOR_VOLUME_THRESHOLD]
                    chunk_notes.extend(anchors)

            elif leader == "OTHER":
                # Primary
                chunk_notes.extend(get_in_window("other"))
                
                if allow_tertiary:  # Insane gets full drums backing
                    chunk_notes.extend(get_in_window("drums"))
                else:
                    # Anchors only
                    dr = get_in_window("drums")
                    anchors = [d for d in dr if d["score"] > ANCHOR_VOLUME_THRESHOLD]
                    chunk_notes.extend(anchors)
                    
            else: # LEADER == DRUMS
                # Primary: All Drums
                chunk_notes.extend(get_in_window("drums"))
                
                # Support: Bass (Groove)
                chunk_notes.extend(get_in_window("bass"))
                
                # Support: Melody (Background) - Only if strong
                mel = get_in_window("other")
                strong_mel = [m for m in mel if m["score"] > 0.2] # higher gate
                chunk_notes.extend(strong_mel)

            timeline.extend(chunk_notes)
            cursor = end_t
            
        return timeline

    def _get_rms(self, y, start_t, end_t):
        """Compute RMS with JIT-optimized helper"""
        start_sample = int(start_t * self.sr)
        end_sample = int(end_t * self.sr)
        if start_sample >= len(y): return 0.0
        chunk = y[start_sample:end_sample]
        if len(chunk) == 0: return 0.0
        return _compute_rms_jit(chunk)
    
    def _resolve_collisions_optimized(self, slots, allow_chords):
        """Optimized collision resolution"""
        clean_notes = []
        sorted_times = sorted(slots.keys())
        
        for t in sorted_times:
            candidates = slots[t]
            # Sort by Score (Strongest first)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Winner takes the slot
            winner = candidates[0]
            stack = [winner]
            
            # Check for Chords (if allowed)
            if allow_chords and len(candidates) > 1:
                second = candidates[1]
                # Logic: Don't chord 2 vocals. But Vocal + Drum is OK.
                if winner["source"] == "vocal" and second["source"] == "vocal":
                    pass
                # Only chord if second note is reasonably strong relative to winner
                elif second["score"] > (winner["score"] * 0.5):
                    stack.append(second)
            
            clean_notes.extend(stack)
        
        return clean_notes

    def _finalize_chart(self, notes, beat_times, cfg, diff_name="NORMAL"):
        """
        Quantize -> Resolve Collisions -> Lane Alloc
        """
        # 1. QUANTIZE & FILTER (Grid Rejection)
        slots = {}
        allowed_grids = cfg["grids"]
        threshold = cfg.get("snap_threshold", 0.1)
        
        for n in notes:
            # Try to snap
            snapped_t, success = self._weighted_smart_snap(n["time"], beat_times, allowed_grids, threshold)
            
            # If snap failed (too far off grid for this difficulty), DELETE NOTE
            if not success:
                continue 
            
            n["time"] = snapped_t
            
            # Key by time (rounded)
            t_key = round(snapped_t, 3)
            if t_key not in slots: slots[t_key] = []
            slots[t_key].append(n)
            
        # 2. RESOLVE COLLISIONS (Stacking)
        clean_notes = self._resolve_collisions_optimized(
            slots, cfg["chords"]
        )
            
        # 3. LANE ALLOCATION
        # Simple lane distribution based on pitch
        final_notes = []
        lanes = cfg["lanes"]
        last_lane = -1
        last_time = -999
        
        # Pitch ranges
        melodic_min, melodic_max = 48, 84
        bass_min, bass_max = 36, 60
        
        for n in clean_notes:
            # Determine ideal lane
            ideal = 0
            if n["source"] == "drums":
                # Kicks (low) center, Snares/Hats sides
                # Simple logic for now: toggle based on score or time
                ideal = int(lanes / 2) if n["score"] > 0.6 else 0
            elif n["source"] == "bass":
                norm = _normalize_pitch_jit(n["midi"], bass_min, bass_max)
                ideal = int(norm * (lanes - 1))
            else: # Vocal/Other
                norm = _normalize_pitch_jit(n["midi"], melodic_min, melodic_max)
                ideal = int(norm * (lanes - 1))
            
            # Avoid Jackhammering (Same lane too fast)
            if ideal == last_lane and (n["time"] - last_time) < JACK_MIN_DELAY:
                # Move it
                ideal = (ideal + 1) % lanes
            
            n["lane"] = ideal
            last_lane = ideal
            last_time = n["time"]
            final_notes.append(n)
            
        return final_notes

    def _weighted_smart_snap(self, t, beats, allowed_grids, max_error_sec):
        """
        Returns: (best_time, success_boolean)
        """
        if len(beats) < 2: return t, True
        idx = (np.abs(beats - t)).argmin()
        closest_beat = beats[idx]
        
        if idx < len(beats)-1: beat_dur = beats[idx+1] - beats[idx]
        else: beat_dur = beats[idx] - beats[idx-1]
        
        best_time = t
        min_error = float('inf')
        
        for div in allowed_grids:
            step = beat_dur / (div / 4) 
            ticks = round((t - closest_beat) / step)
            candidate = closest_beat + (ticks * step)
            
            # Calculate absolute time error
            error = abs(candidate - t)
            
            # We still weight it to prefer simple grids
            penalty = 1.0
            if div >= 16: penalty = 1.2
            if div >= 24: penalty = 1.5
            
            weighted_error = error * penalty
            
            if weighted_error < min_error:
                min_error = weighted_error
                best_time = candidate
        
        # DECIMATION LOGIC:
        # If the best grid point is too far away, this note is "off-rhythm"
        # for this difficulty level. Return False to delete it.
        if min_error > max_error_sec:
            return t, False
            
        return best_time, True

    def _save_beatmap(self, notes, diff_name):
        vocals_path = self.stems_path["vocals"]
        base_dir = os.path.dirname(vocals_path)
        base_name = os.path.basename(base_dir)
        output_file = os.path.join(base_dir, f"{base_name}_{diff_name}.json")
        
        serializable_notes = []
        for note in notes:
            new_note = note.copy()
            for k, v in new_note.items():
                if isinstance(v, (np.integer, np.int64, np.int32)): new_note[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)): new_note[k] = float(v)
            serializable_notes.append(new_note)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_notes, f, indent=2)
        print(f"[GEN] Saved {diff_name} beatmap to {output_file}")


# ==========================================
#        JIT-OPTIMIZED HELPER FUNCTIONS
# ==========================================

@jit(nopython=True, cache=True)
def _compute_rms_jit(chunk):
    """JIT-compiled RMS computation for performance"""
    return np.sqrt(np.mean(chunk**2))

@jit(nopython=True, cache=True)
def _normalize_pitch_jit(midi, melodic_min, melodic_max):
    """JIT-compiled pitch normalization"""
    norm = (midi - melodic_min) / (melodic_max - melodic_min)
    return max(0.0, min(1.0, norm))

@jit(nopython=True, cache=True)
def _compute_weighted_error_jit(error, div):
    """JIT-compiled weighted error calculation for grid snapping"""
    penalty = 1.0
    if div >= 16:
        penalty = 1.5
    if div >= 24:
        penalty = 2.5
    return error * penalty