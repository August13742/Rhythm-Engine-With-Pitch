import numpy as np
import librosa
import scipy.signal

# --- CONFIG ---
ANALYSIS_FMIN = librosa.note_to_hz('C2') 
ANALYSIS_OCTAVES = 6

# WEIGHTS
WEIGHT_VOCAL = 1.0 
WEIGHT_OTHER = 0.9 

# THRESHOLDS
VOCAL_PITCH_THRESHOLD = 0.05  # If pitch is weaker than this, it's breath/noise -> Ignore
MIN_VOCAL_DUR = 0.15          # Vocals don't trill as fast as pianos. Debounce them harder.

DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "min_dist": 0.25, "grid": 4 },
    "NORMAL": { "lanes": 4, "min_dist": 0.18, "grid": 8 },
    "HARD":   { "lanes": 5, "min_dist": 0.12, "grid": 12 },
    "INSANE": { "lanes": 6, "min_dist": 0.08, "grid": 16 },
}

class MapGenerator:
    def __init__(self, vocals_path, other_path, rhythm_path):
        print("[GEN] Loading stems...")
        self.sr = 44100
        self.y_voc, _ = librosa.load(vocals_path, sr=self.sr)
        self.y_oth, _ = librosa.load(other_path, sr=self.sr)
        self.y_rhy, _ = librosa.load(rhythm_path, sr=self.sr)
        
        # --- NEW IN V38: SIBILANCE REMOVAL ---
        # Split Vocals into Harmonic (Vowels) and Percussive (Consonants/Breath)
        # margin=3.0 pushes more ambiguous sound into Percussive, leaving only pure Tone in Harmonic
        print("[GEN] Filtering Vocal Sibilance (HPSS)...")
        self.y_voc_harm, self.y_voc_perc = librosa.effects.hpss(self.y_voc, margin=3.0)
        
        self.data = self._analyze()

    def _analyze(self):
        # 1. RHYTHM (For Grid only)
        print("[GEN] Extracting Rhythm Grid...")
        onset_rhy = librosa.onset.onset_strength(y=self.y_rhy, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_rhy, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        # 2. MELODY ENVELOPES
        # NOTE: We use y_voc_harm (The Vowels) for detection, ignoring breath noise
        print("[GEN] Analyzing Melody Energy...")
        onset_voc = librosa.onset.onset_strength(y=self.y_voc_harm, sr=self.sr)
        onset_oth = librosa.onset.onset_strength(y=self.y_oth, sr=self.sr)

        # Normalize
        if onset_voc.max() > 0: onset_voc /= onset_voc.max()
        if onset_oth.max() > 0: onset_oth /= onset_oth.max()

        # 3. PITCH DATA
        # We run CQT on the Clean Vowels to get stable pitch
        print("[GEN] Extracting Pitch...")
        cqt_voc = np.abs(librosa.cqt(self.y_voc_harm, sr=self.sr, fmin=ANALYSIS_FMIN, 
                                   n_bins=12*ANALYSIS_OCTAVES, bins_per_octave=12))
        
        cqt_oth = np.abs(librosa.cqt(self.y_oth, sr=self.sr, fmin=ANALYSIS_FMIN, 
                                   n_bins=12*ANALYSIS_OCTAVES, bins_per_octave=12))

        return {
            "onset_voc": onset_voc,
            "onset_oth": onset_oth,
            "beat_times": beat_times,
            "cqt_voc": cqt_voc,
            "cqt_oth": cqt_oth,
            "duration": librosa.get_duration(y=self.y_rhy, sr=self.sr)
        }

    def _get_pitch_and_confidence(self, frame, source_type):
        d = self.data
        source_cqt = d["cqt_voc"] if source_type == "vocal" else d["cqt_oth"]
        
        if frame >= source_cqt.shape[1]: frame = source_cqt.shape[1] - 1
        
        # Window averaging for stability
        window = source_cqt[:, max(0, frame-1):min(source_cqt.shape[1], frame+2)]
        avg_col = np.mean(window, axis=1)
        
        # Find Peak
        midi_idx = np.argmax(avg_col)
        max_val = avg_col[midi_idx]
        
        # Calculate Confidence (Peak strength relative to average noise floor)
        # If the spectrum is flat (white noise/breath), mean is close to max.
        noise_floor = np.mean(avg_col)
        confidence = (max_val - noise_floor) / (max_val + 1e-6)
        
        midi = 36 + midi_idx 
        return max(40, min(100, midi)), confidence

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        
        # --- TRIGGER LOGIC ---
        # Combine envelopes
        master_onset = (d["onset_voc"] * WEIGHT_VOCAL) + (d["onset_oth"] * WEIGHT_OTHER)
        master_onset = scipy.signal.medfilt(master_onset, kernel_size=5) # Heavy smoothing to fix "Shadow Singer" wobble

        onset_frames = librosa.util.peak_pick(master_onset, 
                                            pre_max=5, post_max=5, # Wider window to ignore double-track flutters 
                                            pre_avg=5, post_avg=5, 
                                            delta=0.04, wait=5)
        
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        candidates = []
        
        for i, frame in enumerate(onset_frames):
            # SOURCE DECISION
            v_str = d["onset_voc"][frame]
            o_str = d["onset_oth"][frame]
            source_type = "vocal" if (v_str * WEIGHT_VOCAL) > (o_str * WEIGHT_OTHER) else "other"
            
            # PITCH & CONFIDENCE CHECK
            midi, confidence = self._get_pitch_and_confidence(frame, source_type)
            
            # --- THE FILTER: REJECT GARBAGE VOCALS ---
            # If it's a vocal note but the pitch is "muddy" (breath), skip it.
            if source_type == "vocal" and confidence < VOCAL_PITCH_THRESHOLD:
                continue

            # TIMING
            t = onset_times[i]
            snapped_t = self._snap_time(t, d["beat_times"], cfg["grid"])
            final_t = (snapped_t * 0.85) + (t * 0.15) 

            candidates.append({
                "time": final_t,
                "midi": midi,
                "strength": master_onset[frame],
                "source": source_type
            })

        # --- SORT & FILTER ---
        candidates.sort(key=lambda x: x["time"])
        filtered = []
        last_t = -999
        last_source = ""
        
        for c in candidates:
            dt = c["time"] - last_t
            
            # DYNAMIC DEBOUNCE
            # Vocals need more time to breathe than pianos.
            required_dist = cfg["min_dist"]
            if c["source"] == "vocal" or last_source == "vocal":
                required_dist = max(required_dist, MIN_VOCAL_DUR)

            if dt < required_dist:
                # Contest: Keep the louder note
                if filtered and c["strength"] > filtered[-1]["strength"] * 1.2:
                    filtered.pop() # Remove previous weak note
                    # Fall through to append this one
                else:
                    continue # Skip this one
            
            filtered.append(c)
            last_t = c["time"]
            last_source = c["source"]

        # --- MAPPING ---
        notes = []
        lanes = cfg["lanes"]
        
        midis = [x["midi"] for x in filtered] if filtered else [60]
        p_min, p_max = np.percentile(midis, 10), np.percentile(midis, 90)
        spread = max(1, p_max - p_min)
        last_lane = lanes // 2
        
        for c in filtered:
            pitch_norm = (c["midi"] - p_min) / spread
            pitch_norm = max(0, min(1, pitch_norm))
            
            ideal_lane = int(pitch_norm * (lanes - 1))
            
            if abs(ideal_lane - last_lane) > 2:
                ideal_lane = last_lane + (2 if ideal_lane > last_lane else -2)
            
            final_lane = max(0, min(lanes-1, ideal_lane))
            
            notes.append({
                "time": round(c["time"], 3),
                "lane": final_lane,
                "midi": c["midi"],
                "vol": 1.1,
                "dur": 0,
                "source": c["source"]
            })
            last_lane = final_lane

        return notes

    def _snap_time(self, t, beats, subdivisions):
        if len(beats) < 2: return t
        idx = (np.abs(beats - t)).argmin()
        closest_beat = beats[idx]
        
        if idx < len(beats) - 1:
            beat_dur = beats[idx+1] - beats[idx]
        else:
            beat_dur = beats[idx] - beats[idx-1]
            
        grid_dur = beat_dur / (subdivisions / 4)
        diff = t - closest_beat
        return closest_beat + round(diff / grid_dur) * grid_dur

if __name__ == "__main__":
    pass