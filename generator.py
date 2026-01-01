import numpy as np
import librosa
import scipy.signal

# --- CONFIG ---
ANALYSIS_FMIN = librosa.note_to_hz('C2') 
ANALYSIS_OCTAVES = 6
WEIGHT_VOCAL = 1.0 
WEIGHT_OTHER = 0.9 

# Allow very short slices for fast sections, but cap long ones
MIN_SLICE_DUR = 0.1 
MAX_SLICE_DUR = 2.0 

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
        
        # HPSS for cleaner detection (same as V38)
        self.y_voc_harm, self.y_voc_perc = librosa.effects.hpss(self.y_voc, margin=3.0)
        self.data = self._analyze()

    def _analyze(self):
        onset_rhy = librosa.onset.onset_strength(y=self.y_rhy, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_rhy, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        onset_voc = librosa.onset.onset_strength(y=self.y_voc_harm, sr=self.sr)
        onset_oth = librosa.onset.onset_strength(y=self.y_oth, sr=self.sr)

        if onset_voc.max() > 0: onset_voc /= onset_voc.max()
        if onset_oth.max() > 0: onset_oth /= onset_oth.max()

        # We still need pitch for LANE PLACEMENT, even if we don't use it for audio
        cqt_voc = np.abs(librosa.cqt(self.y_voc_harm, sr=self.sr, fmin=ANALYSIS_FMIN, 
                                   n_bins=12*ANALYSIS_OCTAVES, bins_per_octave=12))
        cqt_oth = np.abs(librosa.cqt(self.y_oth, sr=self.sr, fmin=ANALYSIS_FMIN, 
                                   n_bins=12*ANALYSIS_OCTAVES, bins_per_octave=12))

        return {
            "onset_voc": onset_voc, "onset_oth": onset_oth,
            "beat_times": beat_times,
            "cqt_voc": cqt_voc, "cqt_oth": cqt_oth,
            "duration": librosa.get_duration(y=self.y_rhy, sr=self.sr)
        }

    def _get_pitch(self, frame, source_type):
        d = self.data
        source_cqt = d["cqt_voc"] if source_type == "vocal" else d["cqt_oth"]
        if frame >= source_cqt.shape[1]: frame = source_cqt.shape[1] - 1
        
        window = source_cqt[:, max(0, frame-1):min(source_cqt.shape[1], frame+2)]
        avg_col = np.mean(window, axis=1)
        midi = 36 + np.argmax(avg_col)
        return max(40, min(100, midi))

    def generate(self, diff_name):
        cfg = DIFF_CONFIGS[diff_name]
        d = self.data
        
        # Trigger Logic
        master_onset = (d["onset_voc"] * WEIGHT_VOCAL) + (d["onset_oth"] * WEIGHT_OTHER)
        master_onset = scipy.signal.medfilt(master_onset, kernel_size=5)
        onset_frames = librosa.util.peak_pick(master_onset, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.04, wait=5)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        candidates = []
        for i, frame in enumerate(onset_frames):
            v_str = d["onset_voc"][frame]
            o_str = d["onset_oth"][frame]
            source_type = "vocal" if (v_str * WEIGHT_VOCAL) > (o_str * WEIGHT_OTHER) else "other"
            
            midi = self._get_pitch(frame, source_type)
            t = onset_times[i]
            snapped_t = self._snap_time(t, d["beat_times"], cfg["grid"])
            final_t = (snapped_t * 0.85) + (t * 0.15) 

            candidates.append({
                "time": final_t, "midi": midi, "strength": master_onset[frame], "source": source_type
            })

        candidates.sort(key=lambda x: x["time"])
        filtered = []
        last_t = -999
        
        # Filter Logic
        for c in candidates:
            dt = c["time"] - last_t
            if dt < cfg["min_dist"]:
                if filtered and c["strength"] > filtered[-1]["strength"] * 1.2:
                    filtered.pop()
                else: continue
            filtered.append(c)
            last_t = c["time"]

        # --- KEYSOUND DURATION CALCULATION ---
        # Calculate how long each audio slice should be
        notes = []
        lanes = cfg["lanes"]
        
        midis = [x["midi"] for x in filtered] if filtered else [60]
        p_min, p_max = np.percentile(midis, 10), np.percentile(midis, 90)
        spread = max(1, p_max - p_min)
        last_lane = lanes // 2
        
        for i, c in enumerate(filtered):
            # Calculate duration until next note (or max limit)
            if i < len(filtered) - 1:
                dur = filtered[i+1]["time"] - c["time"]
            else:
                dur = 1.0 # Last note
            
            # Clamp duration for cleaner slicing
            eff_dur = max(MIN_SLICE_DUR, min(MAX_SLICE_DUR, dur))
            
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
                "dur": round(eff_dur, 3), # Tell Visualizer how much audio to grab
                "source": c["source"]
            })
            last_lane = final_lane

        return notes

    def _snap_time(self, t, beats, subdivisions):
        if len(beats) < 2: return t
        idx = (np.abs(beats - t)).argmin()
        closest_beat = beats[idx]
        grid_dur = (beats[1]-beats[0])/(subdivisions/4) if len(beats)>1 else 0.1
        return closest_beat + round((t-closest_beat)/grid_dur)*grid_dur