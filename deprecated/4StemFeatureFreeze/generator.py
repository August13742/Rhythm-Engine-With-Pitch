import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import json
import os
import argparse
from numba import jit
from functools import lru_cache

# ==========================================
#        V99: "THE CLEANUP" CONFIG
# ==========================================

# ==========================================

HPSS_CONFIG = { "vocal": 2.0, "other": 3.0, "bass": 1.5, "drums": 3.5 }
VOCAL_SMOOTH_FACTOR = 1

GLOBAL_DENSITY_CAP = {
    "EASY":   1.5,
    "NORMAL": 3.5,
    "HARD":   5.5,
    "INSANE": 7.5 
}

STEM_PRIORITIES = {
    "vocal": 1.25, 
    "other": 1.1, 
    "bass":  0.8, 
    "drums": 1.0   
}

CROSS_STEM_DEBOUNCE = 0.06 

STEM_VOLUMES = {
    "vocal": 1.0,
    "other": 0.75,
    "bass":  0.8,
    "drums": 0.7 
}

# --- PITCH & VISUAL ---
RANGE_VOCAL = (45, 84) 
RANGE_OTHER = (48, 96) 
RANGE_BASS  = (36, 60)

VISUAL_RANGE_VOCAL = (48, 84)
VISUAL_RANGE_OTHER = (48, 84)
VISUAL_RANGE_BASS  = (36, 60) 

DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "grids": [4], "chords": False, "snap_threshold": 0.10 },
    "NORMAL": { "lanes": 4, "grids": [4, 8], "chords": True, "snap_threshold": 0.08 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 12, 16], "chords": True, "snap_threshold": 0.05 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24], "chords": True, "snap_threshold": 1.0 },
}

# Constants
PYIN_FRAME_LENGTH = 4096 
ONSET_PRE_MAX = 3
ONSET_POST_MAX = 3
ONSET_PRE_AVG = 3
ONSET_POST_AVG = 3
DRUM_ONSET_DELTA = 0.2

# NEW: Consolidation Thresholds (in seconds)
# If a note is within this window of a previous note (same lane), merge them.
HOLD_MERGE_WINDOW = 0.25 

# ==========================================

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[GEN] Loading stems (Holds: {use_holds})...")
        self.sr = 44100
        self.stems_path = stems_path
        self.use_holds = use_holds # Store user preference
        
        self.y_voc, _ = librosa.load(stems_path["vocals"], sr=self.sr)
        self.y_oth, _ = librosa.load(stems_path["other"], sr=self.sr)
        self.y_bass, _ = librosa.load(stems_path["bass"], sr=self.sr)
        self.y_drum, _ = librosa.load(stems_path["drums"], sr=self.sr)
            
        # FIX 1: VOCAL BLEED REMOVAL
        # Kill "Ghost Vocals" before we even process harmonics
        print("[GEN] Cleaning Vocal Bleed...")
        self.y_voc = self._clean_vocal_bleed(self.y_voc, self.y_oth + self.y_bass + self.y_drum)

        print("[GEN] Separating Harmonics...")
        self.y_voc_harm, _ = librosa.effects.hpss(self.y_voc, margin=HPSS_CONFIG["vocal"])
        self.y_bass_harm, _ = librosa.effects.hpss(self.y_bass, margin=HPSS_CONFIG["bass"])
        _, self.y_drum_perc = librosa.effects.hpss(self.y_drum, margin=HPSS_CONFIG["drums"])
        self.y_oth_harm, self.y_oth_perc = librosa.effects.hpss(self.y_oth, margin=HPSS_CONFIG["other"])
        
        self.env_drum = self._get_env(self.y_drum_perc) 
        self.env_voc = self._get_env(self.y_voc_harm, smooth_factor=VOCAL_SMOOTH_FACTOR) 
        self.env_bass = self._get_env(self.y_bass_harm)
        
        S_oth = librosa.stft(self.y_oth_harm)
        env_oth_flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(np.abs(S_oth), ref=np.max), sr=self.sr)
        if env_oth_flux.max() > 0: env_oth_flux /= env_oth_flux.max()
        self.env_oth = env_oth_flux

        self.data = self._analyze_rhythm_composite()
        
        raw_pool = self._harvest_all()
        self.master_pool = self._score_with_priority(raw_pool, self.data["beat_times"])
        print(f"[GEN] Master Pool Ready: {len(self.master_pool)} notes")

    def _clean_vocal_bleed(self, y_voc, y_backing):
        """
        Demucs Hallucination Killer.
        If Vocal RMS is low AND Backing RMS is high -> Silence the Vocal.
        """
        frame_len = 2048
        hop_len = 512
        
        rms_voc = librosa.feature.rms(y=y_voc, frame_length=frame_len, hop_length=hop_len)[0]
        rms_back = librosa.feature.rms(y=y_backing, frame_length=frame_len, hop_length=hop_len)[0]
        
        # Soft Gate Logic
        # If vocal is 10x quieter than backing, and absolutely quiet (<0.01), kill it.
        mask = np.ones_like(rms_voc)
        
        # Vectorized condition
        # 1. Backing is loud enough to cause bleed (> 0.05)
        # 2. Vocal is suspiciously quiet relative to backing (< 10%)
        bleed_indices = (rms_back > 0.05) & (rms_voc < (rms_back * 0.15))
        
        # Also kill absolute silence floor to reduce noise
        silence_indices = (rms_voc < 0.005)
        
        mask[bleed_indices] = 0.0
        mask[silence_indices] = 0.0
        
        # Smooth the mask to prevent clicking
        mask = scipy.signal.medfilt(mask, kernel_size=5)
        
        # Interpolate mask back to sample rate
        mask_upsampled = scipy.ndimage.zoom(mask, len(y_voc) / len(mask), order=1)
        
        # Ensure lengths match exactly
        if len(mask_upsampled) < len(y_voc):
            mask_upsampled = np.pad(mask_upsampled, (0, len(y_voc) - len(mask_upsampled)))
        elif len(mask_upsampled) > len(y_voc):
            mask_upsampled = mask_upsampled[:len(y_voc)]
            
        return y_voc * mask_upsampled

    # ... (Rest of init and harvest methods same as V98) ...
    @lru_cache(maxsize=32)
    def _get_env_cached(self, y_hash, smooth_factor=0):
        return self._get_env_impl(y_hash, smooth_factor)
    
    def _get_env(self, y, smooth_factor=0):
        try:
            y_hash = hash(y.tobytes())
            return self._get_env_cached(y_hash, smooth_factor)
        except:
            return self._get_env_impl(y, smooth_factor)
    
    def _get_env_impl(self, y, smooth_factor=0):
        env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if env.max() > 0: env /= env.max()
        if smooth_factor > 0:
            env = scipy.signal.medfilt(env, kernel_size=smooth_factor)
        return env

    def _analyze_rhythm_composite(self):
        print("[GEN] Analyzing Composite Rhythm...")
        y_composite = self.y_drum + self.y_bass + (self.y_oth * 0.5)
        onset_env = librosa.onset.onset_strength(y=y_composite, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        bpm = 120
        if len(beat_times) > 1:
            bpm = 60.0 / np.mean(np.diff(beat_times))
        print(f"[GEN] Detected BPM (Composite): {bpm:.1f}")
        return { "beat_times": beat_times, "duration": librosa.get_duration(y=self.y_drum, sr=self.sr) }

    def _harvest_all(self):
        pool = []
        pool.extend(self._harvest_stem_pyin(self.env_voc, self.y_voc, "vocal"))
        pool.extend(self._harvest_stem_pyin(self.env_oth, self.y_oth, "other"))
        pool.extend(self._harvest_stem_pyin(self.env_bass, self.y_bass, "bass"))
        pool.extend(self._harvest_stem_pyin(self.env_drum, None, "drums"))
        pool.sort(key=lambda x: x["time"])
        return pool

    def _score_with_priority(self, pool, beat_times):
        if not pool: return []
        beat_set = np.array(beat_times)
        last_midi_by_src = {}
        for n in pool:
            src = n["source"]
            n["score"] *= STEM_PRIORITIES.get(src, 1.0)
            if src != "drums":
                last = last_midi_by_src.get(src, -1)
                if last != -1 and abs(n["midi"] - last) > 0:
                    n["score"] *= 1.2 
                last_midi_by_src[src] = n["midi"]
            idx = (np.abs(beat_set - n["time"])).argmin()
            nearest = beat_set[idx]
            if abs(n["time"] - nearest) < 0.05:
                n["score"] *= 1.15 
        return pool

    def _harvest_stem_pyin(self, env, y_audio, source_name):
        # Calculate BPM from beat times if not already available
        beat_times = self.data["beat_times"]
        bpm = 120
        if len(beat_times) > 1:
            bpm = 60.0 / np.mean(np.diff(beat_times))
        
        # Debounce: Stricter for vocals to prevent flutter
        sixteenth_dur = 60.0 / bpm / 4.0
        wait_frames = int(sixteenth_dur * self.sr / 512)
        if source_name == "vocal": wait_frames = max(wait_frames, 5)
        else: wait_frames = max(wait_frames, 2)
        delta = 0.04
        if source_name == "drums": 
            delta = DRUM_ONSET_DELTA
            wait_frames = 2
        onset_frames = librosa.util.peak_pick(env, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                              pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                              delta=delta, wait=wait_frames)
        min_score = 0.1
        if source_name == "other": min_score = 0.25 
        candidates = []
        for t in onset_frames:
            score = env[t]
            if score < min_score: continue 
            midi = 0
            if source_name == "drums": midi = 36
            else:
                time_sec = librosa.frames_to_time(t, sr=self.sr)
                start_samp = int((time_sec + 0.02) * self.sr)
                end_samp = start_samp + PYIN_FRAME_LENGTH 
                if start_samp < len(y_audio):
                    slice_y = y_audio[start_samp:end_samp]
                    if len(slice_y) >= 1024:
                        fmin, fmax = 60, 1000 
                        if source_name == "bass": fmin, fmax = 40, 400
                        f0, _, _ = librosa.pyin(slice_y, fmin=fmin, fmax=fmax, sr=self.sr, frame_length=PYIN_FRAME_LENGTH)
                        f0 = f0[~np.isnan(f0)]
                        if len(f0) > 0: midi = int(round(librosa.hz_to_midi(np.median(f0))))
            if midi == 0: continue
            candidates.append({ "time": librosa.frames_to_time(t, sr=self.sr), "midi": midi, "score": score, "source": source_name, "dur": 0.0 })
        return candidates

    def generate(self, diff_name):
        print(f"[GEN] Generating {diff_name}...")
        cfg = DIFF_CONFIGS[diff_name]
        target_density = GLOBAL_DENSITY_CAP[diff_name]
        
        # 1. SIEVE
        filtered_pool = self._integrated_sieve(self.master_pool.copy(), target_density)

        # 2. QUANTIZE
        quantized_pool = self._quantize_strict(filtered_pool, self.data["beat_times"], cfg)
        
        # 3. LANE ALLOCATION
        poly_notes = self._allocate_lanes_poly(quantized_pool, cfg)
        
        # FIX 2: CONSOLIDATE & HOLDS
        # This happens AFTER lanes are assigned so we know which notes are truly "connected"
        final_notes = self._consolidate_holds(poly_notes)
        
        self._save_beatmap(final_notes, diff_name)
        return final_notes

    def _consolidate_holds(self, notes):
        """
        Merges rapid-fire notes in the same lane/source.
        If self.use_holds is True, creates Hold Notes.
        If False, just deletes the spam (De-Jitter).
        """
        if not notes: return []
        
        # Sort by time first
        notes.sort(key=lambda x: x["time"])
        
        consolidated = []
        active_stack = {} # Key: (lane, source), Value: index in consolidated list
        
        for n in notes:
            key = (n["lane"], n["source"])
            
            # Check if we can merge with the previous note in this lane
            if key in active_stack:
                prev_idx = active_stack[key]
                prev_note = consolidated[prev_idx]
                
                dt = n["time"] - prev_note["time"]
                
                # MERGE CONDITION:
                # 1. Within window (0.25s)
                # 2. Pitch is identical (OR logic allows for slight drift if you wanted)
                if dt < HOLD_MERGE_WINDOW and prev_note["midi"] == n["midi"]:
                    
                    # Extension Logic
                    if self.use_holds:
                        # Extend duration of previous note to reach this one
                        new_dur = (n["time"] - prev_note["time"]) + n.get("dur", 0)
                        prev_note["dur"] = max(prev_note["dur"], new_dur)
                    
                    # If NOT using holds, we do nothing to prev_note duration.
                    # BUT we effectively "absorb" the current note by NOT appending it to 'consolidated'.
                    # This acts as the "Spam Filter".
                    
                    continue # Skip adding this note (it's merged)
            
            # If no merge, register this as the new active note for this lane
            consolidated.append(n)
            active_stack[key] = len(consolidated) - 1
            
        return consolidated

    # ... (Sieve, Quantize, and Lane Alloc same as V98) ...

    def _integrated_sieve(self, pool, target_nps):
        duration = self.data["duration"]
        filtered = []
        window_size = 2.0
        cursor = 0.0
        while cursor < duration:
            window_notes = [n for n in pool if cursor <= n["time"] < cursor + window_size]
            if not window_notes:
                cursor += window_size
                continue
            window_notes.sort(key=lambda x: x["score"], reverse=True)
            budget = int(target_nps * window_size)
            accepted_in_window = []
            for note in window_notes:
                if len(accepted_in_window) >= budget: break
                is_clashing = False
                for acc in accepted_in_window:
                    dt = abs(note["time"] - acc["time"])
                    if dt < CROSS_STEM_DEBOUNCE:
                        is_clashing = True
                        break
                if not is_clashing: accepted_in_window.append(note)
            filtered.extend(accepted_in_window)
            cursor += window_size
        filtered.sort(key=lambda x: x["time"])
        return filtered

    def _quantize_strict(self, pool, beat_times, cfg):
        quantized = []
        allowed_grids = cfg["grids"]
        threshold = cfg.get("snap_threshold", 0.1)
        for n in pool:
            snapped_t, success = self._weighted_smart_snap(n["time"], beat_times, allowed_grids, threshold)
            if success:
                n["time"] = snapped_t
                quantized.append(n)
        return quantized

    def _allocate_lanes_poly(self, notes, cfg):
        lanes = cfg["lanes"]
        final_notes = []
        groups = {}
        for n in notes:
            k = round(n["time"], 3)
            if k not in groups: groups[k] = []
            groups[k].append(n)
        last_lane_center = lanes // 2
        for t in sorted(groups.keys()):
            stack = groups[t]
            max_poly = 1 if not cfg["chords"] else 4
            if len(stack) > max_poly:
                stack.sort(key=lambda x: x["score"], reverse=True)
                stack = stack[:max_poly]
            stack.sort(key=lambda x: x["midi"])
            count = len(stack)
            assigned_lanes = []
            if count == 1:
                n = stack[0]
                ideal = 0
                if n["source"] == "drums": 
                    ideal = int(lanes/2) if n["score"]>0.6 else 0
                else: 
                    r_min, r_max = VISUAL_RANGE_OTHER
                    if n["source"] == "vocal": r_min, r_max = VISUAL_RANGE_VOCAL
                    elif n["source"] == "bass": r_min, r_max = VISUAL_RANGE_BASS
                    norm = (n["midi"] - r_min) / (r_max - r_min)
                    norm = max(0.0, min(1.0, norm)) 
                    ideal = int(norm * (lanes-1))
                if ideal == last_lane_center: ideal = (ideal + 1) % lanes
                assigned_lanes.append(ideal)
            else:
                for i in range(count):
                    l = int(i * (lanes-1) / (count-1))
                    assigned_lanes.append(l)
            for i, n in enumerate(stack):
                n["lane"] = assigned_lanes[i]
                base_vol = STEM_VOLUMES.get(n["source"], 0.8)
                n["vol"] = base_vol * (0.7 + (n["score"] * 0.3))
                final_notes.append(n)
            if assigned_lanes: last_lane_center = int(np.mean(assigned_lanes))
        return final_notes

    def _weighted_smart_snap(self, t, beats, allowed_grids, max_error_sec):
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
            error = abs(candidate - t)
            penalty = 1.0 
            if div >= 16: penalty = 1.2
            if div >= 24: penalty = 1.5
            weighted_error = error * penalty
            if weighted_error < min_error:
                min_error = weighted_error
                best_time = candidate
        if min_error > max_error_sec: return t, False
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
                elif isinstance(v, (bool, np.bool_)): new_note[k] = bool(v)
            serializable_notes.append(new_note)
        with open(output_file, 'w') as f:
            json.dump(serializable_notes, f, indent=2)
        print(f"[GEN] Saved {diff_name} beatmap to {output_file}")

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
    }
    
    gen = MapGenerator(stems, use_holds=args.holds)
    for diff in ["EASY", "NORMAL", "HARD", "INSANE"]:
        gen.generate(diff)