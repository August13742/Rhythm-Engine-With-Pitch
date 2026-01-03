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
#        V102: "THE ADAPTIVE CONDUCTOR"
# ==========================================

HPSS_CONFIG = { 
    "vocal": 2.0, 
    "other": 3.0, 
    "bass": 1.5, 
    "drums": 3.5,
    "piano": 2.0, 
    "guitar": 2.5 
}

VOCAL_SMOOTH_FACTOR = 1

GLOBAL_DENSITY_CAP = {
    "EASY":   1.5,
    "NORMAL": 3.5,
    "HARD":   6.0, 
    "INSANE": 8.0 
}

# BASE PRIORITIES
# These are defaults. If Vocals are missing, these get promoted.
DEFAULT_PRIORITIES = {
    "vocal":  1.3,
    "piano":  1.1,
    "guitar": 1.1,
    "drums":  1.0,
    "bass":   0.8,
    "other":  0.6
}

CROSS_STEM_DEBOUNCE = 0.04 

# DYNAMIC MIXING
VOCAL_MASKING_STRENGTH = 0.7
VOCAL_SOLO_BOOST = 1.3

STEM_VOLUMES = {
    "vocal":  1.0,
    "piano":  0.9,
    "guitar": 0.85,
    "other":  0.6,
    "bass":   0.8,
    "drums":  0.7 
}

# VISUAL MAPPING
VISUAL_RANGE_VOCAL  = (48, 84) 
VISUAL_RANGE_PIANO  = (48, 96) 
VISUAL_RANGE_GUITAR = (40, 76) 
VISUAL_RANGE_OTHER  = (48, 84)
VISUAL_RANGE_BASS   = (36, 60) 

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
HOLD_MERGE_WINDOW = 0.25 

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[GEN] V102 Initializing (Holds: {use_holds})...")
        self.sr = 44100
        self.stems_path = stems_path
        self.use_holds = use_holds
        
        # 1. LOAD MANIFEST & AUDIO
        # We need to load strictly to handle the 1s dummy files correctly
        self.manifest = self._load_manifest(stems_path)
        self.audio_data = self._smart_load_stems(stems_path)
        
        # Assign to convenient handles
        self.y_voc = self.audio_data["vocals"]
        self.y_oth = self.audio_data["other"]
        self.y_bass = self.audio_data["bass"]
        self.y_drum = self.audio_data["drums"]
        self.y_piano = self.audio_data["piano"]
        self.y_guitar = self.audio_data["guitar"]

        # 2. ADAPTIVE HIERARCHY
        # If Vocals are silent, promote instruments to Protagonist status
        self.priorities = DEFAULT_PRIORITIES.copy()
        if not self.manifest.get("vocals", {}).get("exists", False):
            print("[GEN] Instrumental Track Detected! Promoting Piano & Guitar.")
            self.priorities["piano"] = 1.3
            self.priorities["guitar"] = 1.3
            # Disable masking (since there is no vocal to mask against)
            global VOCAL_MASKING_STRENGTH
            VOCAL_MASKING_STRENGTH = 1.0 

        # 3. ANALYZE VOCAL PRESENCE
        # Only if vocals actually exist
        if self.manifest.get("vocals", {}).get("exists", False):
            print("[GEN] Analyzing Vocal Presence...")
            self.vocal_rms = self._calculate_rms(self.y_voc)
            
            # BLEED REMOVAL
            print("[GEN] Cleaning Vocal Bleed...")
            backing_mix = self.y_oth + self.y_bass + self.y_drum + self.y_piano + self.y_guitar
            self.y_voc = self._clean_vocal_bleed(self.y_voc, backing_mix)
        else:
            self.vocal_rms = np.zeros(1)

        # 4. HPSS & ENVELOPE (Selective Processing)
        # We only process stems that are marked as existing in the manifest
        self.envs = {}
        self.hpss_stems = {}
        
        print("[GEN] Processing Active Stems...")
        for stem_name in ["vocals", "bass", "drums", "piano", "guitar", "other"]:
            if self.manifest.get(stem_name, {}).get("exists", False):
                self._process_stem(stem_name)
            else:
                # Assign dummy envelopes for logic safety
                self.envs[stem_name] = np.zeros(1)

        # 5. RHYTHM & HARVEST
        self.data = self._analyze_rhythm_composite()
        raw_pool = self._harvest_all()
        self.master_pool = self._score_with_priority(raw_pool, self.data["beat_times"])
        print(f"[GEN] Master Pool Ready: {len(self.master_pool)} notes")

    def _load_manifest(self, stems_path):
        # Try to find manifest in the same folder as vocals
        folder = os.path.dirname(stems_path["vocals"])
        manifest_path = os.path.join(folder, "stems_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        # Fallback if no manifest (assume everything exists)
        return {k: {"exists": True, "is_silent": False} for k in stems_path.keys()}

    def _smart_load_stems(self, paths):
        """
        Loads all stems, determines the Max Duration, and pads everyone to match.
        This prevents numpy errors when adding a 1s silent file to a 3m song.
        """
        loaded = {}
        max_len = 0
        
        # Pass 1: Load and find max length
        print("[GEN] Loading Audio into Memory...")
        for name, path in paths.items():
            if os.path.exists(path):
                # We check manifest to see if we should treat it as real audio
                info = self.manifest.get(name, {"is_silent": False})
                
                if info["is_silent"]:
                    # It's a dummy file. Don't load it yet, we will generate zeros later.
                    loaded[name] = None
                else:
                    y, _ = librosa.load(path, sr=self.sr)
                    loaded[name] = y
                    if len(y) > max_len: max_len = len(y)
            else:
                loaded[name] = None

        if max_len == 0: raise ValueError("No valid audio stems found!")

        # Pass 2: Pad/Generate
        final_audio = {}
        for name, data in loaded.items():
            if data is None:
                final_audio[name] = np.zeros(max_len, dtype=np.float32)
            else:
                # Pad if slightly shorter (e.g. mp3/wav conversion drift)
                if len(data) < max_len:
                    padded = np.zeros(max_len, dtype=np.float32)
                    padded[:len(data)] = data
                    final_audio[name] = padded
                else:
                    final_audio[name] = data
        
        return final_audio

    def _process_stem(self, name):
        # HPSS Separation
        y = self.audio_data[name]
        margin = HPSS_CONFIG.get(name, 2.0)
        
        if name == "drums":
            # Drums: Percussive focus
            _, y_processed = librosa.effects.hpss(y, margin=margin)
            self.hpss_stems[name] = y_processed
            self.envs[name] = self._get_env(y_processed)
        else:
            # Melodic: Harmonic focus
            y_processed, _ = librosa.effects.hpss(y, margin=margin)
            self.hpss_stems[name] = y_processed
            smooth = VOCAL_SMOOTH_FACTOR if name == "vocals" else 0
            self.envs[name] = self._get_env(y_processed, smooth_factor=smooth)

    def _analyze_rhythm_composite(self):
        print("[GEN] Analyzing Composite Rhythm...")
        # Add tracks only if they have energy
        y_composite = self.y_drum.copy()
        
        if self.manifest["bass"]["exists"]: y_composite += self.y_bass
        if self.manifest["piano"]["exists"]: y_composite += (self.y_piano * 0.8)
        if self.manifest["guitar"]["exists"]: y_composite += (self.y_guitar * 0.8)
        
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
        # Only harvest active stems
        if self.manifest["vocals"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["vocals"], self.y_voc, "vocal"))
        if self.manifest["piano"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["piano"], self.audio_data["piano"], "piano")) # Use raw audio for pYIN pitch
        if self.manifest["guitar"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["guitar"], self.audio_data["guitar"], "guitar"))
        if self.manifest["other"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["other"], self.audio_data["other"], "other"))
        if self.manifest["bass"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["bass"], self.audio_data["bass"], "bass"))
        if self.manifest["drums"]["exists"]: 
            pool.extend(self._harvest_stem_pyin(self.envs["drums"], None, "drums"))
        
        pool.sort(key=lambda x: x["time"])
        return pool

    def _score_with_priority(self, pool, beat_times):
        if not pool: return []
        beat_set = np.array(beat_times)
        last_midi_by_src = {}
        
        for n in pool:
            src = n["source"]
            t = n["time"]
            
            # 1. Base Priority (Using Adaptive Priorities)
            n["score"] *= self.priorities.get(src, 1.0)
            
            # 2. THE CONDUCTOR
            voc_energy = self._get_vocal_presence_at_time(t)
            
            if src in ["piano", "guitar"]:
                if voc_energy > 0.2: 
                    n["score"] *= VOCAL_MASKING_STRENGTH
                else:
                    n["score"] *= VOCAL_SOLO_BOOST
            
            if src == "other" and voc_energy > 0.1:
                n["score"] *= 0.5
            
            # 3. Contour Bonus
            if src not in ["drums", "other"]:
                last = last_midi_by_src.get(src, -1)
                if last != -1 and abs(n["midi"] - last) > 0:
                    n["score"] *= 1.2 
                last_midi_by_src[src] = n["midi"]
            
            # 4. Grid Gravity
            idx = (np.abs(beat_set - t)).argmin()
            nearest = beat_set[idx]
            if abs(t - nearest) < 0.05:
                n["score"] *= 1.15 
                
        return pool

    # ... (Rest of the methods: _clean_vocal_bleed, _get_env, _harvest_stem_pyin, generate, etc. remain unchanged from V101)
    
    def _get_vocal_presence_at_time(self, t):
        if len(self.vocal_rms) <= 1: return 0.0
        hop_len = 512
        frame = int(t * self.sr / hop_len)
        if 0 <= frame < len(self.vocal_rms):
            return self.vocal_rms[frame]
        return 0.0
    
    def _clean_vocal_bleed(self, y_voc, y_backing):
        frame_len = 2048
        hop_len = 512
        if len(y_voc) < frame_len: return y_voc
        rms_voc = librosa.feature.rms(y=y_voc, frame_length=frame_len, hop_length=hop_len)[0]
        rms_back = librosa.feature.rms(y=y_backing, frame_length=frame_len, hop_length=hop_len)[0]
        mask = np.ones_like(rms_voc)
        bleed_indices = (rms_back > 0.05) & (rms_voc < (rms_back * 0.15))
        silence_indices = (rms_voc < 0.005)
        mask[bleed_indices] = 0.0
        mask[silence_indices] = 0.0
        mask = scipy.signal.medfilt(mask, kernel_size=5)
        mask_upsampled = scipy.ndimage.zoom(mask, len(y_voc) / len(mask), order=1)
        if len(mask_upsampled) < len(y_voc):
            mask_upsampled = np.pad(mask_upsampled, (0, len(y_voc) - len(mask_upsampled)))
        elif len(mask_upsampled) > len(y_voc):
            mask_upsampled = mask_upsampled[:len(y_voc)]
        return y_voc * mask_upsampled

    @lru_cache(maxsize=32)
    def _get_env_cached(self, y_hash, smooth_factor=0):
        return self._get_env_impl(y_hash, smooth_factor)
    
    def _get_env(self, y, smooth_factor=0):
        if len(y) <= 1: return np.zeros(1)
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

    def _harvest_stem_pyin(self, env, y_audio, source_name):
        if len(env) <= 1: return []
        beat_times = self.data["beat_times"]
        bpm = 120
        if len(beat_times) > 1:
            bpm = 60.0 / np.mean(np.diff(beat_times))
        
        sixteenth_dur = 60.0 / bpm / 4.0
        wait_frames = int(sixteenth_dur * self.sr / 512)
        
        if source_name == "vocal": wait_frames = max(wait_frames, 5)
        elif source_name == "piano": wait_frames = max(wait_frames, 3)
        elif source_name == "guitar": wait_frames = max(wait_frames, 3)
        else: wait_frames = max(wait_frames, 2)
        
        delta = 0.04
        if source_name == "drums": 
            delta = DRUM_ONSET_DELTA
            wait_frames = 2
            
        onset_frames = librosa.util.peak_pick(env, pre_max=ONSET_PRE_MAX, post_max=ONSET_POST_MAX, 
                                              pre_avg=ONSET_PRE_AVG, post_avg=ONSET_POST_AVG, 
                                              delta=delta, wait=wait_frames)
        
        min_score = 0.1
        if source_name == "other": min_score = 0.3 
        if source_name == "piano": min_score = 0.15 
        if source_name == "guitar": min_score = 0.15

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
                        elif source_name == "piano": fmin, fmax = 27, 4000
                        elif source_name == "guitar": fmin, fmax = 80, 1200
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
        filtered_pool = self._integrated_sieve(self.master_pool.copy(), target_density)
        quantized_pool = self._quantize_strict(filtered_pool, self.data["beat_times"], cfg)
        poly_notes = self._allocate_lanes_poly(quantized_pool, cfg)
        final_notes = self._consolidate_holds(poly_notes)
        self._save_beatmap(final_notes, diff_name)
        return final_notes

    def _consolidate_holds(self, notes):
        if not notes: return []
        notes.sort(key=lambda x: x["time"])
        consolidated = []
        active_stack = {} 
        for n in notes:
            key = (n["lane"], n["source"])
            if key in active_stack:
                prev_idx = active_stack[key]
                prev_note = consolidated[prev_idx]
                dt = n["time"] - prev_note["time"]
                if dt < HOLD_MERGE_WINDOW and prev_note["midi"] == n["midi"]:
                    if self.use_holds:
                        new_dur = (n["time"] - prev_note["time"]) + n.get("dur", 0)
                        prev_note["dur"] = max(prev_note["dur"], new_dur)
                    continue 
            consolidated.append(n)
            active_stack[key] = len(consolidated) - 1
        return consolidated

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
                    elif n["source"] == "piano": r_min, r_max = VISUAL_RANGE_PIANO
                    elif n["source"] == "guitar": r_min, r_max = VISUAL_RANGE_GUITAR
                    
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
    
    def _calculate_rms(self, y):
        # Calculate RMS for dynamic scoring lookups
        frame_len = 2048
        hop_len = 512
        if len(y) < frame_len: return np.zeros(1)
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        # Normalize roughly 0-1
        if rms.max() > 0: rms /= rms.max()
        return rms

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
    
    gen = MapGenerator(stems, use_holds=args.holds)
    for diff in ["EASY", "NORMAL", "HARD", "INSANE"]:
        gen.generate(diff)