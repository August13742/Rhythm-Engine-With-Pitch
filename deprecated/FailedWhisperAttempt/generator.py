"""
dependencies: 
pip install faster-whisper torchcrepe librosa soundfile numpy torch scipy
"""
import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse
import torch
import torchcrepe
from separator import generate_speech_notes

# ==========================================
#        V306: ZERO-POINT ENGINE
#        Consensus Voting System (Rap Cleaner)
# ==========================================

# ==========================================
#        MASTER CONFIGURATION
# ==========================================
CONFIG = {
    "audio": {
        "sr": 44100,
        "hop_length": 512,
        "inst_offset_sec": -0.015, 
    },
    "holds": {
        "min_dur": 0.75,      
        "max_dur": 2.00,
        "gap_buffer": 0.05,   
        "energy_decay": 0.50, 
    },
    "analysis": {
        "pre_max": 3, "post_max": 3, "pre_avg": 3, "post_avg": 3, "wait": 4,
        "sens": {
            "drums": 0.10, "bass": 0.10, "piano": 0.06,
            "guitar": 0.05, "other": 0.08, "vocals": 0.05
        },
        "p_center_window": 0.25, 
        "p_center_thresh": 0.2, 
        
        # VOTING CONFIG
        "min_vocal_prob": 0.25,     # Whisper text probability
        "voting_tolerance": 2.5,    # Semitones allowed between Crepe and pYIN
    },
    "mixing": {
        "prio": {
            "vocals": 2.0, "drums": 1.2, "bass": 1.0, 
            "piano": 1.2, "guitar": 1.1, "other": 0.8
        }
    },
    "voting": {
        "window": 0.05,         
        "min_support": 0.4,     
        "boost_scale": 0.6,     
        "mask_threshold": 0.4,  
        "mask_ratio": 2.5       
    },
    "gameplay": {
        "flam_window": 0.06,    
        "hard_limit_scale": 1.5 
    },
    # Magic Numbers & Constants
    "harvest": {
        "pyin_window_samples": 4096,
        "wait_frames_melody": 5,
        "p_center_shift_frames": 20,
    },
    "voting_consensus": {
        "crepe_high_confidence": 0.85,
        "octave_error_tolerance": 12,
        "min_tap_duration": 0.15,
    },
    "hierarchy": {
        "ambient_drum_threshold": 0.15,
        "rhythm_density_threshold": 0.4,
        "melodic_density_ratio": 1.5,
        "rhythm_dominance_ratio": 1.3,
        "vocal_presence_threshold": 0.1,
        "percentile_density": 85,
    },
    "quantization": {
        "quantize_error_threshold": 0.07,
        "quantize_blend_ratio": 0.5,
    },
    "sieve": {
        "window_duration": 1.0,
        "window_step": 0.5,
        "high_score_threshold": 0.8,
    },
    "lanes": {
        "drum_high_score": 0.8,
        "vocal_range_scale": 3.5,
    },
    "overlap": {
        "bass_mix_ratio": 0.8,
        "vocal_energy_high": 0.2,
        "vocal_energy_ratio_high": 2.0,
        "vocal_energy_ratio_mid": 1.2,
        "beat_snap_threshold": 0.05,
        "beat_snap_bonus": 1.2,
        "hold_score_bonus": 1.1,
    }
}

VISUAL_RANGES = {
    "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
    "bass": (36, 60), "other": (48, 88)
}

DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "grids": [4], "poly": 1, "density_cap": 1.5, "min_score": 0.50, "chaos": 0.0 },
    "NORMAL": { "lanes": 4, "grids": [4, 8], "poly": 2, "density_cap": 3.0, "min_score": 0.4, "chaos": 0.0 },
    "HARD":   { "lanes": 4, "grids": [4, 8, 12, 16], "poly": 3, "density_cap": 4.5, "min_score": 0.3, "chaos": 0.1 },
    "INSANE": { "lanes": 4, "grids": [4, 8, 12, 16, 24, 32], "poly": 4, "density_cap": 6.5, "min_score": 0.2, "chaos": 0.2 },
}

class MapGenerator:
    def __init__(self, stems_path, use_holds=True, speech_notes_path=None):
        print(f"[GEN] ENGINE INITIALIZING...")
        self.stems_path = stems_path
        self.use_holds = use_holds
        self.speech_notes_path = speech_notes_path
        
        # Load Config
        self.c_audio = CONFIG["audio"]
        self.c_analysis = CONFIG["analysis"]
        self.c_mix = CONFIG["mixing"]
        
        # Load Data
        self.manifest = self._load_manifest(stems_path)
        self.audio_data = self._smart_load_stems(stems_path)
        
        # Preprocess
        self.envs = {}
        self.rms_curves = {} 
        self._preprocess_audio()

        self.rhythm_data = self._analyze_rhythm_composite()
        self.vote_weights = self._calculate_dynamic_weights()
        
        # 1. Harvest
        raw_pool = self._harvest_all()
        print(f"[GEN] Total Candidates: {len(raw_pool)}")

        # 2. Vote (Psychoacoustic Sync)
        voted_pool = self._apply_coincidence_voting(raw_pool)

        # 3. Score
        self.master_pool = self._score_and_sort(voted_pool, self.rhythm_data["beat_times"])

    # =========================================================
    #  HARVESTING (With Latency Correction)
    # =========================================================
    def _harvest_all(self):
        pool = []
        m = self.manifest
        
        def exists(key): return m.get(key, {}).get("exists", False) and not m.get(key, {}).get("is_silent", False)

        # 1. Vocals (Whisper + P-Center) - No Offset (Already Tight)
        if exists("vocals"):
            if self.speech_notes_path and os.path.exists(self.speech_notes_path):
                pool.extend(self._load_speech_notes())
            else:
                pool.extend(self._harvest_vocals_sota())

        # 2. Instruments (DSP) - APPLYING OFFSET HERE
        sens = self.c_analysis["sens"]
        if exists("drums"):
            pool.extend(self._harvest_onsets("drums", 36, False, sens["drums"]))
        if exists("bass"):
            pool.extend(self._harvest_melodic_legacy("bass", (40, 400), sens["bass"], False))
        if exists("piano"):
            pool.extend(self._harvest_melodic_legacy("piano", (27, 4000), sens["piano"], False))
        if exists("guitar"):
            pool.extend(self._harvest_melodic_legacy("guitar", (80, 1200), sens["guitar"], False))
        if exists("other"):
            pool.extend(self._harvest_melodic_legacy("other", (100, 1500), sens["other"], True))

        return pool

    def _harvest_onsets(self, source, midi, can_hold, sens):
        if source not in self.envs: return []
        env = self.envs[source]
        cfg = self.c_analysis
        sr = self.c_audio["sr"]
        offset = self.c_audio["inst_offset_sec"] # FIX: Apply -18ms shift
        
        frames = librosa.util.peak_pick(env, pre_max=cfg["pre_max"], post_max=cfg["post_max"], 
                                        pre_avg=cfg["pre_avg"], post_avg=cfg["post_avg"], 
                                        delta=sens, wait=cfg["wait"])
        notes = []
        min_dur = CONFIG["holds"]["min_dur"]
        
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + offset # <--- LATENCY FIX
            if t < 0: continue
            
            dur = 0.0
            if can_hold and self.use_holds: dur = self._measure_signal_duration(source, f)
            
            notes.append({ 
                "time": t, "midi": midi, "dur": dur, "source": source, 
                "score": float(env[f]), "type": "hold" if dur >= min_dur else "tap" 
            })
        return notes

    def _harvest_melodic_legacy(self, source, freq_range, sens, can_hold):
        if source not in self.envs: return []
        env = self.envs[source]; y = self.audio_data[source]
        sr = self.c_audio["sr"]; hop = self.c_audio["hop_length"]
        offset = self.c_audio["inst_offset_sec"] # FIX: Apply -18ms shift
        cfg = self.c_analysis
        
        frames = librosa.util.peak_pick(env, pre_max=cfg["pre_max"], post_max=cfg["post_max"], 
                                        pre_avg=cfg["pre_avg"], post_avg=cfg["post_avg"], 
                                        delta=sens, wait=CONFIG["harvest"]["wait_frames_melody"]) # Wait is slightly higher for melody
        notes = []
        
        for f in frames:
            if env[f] < sens: continue
            t = librosa.frames_to_time(f, sr=sr) + offset # <--- LATENCY FIX
            if t < 0: continue
            
            # Pitch detect
            pyin_len = CONFIG["harvest"]["pyin_window_samples"]
            start = int(f * hop); end = start + pyin_len
            if end > len(y): break
            f0, _, _ = librosa.pyin(y[start:end], fmin=freq_range[0], fmax=freq_range[1], sr=sr, frame_length=pyin_len)
            
            valid = f0[~np.isnan(f0)]
            if len(valid) == 0: continue
            
            midi = int(round(librosa.hz_to_midi(np.median(valid))))
            dur = 0.0
            if can_hold: dur = self._measure_signal_duration(source, f)
            
            notes.append({ "time": t, "midi": midi, "dur": dur, "source": source, "score": float(env[f]), "type": "hold" if dur > 0 else "tap" })
        return notes

    def _harvest_vocals_sota(self):
        """
        Retrieves RAW notes from Separator and applies CONSENSUS VOTING.
        """
        # 1. Get Raw Data (Cached or Fresh)
        raw_notes = generate_speech_notes(self.stems_path["vocals"], use_holds=self.use_holds)
        
        # 2. Apply Voting Logic to calc 'midi' key
        return self._process_vocal_consensus(raw_notes)
    
    def _load_speech_notes(self):
        """
        Loads Raw notes from JSON and applies CONSENSUS VOTING.
        """
        print(f"[GEN] Loading cached speech notes from: {self.speech_notes_path}")
        with open(self.speech_notes_path, 'r', encoding='utf-8') as f: 
            raw_notes = json.load(f)
            
        # 2. Apply Voting Logic to calc 'midi' key
        return self._process_vocal_consensus(raw_notes)

    def _process_vocal_consensus(self, raw_notes):
        """
        Shared Logic: Converts Raw {crepe, pyin} notes -> Final {midi} notes.
        """
        processed = []
        min_prob = self.c_analysis["min_vocal_prob"]
        tolerance = self.c_analysis["voting_tolerance"]
        
        killed_count = 0
        accepted_count = 0

        for n in raw_notes:
            # 1. Filter Whispers
            # Handle cases where score might be missing in older caches
            score = n.get("score", n.get("probability", 0.0))
            if score < min_prob: continue
            
            # Ensure we have the raw data keys, defaulting to 0 if missing
            crepe = n.get("crepe_midi", n.get("midi", 0)) # Fallback to 'midi' if already processed
            pyin = n.get("pyin_midi", 0)
            c_conf = n.get("crepe_conf", 0)
            
            # If 'midi' already exists and looks processed, trust it (Legacy support)
            if "midi" in n and "crepe_midi" not in n:
                processed.append(n)
                continue

            final_midi = 0
            
            # --- VOTING LOGIC ---
            
            # Case A: Crepe is highly confident (Singing)
            if c_conf > self.c_analysis["crepe_high_confidence"] and crepe > 0:
                final_midi = int(round(crepe))
                n["mode"] = "sung"
            
            # Case B: Disagreement Check (Rap / Grit)
            else:
                diff = abs(crepe - pyin)
                
                # B1. Consensus (Both engines agree roughly)
                if diff <= tolerance and crepe > 0 and pyin > 0:
                    final_midi = int(round((crepe + pyin) / 2))
                    n["mode"] = "rap_consensus"
                
                # B2. Octave Error (One is exactly an octave off)
                elif abs(diff - self.c_analysis["octave_error_tolerance"]) <= tolerance:
                    final_midi = int(round(min(crepe, pyin)))
                    n["mode"] = "rap_octave_fix"
                    
                # B3. DIVERGENCE (Engines disagree -> Noise/Garbage)
                else:
                    killed_count += 1
                    continue

            # CRITICAL: Set the 'midi' key that _allocate_lanes expects
            n["midi"] = final_midi
            
            # Duration Validation
            if n.get("dur", 0) < self.c_analysis["min_tap_duration"]: 
                n["type"] = "tap"
                n["dur"] = 0.0
            else:
                n["type"] = "hold"
                
            processed.append(n)
            accepted_count += 1

        print(f"[GEN] Vocal Consensus: Accepted {accepted_count}. Discarded {killed_count} (Ambiguous Pitch).")
        
        # 3. P-Center Alignment
        return self._align_to_acoustic_onset(processed, "vocals")
    
    
    def _align_to_acoustic_onset(self, notes, stem_name):
        """P-Center Correction: Moves timestamps FORWARD to the energy peak."""
        env = self.envs.get(stem_name)
        if env is None: return notes
        
        sr = self.c_audio["sr"]; hop = self.c_audio["hop_length"]
        win_frames = int(self.c_analysis["p_center_window"] * sr / hop)
        thresh = self.c_analysis["p_center_thresh"]
        
        for n in notes:
            start = int(n["time"] * sr / hop)
            end = min(len(env), start + win_frames)
            if end <= start: continue
            
            window = env[start:end]
            if len(window) == 0: continue
            
            peak = np.argmax(window)
            if window[peak] > thresh: 
                new_time = librosa.frames_to_time(start + peak, sr=sr)
                if 0 < (new_time - n["time"]) < 0.2: 
                    n["time"] = new_time
        return notes

    # =========================================================
    #  ANALYSIS & HELPERS
    # =========================================================
    def _measure_signal_duration(self, source, start_frame):
        if source not in self.rms_curves: return 0.0
        rms = self.rms_curves[source]
        sr = self.c_audio["sr"]; hop = self.c_audio["hop_length"]
        h_cfg = CONFIG["holds"]
        
        thresh = rms[start_frame] * h_cfg["energy_decay"]
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        
        curr = start_frame + 1; end = min(len(rms), start_frame + max_dist)
        while curr < end:
            if rms[curr] < thresh: break
            curr += 1
        dur = librosa.frames_to_time(curr - start_frame, sr=sr)
        return dur if dur > h_cfg["min_dur"] else 0.0

    def _calculate_dynamic_weights(self):
        """
        Dynamic Hierarchy V3: Relative Dominance
        - Fixes "Everything is Drum Focused" by comparing Melodic Sum vs Rhythm Sum.
        - Detects "Lead Instrument" (Sax in 'other', Solo in 'guitar') by finding the outlier.
        """
        m = self.manifest
        weights = { "drums": 0.0, "bass": 0.0, "vocals": 0.0, "piano": 0.0, "guitar": 0.0, "other": 0.0 }
        
        # 1. Base Existence & RMS Calculation
        if m.get("drums", {}).get("exists"): weights["drums"] = 1.0
        if m.get("bass", {}).get("exists"):  weights["bass"]  = 0.9
        
        def get_density(stem):
            if stem in self.rms_curves and len(self.rms_curves[stem]) > 0:
                # Percentile captures "active playing" better than mean
                percentile = CONFIG["hierarchy"]["percentile_density"]
                return float(np.percentile(self.rms_curves[stem], percentile)) 
            return 0.0

        d = { k: get_density(k) for k in ["drums", "bass", "piano", "guitar", "other", "vocals"] }
        
        print("-" * 40)
        print(f"[GEN] HIERARCHY V3 ANALYSIS:")
        print(f"      > Densities: {json.dumps({k: round(v, 2) for k, v in d.items()})}")

        # 2. Identify the Primary Lead (Melodic)
        melodic_keys = ["piano", "guitar", "other"]
        h_cfg = CONFIG["hierarchy"]
        # Filter out empty stems
        active_melodics = {k: d[k] for k in melodic_keys if d[k] > 0.01}
        
        lead_inst = "vocals"
        max_mel_density = 0.0
        
        if active_melodics:
            lead_inst = max(active_melodics, key=active_melodics.get)
            max_mel_density = active_melodics[lead_inst]

        # 3. Determine Context
        # Compare Rhythm (Drums) vs The Loudest Melodic Instrument
        
        # Scenario A: AMBIENT / ACOUSTIC
        # Drums are very quiet OR significantly quieter than the lead instrument
        if d["drums"] < h_cfg["ambient_drum_threshold"] or (max_mel_density > d["drums"] * h_cfg["melodic_density_ratio"]):
            print(f"[GEN] >> MODE: MELODIC DRIVER (Lead: {lead_inst.upper()})")
            weights["drums"] = 0.8
            weights["bass"] = 0.8
            weights[lead_inst] = 1.2 # Boost the lead (e.g., Sax in 'other')
            # Boost other melodics slightly less
            for k in melodic_keys:
                if k != lead_inst and k in active_melodics: weights[k] = 1.0
                
        # Scenario B: HEAVY RHYTHM / DANCE
        # Drums are dominant and much louder than melody
        elif d["drums"] > h_cfg["rhythm_density_threshold"] and d["drums"] > (max_mel_density * h_cfg["rhythm_dominance_ratio"]):
             print(f"[GEN] >> MODE: RHYTHM DRIVER (Drums > All)")
             weights["drums"] = 1.2
             weights["bass"] = 1.1
             # Suppress melody slightly to clean up beat
             for k in melodic_keys: weights[k] = 0.8
             
        # Scenario C: ENSEMBLE / JAZZ / ROCK
        # Drums are loud, but instruments are also loud (balanced)
        else:
            print(f"[GEN] >> MODE: BALANCED ENSEMBLE")
            weights["drums"] = 1.0
            weights["bass"] = 1.0
            # Everything gets fair play, but Lead gets a tiny edge
            for k in melodic_keys: weights[k] = 0.9
            if lead_inst in weights: weights[lead_inst] = 1.05

        # 4. Vocal Handling
        # If vocals are present, they usually sit on top, but not always 
        if d["vocals"] > h_cfg["vocal_presence_threshold"]:
            weights["vocals"] = 1.1
        else:
            weights["vocals"] = 0.8 # Instrumental section or quiet vox

        print(f"      > Final Weights: {json.dumps(weights)}")
        print("-" * 40)
        return weights

    def _apply_coincidence_voting(self, pool):
        v_cfg = CONFIG["voting"]
        window = v_cfg["window"]
        pool.sort(key=lambda x: x["time"])
        n_notes = len(pool)
        
        for i in range(n_notes):
            curr = pool[i]; t = curr["time"]; src = curr["source"]
            support = 0.0; energy = 0.0
            
            # Simple window scan
            scan_range = CONFIG["harvest"]["p_center_shift_frames"]
            start_scan = max(0, i - scan_range) # Optimization: Only look at neighbor frames
            end_scan = min(n_notes, i + scan_range)
            
            for j in range(start_scan, end_scan):
                if i == j: continue
                n = pool[j]
                if abs(n["time"] - t) > window: continue
                
                if n["source"] != src:
                    w = self.vote_weights.get(n["source"], 0.2)
                    support += w * n["score"]
                    energy += n["score"]
            
            if support > v_cfg["min_support"]:
                curr["score"] *= (1.0 + support * v_cfg["boost_scale"])
            elif curr["score"] < v_cfg["mask_threshold"]:
                if src != "vocals" and energy > (curr["score"] * v_cfg["mask_ratio"]):
                    curr["score"] *= 0.1 # Kill weak note

        return pool

    def _score_and_sort(self, pool, beat_times):
        prio = self.c_mix["prio"]
        beat_arr = np.array(beat_times)
        
        for n in pool:
            # 1. Apply Hierarchy Weights
            n["score"] *= prio.get(n["source"], 1.0)
            
            # 2. Dynamic Vocal Masking
            if self.manifest.get("vocals", {}).get("exists") and n["source"] != "vocals":
                voc_energy = self._get_vocal_energy(n["time"])
                inst_energy = n["score"] 
                
                if voc_energy > 0.2:
                    ratio = voc_energy / (inst_energy + 0.05) 
                    if ratio > 2.0: n["score"] *= 0.5 
                    elif ratio > 1.2: n["score"] *= 0.8
            
            # 3. Beat Snap Bonus
            overlap_cfg = CONFIG["overlap"]
            if len(beat_arr) > 0:
                idx = (np.abs(beat_arr - n["time"])).argmin()
                if abs(n["time"] - beat_arr[idx]) < overlap_cfg["beat_snap_threshold"]: n["score"] *= overlap_cfg["beat_snap_bonus"]
            
            if n.get("type") == "hold": n["score"] *= overlap_cfg["hold_score_bonus"]
            
        pool.sort(key=lambda x: x["time"])
        return pool

    # =========================================================
    #  STANDARD HELPERS
    # =========================================================
    def _load_manifest(self, stems_path):
        folder = os.path.dirname(stems_path["vocals"])
        path = os.path.join(folder, "stems_manifest.json")
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return {k: {"exists": True, "is_silent": False} for k in stems_path.keys()}

    def _smart_load_stems(self, paths):
        loaded = {}; max_len = 0; sr = self.c_audio["sr"]
        print("[GEN] Loading Audio...")
        for name, path in paths.items():
            info = self.manifest.get(name, {"is_silent": False})
            if os.path.exists(path) and not info.get("is_silent", False):
                y, _ = librosa.load(path, sr=sr, mono=True)
                loaded[name] = y; max_len = max(max_len, len(y))
            else: loaded[name] = None
        
        final = {}
        for name in paths.keys():
            data = loaded.get(name)
            if data is None: final[name] = np.zeros(max_len, dtype=np.float32)
            elif len(data) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(data)] = data
                final[name] = padded
            else: final[name] = data
        return final

    def _preprocess_audio(self):
        sr = self.c_audio["sr"]; hop = self.c_audio["hop_length"]
        for name, y in self.audio_data.items():
            if not self.manifest.get(name, {}).get("exists", False):
                self.envs[name] = np.zeros(1); self.rms_curves[name] = np.zeros(1); continue
            
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
            if rms.max() > 0: rms /= rms.max()
            self.rms_curves[name] = rms
            
            if name == "drums":
                y_h, y_p = librosa.effects.hpss(y)
                self.envs[name] = librosa.onset.onset_strength(y=y_p, sr=sr)
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=sr)
            if self.envs[name].max() > 0: self.envs[name] /= self.envs[name].max()

    def _analyze_rhythm_composite(self):
        sr = self.c_audio["sr"]
        bass_ratio = CONFIG["overlap"]["bass_mix_ratio"]
        y = self.audio_data["drums"] + (self.audio_data["bass"] * bass_ratio)
        env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr)
        return { "beat_times": librosa.frames_to_time(beats, sr=sr), "duration": librosa.get_duration(y=y, sr=sr) }

    def _get_vocal_energy(self, t):
        if "vocals" not in self.rms_curves: return 0.0
        f = int(t * self.c_audio["sr"] / self.c_audio["hop_length"])
        return self.rms_curves["vocals"][f] if f < len(self.rms_curves["vocals"]) else 0.0

    # =========================================================
    #  GENERATION PHASE
    # =========================================================
    def generate(self, diff_name):
        print(f"[GEN] Generating {diff_name}")
        diff_cfg = DIFF_CONFIGS.get(diff_name, {})
        
        grids = diff_cfg.get("grids", [4])
        quantized = self._quantize(self.master_pool, self.rhythm_data["beat_times"], grids)
        
        filtered = self._apply_cluster_sieve_v108(quantized, diff_cfg)
        mapped = self._allocate_lanes(filtered, diff_cfg)
        cleaned = self._resolve_overlaps_v108(mapped)
        
        self._save_beatmap(cleaned, diff_name)

    def generate_all(self):
        for d in DIFF_CONFIGS.keys(): self.generate(d)
        print("[GEN] Done.")

    def _quantize(self, pool, beats, grids):
        out = []
        quant_cfg = CONFIG["quantization"]
        for n in pool:
            if len(beats) == 0: out.append(n); continue
            idx = (np.abs(beats - n["time"])).argmin()
            beat_t = beats[idx]
            beat_dur = 0.5 
            if idx < len(beats)-1: beat_dur = beats[idx+1] - beat_t
            
            best_t = n["time"]; min_err = 100
            for div in grids:
                step = beat_dur / (div/4)
                target = round((n["time"] - beat_t) / step) * step + beat_t
                if abs(target - n["time"]) < min_err: min_err = abs(target - n["time"]); best_t = target
            
            if min_err < quant_cfg["quantize_error_threshold"]: n["time"] = n["time"] + (best_t - n["time"]) * quant_cfg["quantize_blend_ratio"]
            out.append(n)
        out.sort(key=lambda x: x["time"])
        return out

    # =========================================================
    #  GAMEPLAY LOGIC
    # =========================================================
    def _apply_cluster_sieve_v108(self, pool, diff_cfg):
        final = []
        duration = self.rhythm_data["duration"]
        max_poly = diff_cfg["poly"]
        soft_limit = diff_cfg["density_cap"] 
        hard_limit = soft_limit * CONFIG["gameplay"]["hard_limit_scale"]
        min_score = diff_cfg["min_score"]

        # 1. Group by Time
        clusters = {}
        for n in pool:
            if n["score"] < min_score: continue
            t = n["time"]
            if t not in clusters: clusters[t] = []
            clusters[t].append(n)
            
        sorted_times = sorted(clusters.keys())
        pruned_clusters = []
        
        # 2. Vertical Pruning
        for t in sorted_times:
            stack = clusters[t]
            stack.sort(key=lambda x: x["score"], reverse=True)
            stack = stack[:max_poly]
            avg_score = sum(n["score"] for n in stack) / len(stack)
            pruned_clusters.append({ "time": t, "notes": stack, "score": avg_score })

        # 3. Horizontal Sieve
        sieve_cfg = CONFIG["sieve"]
        window = sieve_cfg["window_duration"]; cursor = 0.0; idx = 0
        accepted_clusters = []
        
        while cursor < duration:
            candidates = []
            temp_idx = idx
            while temp_idx < len(pruned_clusters) and pruned_clusters[temp_idx]["time"] < cursor + window:
                if pruned_clusters[temp_idx]["time"] >= cursor:
                    candidates.append(pruned_clusters[temp_idx])
                temp_idx += 1
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            current_density = 0
            
            for cluster in candidates:
                count = len(cluster["notes"])
                if current_density + count <= soft_limit:
                    accepted_clusters.append(cluster)
                    current_density += count
                elif current_density + count <= hard_limit:
                    if cluster["score"] > sieve_cfg["high_score_threshold"]: 
                        accepted_clusters.append(cluster)
                        current_density += count
            
            cursor += sieve_cfg["window_step"]
            while idx < len(pruned_clusters) and pruned_clusters[idx]["time"] < cursor: idx += 1

        # 4. Dedup
        unique_map = {}
        for c in accepted_clusters: unique_map[c["time"]] = c["notes"]
        for t in sorted(unique_map.keys()): final.extend(unique_map[t])
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
        lanes_cfg = CONFIG["lanes"]
        
        for t in times:
            stack = groups[t]
            stack.sort(key=lambda x: x["midi"]) 
            assigned = []
            count = len(stack)

            if count == 1:
                n = stack[0]
                src = n["source"]
                if src == "drums":
                    ideal = 1 if n["score"] > lanes_cfg["drum_high_score"] else (0 if last_lane > 1 else 3)
                elif src == "vocals":
                    r_min, r_max = VISUAL_RANGES.get("vocals", (48, 84))
                    norm = max(0.0, min(1.0, (n["midi"] - r_min) / (r_max - r_min)))
                    ideal = int(norm * lanes_cfg["vocal_range_scale"])
                    if ideal > 3: ideal = 3
                else:
                    r_min, r_max = VISUAL_RANGES.get(src, (48, 84))
                    norm = max(0.0, min(1.0, (n["midi"] - r_min) / (r_max - r_min)))
                    ideal = int(norm * (lanes - 1))
                
                if chaos > 0 and np.random.rand() < chaos: ideal = np.random.randint(0, lanes)
                elif ideal == last_lane: ideal = (ideal + 1) % lanes
                assigned.append(ideal)
            else:
                if count == 2: assigned = [0, 3] if last_lane in [1, 2] else [1, 2]
                elif count == 3: assigned = [0, 1, 3] if last_lane == 2 else [0, 2, 3]
                else: assigned = [0, 1, 2, 3][:count]

            for i, n in enumerate(stack):
                lane = assigned[i] if i < len(assigned) else i % lanes
                n["lane"] = lane
                final_notes.append(n)
                last_lane = int(np.mean(assigned))
        return final_notes

    def _resolve_overlaps_v108(self, notes):
        # 1. Flam Doctor
        notes.sort(key=lambda x: x["time"])
        prio_map = CONFIG["mixing"]["prio"]
        accepted = []
        flam_window = CONFIG["gameplay"]["flam_window"]
        
        for n in notes:
            collision_idx = -1
            for j in range(len(accepted) - 1, -1, -1):
                prev = accepted[j]
                if n["time"] - prev["time"] > flam_window: break 
                if prev["lane"] == n["lane"]:
                    collision_idx = j; break
            
            if collision_idx == -1:
                accepted.append(n); continue
                
            prev = accepted[collision_idx]
            p_prev = prio_map.get(prev["source"], 0)
            p_curr = prio_map.get(n["source"], 0)
            
            if p_curr > p_prev: victim, winner, victim_is_prev = prev, n, True
            else: victim, winner, victim_is_prev = n, prev, False
            
            moved = False
            candidates = [l for l in range(4) if l != winner["lane"]]
            for cand_lane in candidates:
                is_free = True
                for check_n in accepted:
                    if abs(check_n["time"] - victim["time"]) < flam_window and check_n["lane"] == cand_lane:
                        is_free = False; break
                if is_free: victim["lane"] = cand_lane; moved = True; break
            
            if moved:
                if not victim_is_prev: accepted.append(n)
            else:
                if victim_is_prev: accepted.pop(collision_idx); accepted.append(n)
        
        # 2. Hold Truncation
        accepted.sort(key=lambda x: x["time"])
        lane_queues = {i: [] for i in range(4)}
        for n in accepted: lane_queues[n["lane"]].append(n)
        
        cleaned = []
        gap = CONFIG["holds"]["gap_buffer"]
        min_dur = CONFIG["holds"]["min_dur"]
        
        for lane, lane_notes in lane_queues.items():
            if not lane_notes: continue
            for i in range(len(lane_notes)):
                current = lane_notes[i]
                if current.get("type") == "hold":
                    if i + 1 < len(lane_notes):
                        next_note = lane_notes[i+1]
                        hold_end = current["time"] + current["dur"]
                        limit = next_note["time"] - gap
                        if hold_end > limit:
                            new_dur = limit - current["time"]
                            if new_dur < min_dur: current["dur"] = 0; current["type"] = "tap"
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
                "time": round(n["time"], 3),
                "lane": int(n["lane"]),
                "dur": round(n["dur"], 3),
                "type": n["type"],
                "midi": int(n["midi"]),
                "source": n["source"]
            })
        with open(output_file, 'w') as f: json.dump(out, f, indent=2)
        print(f"[SAVE] {output_file} ({len(out)} notes)")

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
        gen.generate_all()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GEN] Critical Error: {e}")