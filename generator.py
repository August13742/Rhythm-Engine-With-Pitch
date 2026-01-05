"""
V204.4 - GENERATOR ENGINE (PITCH STABILITY LOGIC)
Changes:
  - Hold Logic Upgrade: "Sustained Tonal Stability"
    - Holds are now cut if Pitch drifts > 1.5 semitones (Handles Runs/Melisma).
    - Holds are cut if Energy drops (Handles Staccato/Speech).
  - Config: Added 'max_pitch_drift' parameter.
"""

import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import json
import os
import argparse
import torch
import torchcrepe

# ==========================================
#        TUNING & CONFIGURATION
# ==========================================

CONSTANTS = {
    "system": {
        "global_offset_sec": -0.02,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "audio": {
        "sr": 44100,
        "crepe_sr": 16000,
        "hop_length": 512,
        
        "pyin": {
            "frame_length": 4096, 
            "fmin": 40,
            "fmax": 2000,
            "confidence": 0.15 
        },
        
        "crepe": {
            "model": "full",
            "hop_length": 160,
            "confidence": 0.60,
            "batch_size": 2048
        },
        
        "filters": {
            "default": {"hp": 60, "lp": 14000, "order": 8},
            "bass":    {"hp": 25, "lp": 8000,  "order": 8},
            "drums":   {"hp": 25, "lp": 18000, "order": 8},
        }
    },
    
    "consensus": {
        "attack_skip": 0.030,   
        "sustain_win": 0.120,   
        "min_valid_ratio": 0.3, 
    },
    
    "harvest": {
        "sens": {
            "vocals": 0.05, "vocals_lead": 0.05, "vocals_backing": 0.08,
            "drums": 0.10, "bass": 0.10,
            "piano": 0.05, "guitar": 0.04, "other": 0.08
        }
    },
    
    "diarization": {
        "reset_time": 4.0,
        "max_jump": 7,
        "backing_gate": 0.35 
    },
    
    "holds": {
        "energy_decay": 0.60,
        "tap_threshold": 0.15,
        "max_dur": 5.0,
        "gap_buffer": 0.10,
        
        # [NEW] Pitch Stability
        "max_pitch_drift": 1.5, # Semitones. Cuts hold if pitch shifts (Runs/Melisma).
        
        "allowed_stems": ["vocals", "vocals_lead", "vocals_backing", "other"]
    },
    
    "mixing": {
        "stem_vol": {
            "vocals": 1.25, "vocals_lead": 1.25, "vocals_backing": 1.0,
            "drums": 0.90, "bass": 0.85, 
            "piano": 0.90, "guitar": 0.85, "other": 0.60
        },
        "priorities": {
            "vocals": 2.0, "vocals_lead": 2.0, "vocals_backing": 1.5,
            "drums": 1.5, "bass": 1.2, 
            "piano": 1.1, "guitar": 1.1, "other": 0.8
        },
        "vol_curve": 0.2
    },
    
    "visuals": {
        "ranges": {
            "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
            "bass": (36, 60), "other": (48, 88)
        }
    },
    
    "difficulty": {
        "EASY":   {"lanes": 4, "grids": [4],       "poly": 1, "density": 2.0, "min_score": 0.60, "chaos": 0.0},
        "NORMAL": {"lanes": 4, "grids": [4, 8],    "poly": 2, "density": 4.0, "min_score": 0.50, "chaos": 0.0},
        "HARD":   {"lanes": 4, "grids": [4, 8, 12, 16], "poly": 3, "density": 6.0, "min_score": 0.35, "chaos": 0.1},
        "INSANE": {"lanes": 4, "grids": [4, 8, 12, 16, 24], "poly": 4, "density": 8.0, "min_score": 0.20, "chaos": 0.2},
    }
}

DIFF_CONFIGS = CONSTANTS["difficulty"]

class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[INIT] V204.4 Generator (Pitch-Stable Holds)...")
        self.paths = stems_path
        self.use_holds = use_holds
        self.cfg = CONSTANTS
        self.manifest = self._load_manifest()
        
        # 1. Load Audio
        self.audio_data = self._load_audio()
        
        # 2. Envelopes
        self.envs = {}
        self.rms = {}
        self._generate_envelopes()
        
        # 3. Mode Check
        self.choir_mode = self._evaluate_vocal_mode()
        
        # 4. Pitch Inference
        self.vocal_maps = {}
        self._infer_vocal_maps()
            
        # 5. Rhythm
        self.rhythm = self._analyze_rhythm()
        
        # 6. Pipeline
        self.raw_notes = self._harvest_all()
        self.master_pool = self._score_and_sort(self.raw_notes)
        
        print(f"[INIT] Ready. Master Pool: {len(self.master_pool)} events.")

    def _evaluate_vocal_mode(self):
        manifest_mode = self.manifest.get("vocal_type", "monophonic")
        if manifest_mode == "monophonic": return False
        
        if "vocals_backing" in self.envs:
            peak = np.max(self.envs["vocals_backing"])
            if peak < 0.05: 
                print(f"[WARN] Backing stem too weak ({peak:.3f}). Downgrading to MONOPHONIC.")
                return False
        
        print(f"[MODE] Polyphonic/Choir Mode CONFIRMED.")
        return True

    def _load_audio(self):
        loaded = {}
        max_len = 0
        sr = self.cfg["audio"]["sr"]
        filters = self.cfg["audio"]["filters"]
        
        print("[DSP] Loading and Filtering Stems...")
        
        for name, path in self.paths.items():
            info = self.manifest.get(name, {"exists": False})
            if not info.get("exists", False) or not os.path.exists(path):
                loaded[name] = None
                continue
            
            if info.get("is_silent", True):
                loaded[name] = None
                continue

            y, _ = librosa.load(path, sr=sr, mono=True)
            
            f_key = "bass" if "bass" in name else ("drums" if "drums" in name else "default")
            f_set = filters[f_key]
            
            sos_hp = scipy.signal.butter(f_set["order"], f_set["hp"], 'hp', fs=sr, output='sos')
            sos_lp = scipy.signal.butter(f_set["order"], f_set["lp"], 'lp', fs=sr, output='sos')
            y = scipy.signal.sosfilt(sos_hp, y)
            y = scipy.signal.sosfilt(sos_lp, y)
            
            peak = np.max(np.abs(y))
            if peak > 0: y /= peak
            
            loaded[name] = y
            max_len = max(max_len, len(y))
            
        final = {}
        for name, data in loaded.items():
            if data is None: final[name] = np.zeros(max_len, dtype=np.float32)
            elif len(data) < max_len:
                padded = np.zeros(max_len, dtype=np.float32)
                padded[:len(data)] = data
                final[name] = padded
            else: final[name] = data
        return final

    def _generate_envelopes(self):
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        
        for name, y in self.audio_data.items():
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
            if rms.max() > 0: rms /= rms.max()
            self.rms[name] = rms
            
            if "vocals" in name:
                onset = librosa.onset.onset_strength(y=y, sr=sr)
                if onset.max() > 0: onset /= onset.max()
                self.envs[name] = (onset * 0.5) + (rms * 0.5)
            elif name == "drums":
                y_h, y_p = librosa.effects.hpss(y)
                self.envs[name] = librosa.onset.onset_strength(y=y_p, sr=sr)
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=sr)
                
            if self.envs[name].max() > 0:
                self.envs[name] /= self.envs[name].max()

    def _infer_vocal_maps(self):
        targets = []
        if self._check_stem("vocals_lead"): targets.append("vocals_lead")
        elif self._check_stem("vocals"): targets.append("vocals")
        
        for t in targets:
            self.vocal_maps[t] = self._infer_single_map(t)

    def _infer_single_map(self, source):
        print(f"[AI] Running Crepe Viterbi on {source}...")
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        crepe_sr = self.cfg["audio"]["crepe_sr"]
        c_cfg = self.cfg["audio"]["crepe"]

        # [NEW] Cross-Stem Cleaning (Anti-Bleed)
        # If we are analyzing the Lead, and a Backing track exists, 
        # dampen the Lead signal whenever the Backing is highly active.
        if source in ["vocals", "vocals_lead"] and "vocals_backing" in self.audio_data:
            y_back = self.audio_data["vocals_backing"]
            
            # Calculate RMS (Energy) of backing track to find where it is singing
            frame_len = 2048
            hop = 512
            rms_back = librosa.feature.rms(y=y_back, frame_length=frame_len, hop_length=hop)[0]
            
            # Stretch RMS map to match exact audio sample length
            zoom_factor = len(y) / len(rms_back)
            rms_back_s = scipy.ndimage.zoom(rms_back, zoom_factor, order=1)
            
            # Create a "Ducking Mask"
            # Logic: If Backing RMS is high (singing), multiply Lead audio by a low number (e.g. 0.3).
            # This lowers Crepe's confidence in those sections, effectively filtering out bleed.
            # We clip the mask so it doesn't silence the lead entirely, just dampens it.
            mask = 1.0 - np.clip(rms_back_s * 1.5, 0.0, 0.7)
            y = y * mask
            print(f"    > Applied Anti-Bleed Masking using vocals_backing.")

        y_16k = librosa.resample(y, orig_sr=sr, target_sr=crepe_sr)
        dev = self.cfg["system"]["device"]
        audio = torch.tensor(y_16k, device=dev).float().unsqueeze(0)
        
        # [TUNING] Increased confidence threshold implies we want stricter detection
        pitch = torchcrepe.predict(
            audio, sample_rate=crepe_sr, hop_length=c_cfg["hop_length"],
            fmin=50, fmax=1200, model=c_cfg["model"],
            batch_size=c_cfg["batch_size"], device=dev, decoder=torchcrepe.decode.viterbi
        )
        _, periodicity = torchcrepe.predict(
            audio, sample_rate=crepe_sr, hop_length=c_cfg["hop_length"],
            fmin=50, fmax=1200, model=c_cfg["model"],
            batch_size=c_cfg["batch_size"], device=dev, return_periodicity=True,
            decoder=torchcrepe.decode.weighted_argmax
        )
        return {
            "f0": pitch.squeeze(0).cpu().numpy(),
            "conf": periodicity.squeeze(0).cpu().numpy(),
            "time_step": c_cfg["hop_length"] / float(crepe_sr)
        }

    # =========================================================
    #   PHASE 2: HARVESTING
    # =========================================================

    def _harvest_all(self):
        pool = []
        sens = self.cfg["harvest"]["sens"]
        
        # 1. Vocals
        main_src = "vocals_lead" if self._check_stem("vocals_lead") else "vocals"
        if self._check_stem(main_src):
            pool.extend(self._harvest_vocals_crepe(main_src, sens.get(main_src, 0.05)))
            
        if self.choir_mode and self._check_stem("vocals_backing"):
            pool.extend(self._harvest_polyphonic("vocals_backing", sens["vocals_backing"]))
        
        # 2. Instruments
        for inst in ["bass", "piano", "guitar", "other"]:
            if self._check_stem(inst):
                pool.extend(self._harvest_inst_v108(inst, sens[inst]))
        
        # 3. Drums
        if self._check_stem("drums"):
            pool.extend(self._harvest_drums(sens["drums"]))
            
        return pool

    def _harvest_vocals_crepe(self, source, sensitivity):
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=2)
        
        notes = []
        pmap = self.vocal_maps[source]
        c_cfg = self.cfg["consensus"]
        
        # [NEW] Load Backing Envelope for Vetoing
        backing_env = None
        if "vocals_backing" in self.envs:
            backing_env = self.envs["vocals_backing"]
        
        voices = [None, None]
        last_times = [-999.0, -999.0]
        
        busy_until = 0.0
        
        for i, f in enumerate(frames):
            # [NEW] The Veto Check
            # If the Backing envelope is strong (>0.5) AND significantly stronger than the Lead envelope,
            # ignore this onset. It is likely harmonic bleed.
            if backing_env is not None:
                if f < len(backing_env):
                    b_val = backing_env[f]
                    l_val = env[f]
                    # If Backing is 'singing' (0.5) and Lead is 'quiet' (<0.5), skip.
                    # Or if Backing is purely overpowering Lead (Ratio > 1.2).
                    if (b_val > 0.5 and l_val < 0.5) or (b_val > l_val * 1.5):
                        continue

            t_onset = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t_onset < 0: continue
            
            # Lock Check
            if t_onset < busy_until: continue 
            
            t_next = 9999.0
            if i + 1 < len(frames):
                t_next = librosa.frames_to_time(frames[i+1], sr=sr) + self.cfg["system"]["global_offset_sec"]
            
            t_start = t_onset + c_cfg["attack_skip"]
            max_end = t_start + c_cfg["sustain_win"]
            safe_end = t_next - 0.010 
            t_end = min(max_end, safe_end)
            if t_end <= t_start: t_end = t_start + 0.02
            
            midi_val, conf = self._sample_crepe_map(pmap, t_start, t_end)
            
            # [TUNING] Strict Confidence Check
            # Since we masked the audio in _infer_single_map, bleed sections will have lower confidence.
            # We raise the requirement (0.4 -> 0.5 or 0.6) to ensure we drop those sections.
            required_conf = 0.55 if backing_env is not None else self.cfg["audio"]["crepe"]["confidence"]
            
            if midi_val is None or conf < required_conf: continue
            
            voice_id = 0
            if not self.choir_mode:
                voice_id, voices, last_times = self._solve_diarization(midi_val, t_onset, voices, last_times)
            
            # [PITCH STABLE DURATION]
            dur = self._measure_duration(source, f, pmap=pmap, start_pitch=midi_val)
            
            if dur > 0: busy_until = t_onset + dur + 0.05
                
            notes.append({
                "time": t_onset, "midi": midi_val, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": voice_id
            })
        return notes

    def _harvest_inst_v108(self, source, sensitivity):
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        p_cfg = self.cfg["audio"]["pyin"]
        
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=4)
        notes = []
        busy_until = 0.0
        
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0: continue
            if t < busy_until: continue
            
            start_samp = int(f * hop)
            chunk = y[start_samp : start_samp + p_cfg["frame_length"]]
            if len(chunk) < p_cfg["frame_length"] // 2: continue
            
            f0, _, prob = librosa.pyin(
                chunk, fmin=p_cfg["fmin"], fmax=p_cfg["fmax"], 
                sr=sr, frame_length=p_cfg["frame_length"], fill_na=np.nan
            )
            mask = prob > p_cfg["confidence"]
            valid_f0 = f0[mask]
            valid_f0 = valid_f0[~np.isnan(valid_f0)]
            
            if len(valid_f0) == 0: continue
            
            midi = int(round(librosa.hz_to_midi(np.median(valid_f0))))
            
            # Instruments don't have global pmap, rely on RMS decay
            dur = self._measure_duration(source, f)
            
            if dur > 0: busy_until = t + dur + 0.05
            
            notes.append({
                "time": t, "midi": midi, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": 0
            })
        return notes

    def _harvest_polyphonic(self, source, sensitivity):
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        y = self.audio_data[source]
        hop = self.cfg["audio"]["hop_length"]
        
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=2)
        
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop)[0]
        cent_log = np.log1p(cent)
        c_min, c_max = np.min(cent_log), np.max(cent_log)
        
        notes = []
        busy_until = 0.0
        
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0: continue
            if t < busy_until: continue
            
            val = (cent_log[min(f, len(cent_log)-1)] - c_min) / (c_max - c_min) if (c_max > c_min) else 0.5
            mock_midi = 48 + int(val * 36)
            
            if env[f] < self.cfg["diarization"]["backing_gate"]: continue
            dur = self._measure_duration(source, f)
            
            if dur > 0: busy_until = t + dur + 0.05
            
            notes.append({
                "time": t, "midi": mock_midi, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": 1
            })
        return notes

    def _harvest_drums(self, sensitivity):
        source = "drums"
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        notes = []
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0: continue
            notes.append({
                "time": t, "midi": 36, "dur": 0.0, 
                "source": source, "score": float(env[f]), "voice_id": 0
            })
        return notes

    # --- UTILS ---
    def _sample_crepe_map(self, pmap, t_start, t_end):
        idx_s = int(t_start / pmap["time_step"])
        idx_e = max(int(t_end / pmap["time_step"]), idx_s + 1)
        if idx_s >= len(pmap["f0"]): return None, 0.0
        idx_e = min(idx_e, len(pmap["f0"]))
        
        f0 = pmap["f0"][idx_s:idx_e]
        conf = pmap["conf"][idx_s:idx_e]
        mask = (conf > self.cfg["audio"]["crepe"]["confidence"]) & (~np.isnan(f0))
        if np.sum(mask) == 0: return None, 0.0
        
        v_f0 = f0[mask]
        v_conf = conf[mask]
        sort_idx = np.argsort(v_f0)
        cumsum = np.cumsum(v_conf[sort_idx])
        mid = np.searchsorted(cumsum, cumsum[-1]/2.0)
        return int(round(librosa.hz_to_midi(v_f0[sort_idx][mid]))), np.mean(v_conf)

    def _solve_diarization(self, midi, t, voices, last_times):
        d_cfg = self.cfg["diarization"]
        silence = [t - last_times[0], t - last_times[1]]
        stale = [s > d_cfg["reset_time"] for s in silence]
        dist = [abs(midi - (voices[0] or 999)), abs(midi - (voices[1] or 999))]
        
        chosen = 0
        if stale[0] and stale[1]: chosen = 0
        elif not stale[0] and stale[1]: chosen = 1 if dist[0] > d_cfg["max_jump"] else 0
        elif not stale[1] and stale[0]: chosen = 0 if dist[1] > d_cfg["max_jump"] else 1
        else: chosen = 0 if dist[0] <= dist[1] else 1
        voices[chosen] = midi
        last_times[chosen] = t
        return chosen, voices, last_times

    def _measure_duration(self, source, start_frame, pmap=None, start_pitch=None):
        """
        Measures hold duration based on:
        1. RMS Decay (All instruments)
        2. Pitch Stability (Vocals Only - Stops hold if pitch shifts)
        """
        if not self.use_holds: return 0.0
        
        allowed = self.cfg["holds"]["allowed_stems"]
        if not any(a in source for a in allowed): return 0.0
        
        rms = self.rms[source]
        h_cfg = self.cfg["holds"]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        
        if start_frame >= len(rms): return 0.0
        threshold = rms[start_frame] * h_cfg["energy_decay"]
        
        curr = start_frame + 1
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        end = min(len(rms), start_frame + max_dist)
        
        # PITCH DRIFT PRE-CALC
        # If we have a pmap (Vocals), we use it to cut holds on runs/melisma
        check_pitch = (pmap is not None and start_pitch is not None)
        crepe_hop = self.cfg["audio"]["crepe"]["hop_length"]
        
        while curr < end:
            # 1. Energy Check
            if rms[curr] < threshold: break
            
            # 2. Pitch Drift Check (Vocals only)
            if check_pitch:
                # Convert RMS frame (512 hop) to Crepe frame (160 hop)
                t_curr = librosa.frames_to_time(curr, sr=sr, hop_length=hop)
                c_idx = int(t_curr * 16000 / crepe_hop)
                
                if c_idx < len(pmap["f0"]):
                    f0 = pmap["f0"][c_idx]
                    conf = pmap["conf"][c_idx]
                    
                    if conf > 0.4 and not np.isnan(f0):
                        curr_midi = librosa.hz_to_midi(f0)
                        if abs(curr_midi - start_pitch) > h_cfg["max_pitch_drift"]:
                            # Pitch changed too much -> Stop Hold
                            break
            
            curr += 1
            
        dur = librosa.frames_to_time(curr - start_frame, sr=sr, hop_length=hop)
        return dur if dur >= h_cfg["tap_threshold"] else 0.0

    def generate_all(self):
        for diff in self.cfg["difficulty"]:
            self._generate_chart(diff)

    def _generate_chart(self, diff_name):
        print(f"[GEN] Processing {diff_name}...")
        d_cfg = self.cfg["difficulty"][diff_name]
        q = self._quantize(self.master_pool, d_cfg["grids"])
        s = self._sieve_density(q, d_cfg)
        m = self._allocate_lanes(s, d_cfg["lanes"], d_cfg["chaos"])
        f = self._resolve_physics(m)
        self._export_json(f, diff_name)

    def _quantize(self, pool, grids):
        beats = self.rhythm["beat_times"]
        out = []
        for n in pool:
            t = n["time"]
            if len(beats) < 2: out.append(n); continue
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            b_dur = (beats[idx+1]-beat_t) if idx<len(beats)-1 else (beat_t-beats[idx-1])
            
            best_t = t
            min_dist = 999.0
            for div in grids:
                step = b_dur/(div/4.0)
                cand = beat_t + round((t-beat_t)/step)*step
                d = abs(cand-t)
                if d<min_dist: min_dist=d; best_t=cand
            
            if min_dist < 0.06:
                n["time"] = best_t
                if n["dur"] > 0:
                    sixteenth = b_dur/4.0
                    n["dur"] = round(n["dur"]/sixteenth)*sixteenth
            out.append(n)
        return sorted(out, key=lambda x: x["time"])

    def _sieve_density(self, pool, d_cfg):
        buckets = {}
        for n in pool:
            s = int(n["time"])
            if s not in buckets: buckets[s] = []
            buckets[s].append(n)
        final = []
        for s in sorted(buckets.keys()):
            candidates = buckets[s]
            if s+1 in buckets: candidates.extend(buckets[s+1])
            window = [c for c in candidates if s <= c["time"] < s+1 and c["score"] >= d_cfg["min_score"]]
            window.sort(key=lambda x: x["score"], reverse=True)
            count = 0
            for node in window:
                if count < d_cfg["density"]:
                    if node not in final: final.append(node)
                    count += 1
        return sorted(final, key=lambda x: x["time"])

    def _allocate_lanes(self, pool, lanes, chaos):
        out = []
        r_cfg = self.cfg["visuals"]["ranges"]
        m_cfg = self.cfg["mixing"]
        last_lane = 0
        for n in pool:
            src = n["source"]
            target = 0
            if "vocals" in src:
                norm = np.clip((n["midi"]-48)/36, 0.0, 1.0)
                if n["voice_id"]==0: target = int(norm*2.5)
                else: target = min(3, 1+int(norm*2.5))
            else:
                low, high = r_cfg.get(src, r_cfg["other"])
                norm = np.clip((n["midi"]-low)/(high-low), 0.0, 1.0)
                target = int(norm*(lanes-1))
            
            if chaos>0 and np.random.rand()<chaos: target = np.random.randint(0, lanes)
            elif target==last_lane and "vocals" not in src: target=(target+1)%lanes
            
            base_vol = m_cfg["stem_vol"].get(src, 0.8)
            if "lead" in src: base_vol = m_cfg["stem_vol"]["vocals_lead"]
            elif "backing" in src: base_vol = m_cfg["stem_vol"]["vocals_backing"]
            
            n["vol"] = np.clip(base_vol*(0.8+n["score"]*0.3), 0.0, 1.0)
            n["lane"] = target
            n["type"] = "hold" if n["dur"]>0 else "tap"
            out.append(n)
            last_lane = target
        return out

    def _resolve_physics(self, notes):
        notes.sort(key=lambda x: x["time"])
        cleaned = []
        lane_end = {i: -1.0 for i in range(4)}
        for n in notes:
            l = n["lane"]
            if n["time"] < lane_end[l]+0.05:
                moved = False
                for c in [0,1,2,3]:
                    if c!=l and n["time"] >= lane_end[c]+0.05:
                        n["lane"]=c; lane_end[c]=n["time"]+n["dur"]
                        cleaned.append(n); moved=True; break
            else:
                lane_end[l]=n["time"]+n["dur"]
                cleaned.append(n)
        
        cleaned.sort(key=lambda x: x["time"])
        final = []
        per_lane = {i:[] for i in range(4)}
        for n in cleaned: per_lane[n["lane"]].append(n)
        gap = self.cfg["holds"]["gap_buffer"]
        for l in per_lane:
            stack = per_lane[l]
            for i in range(len(stack)):
                c = stack[i]
                if c["type"]=="hold" and i+1<len(stack):
                    lim = stack[i+1]["time"] - gap
                    if c["time"]+c["dur"] > lim:
                        c["dur"] = max(0.0, lim - c["time"])
                        if c["dur"] < self.cfg["holds"]["tap_threshold"]: 
                            c["dur"]=0; c["type"]="tap"
                final.append(c)
        return sorted(final, key=lambda x: x["time"])

    def _score_and_sort(self, pool):
        beat_arr = np.array(self.rhythm["beat_times"])
        prio = self.cfg["mixing"]["priorities"]
        for n in pool:
            src = n["source"]
            if "vocals" in src and src not in prio: src="vocals"
            n["score"] *= prio.get(src, 1.0)
            if len(beat_arr)>0 and np.min(np.abs(beat_arr-n["time"])) < 0.05: n["score"] *= 1.25
            if n["dur"]>0: n["score"] *= 1.1
        return sorted(pool, key=lambda x: x["time"])

    def _analyze_rhythm(self):
        y = np.zeros_like(next(iter(self.audio_data.values())))
        if self._check_stem("drums"): y += self.audio_data["drums"]
        if self._check_stem("bass"): y += self.audio_data["bass"]
        sr = self.cfg["audio"]["sr"]
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
        return {"beat_times": librosa.frames_to_time(beats, sr=sr)}

    def _check_stem(self, name):
        return (name in self.audio_data and self.audio_data[name] is not None)

    def _load_manifest(self):
        for path in self.paths.values():
            if os.path.exists(path):
                d = os.path.dirname(path)
                p = os.path.join(d, "stems_manifest.json")
                if os.path.exists(p):
                    with open(p, 'r') as f: return json.load(f)
        return {k: {"exists": True} for k in self.paths}

    def _export_json(self, notes, diff_name):
        folder = "."
        for path in self.paths.values():
            if os.path.exists(path):
                folder = os.path.dirname(path)
                break
        
        base = os.path.basename(folder)
        fname = os.path.join(folder, f"{base}_{diff_name}.json")
        out = []
        for n in notes:
            out.append({
                "time": float(f"{n['time']:.3f}"), "lane": int(n["lane"]),
                "dur": float(f"{n['dur']:.3f}"), "type": n["type"],
                "midi": int(n["midi"]), "score": float(f"{n['score']:.3f}"),
                "source": n["source"], "vol": float(f"{n['vol']:.3f}")
            })
        with open(fname, 'w') as f: json.dump(out, f, indent=2)
        print(f"[EXPORT] Saved {len(out)} notes -> {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to stems folder")
    parser.add_argument("--holds", action="store_true", help="Enable hold notes")
    args = parser.parse_args()
    
    stems = {
        "vocals": os.path.join(args.folder, "vocals.wav"),
        "vocals_lead": os.path.join(args.folder, "vocals_lead.wav"),
        "vocals_backing": os.path.join(args.folder, "vocals_backing.wav"),
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
        print(f"[FATAL] {e}")