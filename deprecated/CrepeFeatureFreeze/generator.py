"""
V205 - HYBRID ENGINE (Best of V108 Instruments + V204 Vocals)
Fixes:
  - Reverted aggressive audio filtering (restores Instrument harmonics).
  - Restored V108 "Psychoacoustic Coincidence Voting" (fixes ghost notes).
  - Restored V108 "Magnetic Quantization" (preserves human timing).
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
        
        # V108 Config for Instruments
        "pyin": {
            "frame_length": 4096, 
            "fmin": 40,
            "fmax": 2000,
            "confidence": 0.15 # Lower confidence allowed, cleaned by Voting later
        },
        
        # V204 Config for Vocals
        "crepe": {
            "model": "full",
            "hop_length": 160,
            "confidence": 0.55, # Stricter for vocals
            "batch_size": 2048
        }
    },
    
    "consensus": {
        "attack_skip": 0.030,   
        "sustain_win": 0.120,   
        "min_valid_ratio": 0.3, 
    },
        
    # RESTORED V108 VOTING LOGIC
    "coincidence": {
        "window": 0.05,          
        "min_support": 0.3,      
        "boost_scale": 0.5,      
        "mask_threshold": 0.35,   
        "mask_ratio": 2.0        
    },
    
    "harvest": {
        # Lower thresholds (V108 style) to catch everything, then filter
        "sens": {
            "vocals": 0.05, "vocals_lead": 0.05, "vocals_backing": 0.08,
            "drums": 0.10, "bass": 0.08, # Lowered from 0.10
            "piano": 0.04, "guitar": 0.03, "other": 0.06 # Lowered
        }
    },
    
    "holds": {
        "energy_decay": 0.60,
        "tap_threshold": 0.15,
        "max_dur": 5.0,
        "gap_buffer": 0.10,
        "max_pitch_drift": 1.5,
        "allowed_stems": ["vocals", "vocals_lead", "vocals_backing", "other"]
    },
    
    "mixing": {
        "stem_vol": {
            "vocals": 1.25, "vocals_lead": 1.25, "vocals_backing": 1.0,
            "drums": 0.90, "bass": 0.85, 
            "piano": 0.90, "guitar": 0.85, "other": 0.60
        },
        "priorities": { # Used for Voting Weights
            "vocals": 2.5, "vocals_lead": 2.5, "vocals_backing": 1.2,
            "drums": 1.1, "bass": 0.9, 
            "piano": 1.1, "guitar": 1.1, "other": 1.0
        }
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
        print(f"[INIT] V205 Hybrid Generator...")
        self.paths = stems_path
        self.use_holds = use_holds
        self.cfg = CONSTANTS
        self.manifest = self._load_manifest()
        
        # 1. Load Audio (V108 Style: No aggressive filters)
        self.audio_data = self._load_audio()
        
        # 2. Envelopes
        self.envs = {}
        self.rms = {}
        self._generate_envelopes()
        
        # 3. Mode Check
        self.choir_mode = self._evaluate_vocal_mode()
        
        # 4. Pitch Inference (V204 Logic for Vocals)
        self.vocal_maps = {}
        self._infer_vocal_maps()
            
        # 5. Rhythm
        self.rhythm = self._analyze_rhythm()
        
        # 6. Pipeline
        self.raw_notes = self._harvest_all()
        
        # 7. V108 RESTORATION: Coincidence Voting
        self.voted_pool = self._apply_coincidence_voting(self.raw_notes)
        
        self.master_pool = self._score_and_sort(self.voted_pool)
        print(f"[INIT] Ready. Master Pool: {len(self.master_pool)} events.")

    def _evaluate_vocal_mode(self):
        manifest_mode = self.manifest.get("vocal_type", "monophonic")
        return manifest_mode != "monophonic"

    def _load_audio(self):
        # REVERTED TO SIMPLE LOADING (V108)
        # Removing the heavy Butterworth filters helps PyIn detect instrument pitch better.
        loaded = {}
        max_len = 0
        sr = self.cfg["audio"]["sr"]
        
        print("[DSP] Loading Stems (Raw Fidelity)...")
        
        for name, path in self.paths.items():
            info = self.manifest.get(name, {"exists": False})
            if not info.get("exists", False) or not os.path.exists(path):
                loaded[name] = None
                continue
            
            if info.get("is_silent", True):
                loaded[name] = None
                continue

            y, _ = librosa.load(path, sr=sr, mono=True)
            
            # Simple normalization only
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
        # KEEPING V204 CREPE (Superior for Vocals)
        targets = []
        if self._check_stem("vocals_lead"): targets.append("vocals_lead")
        elif self._check_stem("vocals"): targets.append("vocals")
        
        for t in targets:
            self.vocal_maps[t] = self._infer_single_map(t)

    def _infer_single_map(self, source):
        print(f"[AI] Running Crepe on {source}...")
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        crepe_sr = self.cfg["audio"]["crepe_sr"]
        c_cfg = self.cfg["audio"]["crepe"]

        # Anti-Bleed Masking (V204 feature - worth keeping)
        if source in ["vocals", "vocals_lead"] and "vocals_backing" in self.audio_data:
            y_back = self.audio_data["vocals_backing"]
            frame_len = 2048; hop = 512
            rms_back = librosa.feature.rms(y=y_back, frame_length=frame_len, hop_length=hop)[0]
            zoom_factor = len(y) / len(rms_back)
            rms_back_s = scipy.ndimage.zoom(rms_back, zoom_factor, order=1)
            mask = 1.0 - np.clip(rms_back_s * 1.5, 0.0, 0.7)
            y = y * mask

        y_16k = librosa.resample(y, orig_sr=sr, target_sr=crepe_sr)
        dev = self.cfg["system"]["device"]
        audio = torch.tensor(y_16k, device=dev).float().unsqueeze(0)
        
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
    #   PHASE 2: HARVESTING (Hybrid)
    # =========================================================

    def _harvest_all(self):
        pool = []
        sens = self.cfg["harvest"]["sens"]
        
        # 1. Vocals (Crepe)
        main_src = "vocals_lead" if self._check_stem("vocals_lead") else "vocals"
        if self._check_stem(main_src):
            pool.extend(self._harvest_vocals_crepe(main_src, sens.get(main_src, 0.05)))
            
        if self.choir_mode and self._check_stem("vocals_backing"):
            pool.extend(self._harvest_polyphonic("vocals_backing", sens["vocals_backing"]))
        
        # 2. Instruments (Restored V108 Logic with PyIn)
        for inst in ["bass", "piano", "guitar", "other"]:
            if self._check_stem(inst):
                pool.extend(self._harvest_inst_pyin(inst, sens[inst]))
        
        # 3. Drums
        if self._check_stem("drums"):
            pool.extend(self._harvest_drums(sens["drums"]))
            
        return pool

    def _harvest_inst_pyin(self, source, sensitivity):
        # RESTORED V108 IMPLEMENTATION
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        p_cfg = self.cfg["audio"]["pyin"]
        
        # V108 used longer wait times to prevent trills from spamming
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=5)
        
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
            
            # Median pitch is safer for instruments than mean
            midi = int(round(librosa.hz_to_midi(np.median(valid_f0))))
            
            dur = self._measure_duration(source, f)
            if dur > 0: busy_until = t + dur + 0.05
            
            notes.append({
                "time": t, "midi": midi, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": 0
            })
        return notes

    def _harvest_vocals_crepe(self, source, sensitivity):
        # V204 Logic
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=2)
        
        notes = []
        pmap = self.vocal_maps[source]
        c_cfg = self.cfg["consensus"]
        
        busy_until = 0.0
        
        for i, f in enumerate(frames):
            t_onset = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t_onset < 0: continue
            if t_onset < busy_until: continue 
            
            # Look ahead for next onset to define window
            t_next = 9999.0
            if i + 1 < len(frames):
                t_next = librosa.frames_to_time(frames[i+1], sr=sr) + self.cfg["system"]["global_offset_sec"]
            
            t_start = t_onset + 0.030
            max_end = t_start + 0.120
            safe_end = t_next - 0.010 
            t_end = min(max_end, safe_end)
            if t_end <= t_start: t_end = t_start + 0.02
            
            midi_val, conf = self._sample_crepe_map(pmap, t_start, t_end)
            
            if midi_val is None or conf < self.cfg["audio"]["crepe"]["confidence"]: continue
            
            voice_id = 0
            # Diarization logic skipped for simplicity in hybrid, focusing on pitch accuracy
            
            dur = self._measure_duration(source, f, pmap=pmap, start_pitch=midi_val)
            if dur > 0: busy_until = t_onset + dur + 0.05
                
            notes.append({
                "time": t_onset, "midi": midi_val, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": voice_id
            })
        return notes

    def _harvest_polyphonic(self, source, sensitivity):
        # Approximate backing vocals via spectral centroid (V204)
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
            if t < 0 or t < busy_until: continue
            
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

    # --- VOTING SYSTEM (FROM V108) ---
    def _apply_coincidence_voting(self, pool):
        """
        V108 LEGACY: Cross-Stem Coincidence Voting.
        This fixes the 'noise' from pyin by checking if other instruments support the note.
        """
        print("[GEN] Applying Psychoacoustic Voting (V108 Logic)...")
        v_cfg = self.cfg["coincidence"]
        priorities = self.cfg["mixing"]["priorities"]
        
        pool.sort(key=lambda x: x["time"])
        n_notes = len(pool)
        
        for i in range(n_notes):
            current = pool[i]
            t = current["time"]
            src = current["source"]
            
            support_score = 0.0
            masking_energy = 0.0
            
            # Search neighbors window
            # Backward
            j = i - 1
            while j >= 0:
                neighbor = pool[j]
                dt = t - neighbor["time"]
                if dt > v_cfg["window"]: break 
                if neighbor["source"] != src:
                    w = priorities.get(neighbor["source"], 0.8)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                j -= 1
            # Forward
            k = i + 1
            while k < n_notes:
                neighbor = pool[k]
                dt = neighbor["time"] - t
                if dt > v_cfg["window"]: break 
                if neighbor["source"] != src:
                    w = priorities.get(neighbor["source"], 0.8)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                k += 1

            # Logic 1: SYNC BOOST
            # If bass and drums hit together, both get boosted.
            if support_score > v_cfg["min_support"]:
                boost = 1.0 + (support_score * v_cfg["boost_scale"])
                current["score"] *= boost
            
            # Logic 2: MASKING
            # If a note is weak and surrounded by loud noise from other stems, kill it.
            elif current["score"] < v_cfg["mask_threshold"]:
                if src == "vocals": pass # Vocals are immune
                elif masking_energy > (current["score"] * v_cfg["mask_ratio"]):
                    current["score"] *= 0.1 # Soft delete
        
        return pool

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
        mid = len(sort_idx) // 2
        return int(round(librosa.hz_to_midi(v_f0[sort_idx][mid]))), np.mean(v_conf)

    
    def _measure_duration(self, source, start_frame, pmap=None, start_pitch=None):
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
        
        check_pitch = (pmap is not None and start_pitch is not None)
        crepe_hop = self.cfg["audio"]["crepe"]["hop_length"]
        
        while curr < end:
            if rms[curr] < threshold: break
            if check_pitch:
                t_curr = librosa.frames_to_time(curr, sr=sr, hop_length=hop)
                c_idx = int(t_curr * 16000 / crepe_hop)
                if c_idx < len(pmap["f0"]):
                    f0 = pmap["f0"][c_idx]
                    conf = pmap["conf"][c_idx]
                    if conf > 0.4 and not np.isnan(f0):
                        curr_midi = librosa.hz_to_midi(f0)
                        if abs(curr_midi - start_pitch) > h_cfg["max_pitch_drift"]:
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
        # V108 Style: Magnetic Quantization (Better for jazz/swing/human timing)
        q = self._quantize_magnetic(self.master_pool, d_cfg["grids"])
        s = self._sieve_density(q, d_cfg)
        m = self._allocate_lanes(s, d_cfg["lanes"], d_cfg["chaos"])
        f = self._resolve_physics(m)
        self._export_json(f, diff_name)

    def _quantize_magnetic(self, pool, grids):
        # RESTORED V108 MAGNETIC SNAP
        # V204 rigid snap kills groove. This allows slight offsets if they are "in the pocket".
        beats = self.rhythm["beat_times"]
        out = []
        snap_strength = 0.6 # Allow 40% human error
        
        for n in pool:
            t = n["time"]
            if len(beats) < 2: out.append(n); continue
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            
            # Determine beat duration
            if idx < len(beats)-1: b_dur = beats[idx+1] - beat_t
            else: b_dur = beat_t - beats[idx-1]
            
            best_t = t
            min_dist = 999.0
            
            for div in grids:
                step = b_dur / (div/4.0)
                cand = beat_t + round((t-beat_t)/step)*step
                d = abs(cand - t)
                if d < min_dist: min_dist = d; best_t = cand
            
            # V108 Magnetic Logic: Only snap if within 0.07s
            if min_dist < 0.07:
                n["time"] = t + (best_t - t) * snap_strength
                if n["dur"] > 0:
                    sixteenth = b_dur/4.0
                    n["dur"] = round(n["dur"]/sixteenth)*sixteenth
            out.append(n)
        return sorted(out, key=lambda x: x["time"])

    def _sieve_density(self, pool, d_cfg):
        # Simple clustering sieve
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
                # Collision: Try move
                moved = False
                for c in [0,1,2,3]:
                    if c!=l and n["time"] >= lane_end[c]+0.05:
                        n["lane"]=c; lane_end[c]=n["time"]+n["dur"]
                        cleaned.append(n); moved=True; break
            else:
                lane_end[l]=n["time"]+n["dur"]
                cleaned.append(n)
        
        cleaned.sort(key=lambda x: x["time"])
        # Cut overlapping holds
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