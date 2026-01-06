"""
V208 - THE COUNCIL
Combined Best Features:
  - V205 Gameplay Logic: Magnetic Quantization, Physics, Coincidence Voting.
  - V208 Pitch Logic: FCPE + Crepe + RMVPE running in parallel.
  - Hardware: Optimized for RTX 5090 (Float32 precision).
"""

import numpy as np
import librosa
import scipy.ndimage
import json
import os
import argparse
import torch
import torchcrepe
from torchfcpe import spawn_bundled_infer_model
from rmvpe_model import RMVPE_Infer

# ==========================================
#        TUNING & CONFIGURATION
# ==========================================

CONSTANTS = {
    "system": {
        "global_offset_sec": -0.02,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "rmvpe_path": "models/rmvpe.pt" 
    },
    "audio": {
        "sr": 44100,
        "hop_length": 512,
        "pyin": {
            "frame_length": 4096, 
            "fmin": 40, "fmax": 2000, "confidence": 0.15 
        },
    },
    "council": {
        "fcpe": {"threshold": 0.09, "f0_min": 50, "f0_max": 1100},
        "crepe": {"confidence": 0.50, "model": "full"},
        "rmvpe": {"threshold": 0.10}
    },
    "coincidence": {
        "window": 0.05,           
        "min_support": 0.3,       
        "boost_scale": 0.5,       
        "mask_threshold": 0.35,   
        "mask_ratio": 2.0         
    },
    "harvest": {
        "sens": {
            "vocals": 0.05, 
            "drums": 0.10, "bass": 0.08, 
            "piano": 0.04, "guitar": 0.03, "other": 0.06
        }
    },
    "holds": {
        "energy_decay": 0.75,
        "tap_threshold": 0.5,
        "max_dur": 2.0,
        "gap_buffer": 0.10,
        "max_pitch_drift": 1.5,
        "allowed_stems": ["vocals", "other"]
    },
    "mixing": {
        "stem_vol": {
            "vocals": 1.25, 
            "drums": 0.90, "bass": 0.85, 
            "piano": 0.90, "guitar": 0.85, "other": 0.60
        },
        "priorities": {
            "vocals": 2.5, 
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
        print(f"[INIT] V208 Tri-Cameral Generator (Optimized)...")
        self.paths = stems_path
        self.use_holds = use_holds
        self.cfg = CONSTANTS
        self.device = self.cfg["system"]["device"]
        self.manifest = self._load_manifest()
        self.council_report = []
        
        self.audio_data = self._load_audio()
        self.envs = {}
        self.rms = {}
        self._generate_envelopes()
        
        self.council_data = {}
        self._convene_council()
            
        self.rhythm = self._analyze_rhythm()
        self.raw_notes = self._harvest_all()
        self.voted_pool = self._apply_coincidence_voting(self.raw_notes)
        self.master_pool = self._score_and_sort(self.voted_pool)
        print(f"[INIT] Ready. Master Pool: {len(self.master_pool)} events.")

    def _load_audio(self):
        loaded = {}
        max_len = 0
        sr = self.cfg["audio"]["sr"]
        print("[DSP] Loading Stems...")
        for name, path in self.paths.items():
            if not os.path.exists(path): continue
            y, _ = librosa.load(path, sr=sr, mono=True)
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
            else:
                self.envs[name] = librosa.onset.onset_strength(y=y, sr=sr)
                if self.envs[name].max() > 0: self.envs[name] /= self.envs[name].max()

    # =========================================================
    #   PHASE 3: THE COUNCIL MEETING
    # =========================================================
    def _convene_council(self):
        if "vocals" not in self.audio_data and "vocals_lead" not in self.audio_data: return
        print(f"\n--- CONVENING THE TRI-CAMERAL COUNCIL ---")
        
        # 1. SELECT SOURCE (Priority: Viperx Lead > Mixed Vocals)
        if "vocals_lead" in self.audio_data and self._check_signal("vocals_lead"):
            print("  [DSP] Source: Viperx Lead Vocal (High Quality)")
            y = self.audio_data["vocals_lead"]
        else:
            print("  [DSP] Source: Mixed Vocals (Standard)")
            y = self.audio_data["vocals"]

        # 2. HYGIENE: Restore Anti-Bleed Masking
        # Even with Viperx, we might want to crush sections where Backing > Lead
        if "vocals_backing" in self.audio_data and self._check_signal("vocals_backing"):
            print("  [DSP] Applying Dynamic Anti-Bleed Masking...")
            y_back = self.audio_data["vocals_backing"]
            
            # Fast RMS subtraction
            hop = 512
            rms_lead = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
            rms_back = librosa.feature.rms(y=y_back, frame_length=2048, hop_length=hop)[0]
            
            # Align lengths
            min_len = min(len(rms_lead), len(rms_back))
            rms_lead = rms_lead[:min_len]
            rms_back = rms_back[:min_len]
            
            # Calculate mask (Aggressive Gating)
            # If backing energy is > 70% of lead energy, crush the signal
            ratio = rms_back / (rms_lead + 1e-6)
            mask = np.where(ratio > 0.7, 0.1, 1.0) 
            
            # Upsample mask to audio rate
            mask_audio = scipy.ndimage.zoom(mask, len(y)/len(mask), order=0)
            y = y * mask_audio[:len(y)]

        # 3. Resample for AI Models
        print("  [DSP] Resampling to 16k for AI Models...")
        y_16k = librosa.resample(y, orig_sr=44100, target_sr=16000)
        
        # 4. Run Models
        self.council_data["fcpe"] = self._run_fcpe(y_16k)
        self.council_data["crepe"] = self._run_crepe(y_16k)
        self.council_data["rmvpe"] = self._run_rmvpe(y_16k)
        print("--- COUNCIL IN SESSION ---\n")

    def _check_signal(self, name):
        if name not in self.audio_data: return False
        return np.max(np.abs(self.audio_data[name])) > 0.02

    def _run_fcpe(self, y_16k):
        print(f"  [1/3] FCPE...")
        audio_t = torch.tensor(y_16k, device=self.device).float().unsqueeze(0).unsqueeze(-1)
        model = spawn_bundled_infer_model(device=self.device)
        f_cfg = self.cfg["council"]["fcpe"]
        f0_t = model.infer(audio_t, sr=16000, decoder_mode="local_argmax",
                           threshold=f_cfg["threshold"], f0_min=f_cfg["f0_min"], f0_max=f_cfg["f0_max"])
        return {"f0": f0_t.squeeze().cpu().numpy(), "time_step": len(y_16k)/len(f0_t.squeeze())/16000.0}

    def _run_crepe(self, y_16k):
        print(f"  [2/3] TorchCrepe...")
        audio_t = torch.tensor(y_16k, device=self.device).float().unsqueeze(0)
        hop_len = 160
        f0, conf = torchcrepe.predict(audio_t, sample_rate=16000, hop_length=hop_len,
                                      fmin=50, fmax=880, model='full', batch_size=2048,
                                      device=self.device, return_periodicity=True)
        return {"f0": f0.squeeze().cpu().numpy(), "conf": conf.squeeze().cpu().numpy(), "time_step": hop_len/16000.0}

    def _run_rmvpe(self, y_16k):
        print(f"  [3/3] RMVPE...")
        os.makedirs("models", exist_ok=True)
        w_path = self.cfg["system"]["rmvpe_path"]
        try:
            model = RMVPE_Infer(w_path, self.device)
            raw_f0 = model.infer(y_16k, thred=self.cfg["council"]["rmvpe"]["threshold"])
            return {"f0": raw_f0, "time_step": len(y_16k)/len(raw_f0)/16000.0}
        except Exception as e:
            print(f"    [ERR] RMVPE Failed: {e}")
            return None

    # =========================================================
    #   PHASE 4: HARVESTING 
    # =========================================================

    def _harvest_all(self):
        pool = []
        sens = self.cfg["harvest"]["sens"]
        
        # Use Lead envelope if available, else standard vocals
        src_vocals = "vocals_lead" if "vocals_lead" in self.envs else "vocals"
        
        if src_vocals in self.audio_data:
            pool.extend(self._harvest_vocals_voting(src_vocals, sens["vocals"]))
        
        for inst in ["bass", "piano", "guitar", "other"]:
            if self._check_stem(inst):
                pool.extend(self._harvest_inst_pyin(inst, sens[inst]))
        if self._check_stem("drums"):
            pool.extend(self._harvest_drums(sens["drums"]))
        return pool

    def _harvest_vocals_voting(self, source, sensitivity):
        env = self.envs[source]
        sr = 44100
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=2)
        notes = []
        busy_until = 0.0
        
        for f in frames:
            t_onset = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t_onset < 0 or t_onset < busy_until: continue 
            
            t_start = t_onset + 0.040
            t_end = t_start + 0.100
            
            vote_result = self._cast_votes(t_start, t_end)
            if vote_result is None: continue
            
            midi_final, confidence_label = vote_result
            
            dur = self._measure_duration(source, f)
            if dur > 0: busy_until = t_onset + dur + 0.05
                
            notes.append({
                "time": t_onset, "midi": midi_final, "dur": dur, 
                "source": "vocals", # Normalize name for mixing priority
                "score": float(env[f]), "voice_id": 0,
                "vote_type": confidence_label
            })
        return notes

    def _cast_votes(self, t_start, t_end):
        log_entry = {
            "time": float(f"{t_start:.3f}"),
            "inputs": {},
            "status": "REJECTED",
            "logic": "N/A",
            "final_midi": None
        }

        # --- 1. GATHER DATA ---
        v_fcpe, c_fcpe = self._sample_f0_detailed(self.council_data["fcpe"], t_start, t_end)
        v_crepe, c_crepe = self._sample_f0_detailed(self.council_data["crepe"], t_start, t_end, conf_key="conf")
        v_rmvpe, c_rmvpe = self._sample_f0_detailed(self.council_data["rmvpe"], t_start, t_end)

        log_entry["inputs"] = {
            "FCPE":  {"midi": v_fcpe, "conf": float(f"{c_fcpe:.2f}")} if v_fcpe else None,
            "CREPE": {"midi": v_crepe, "conf": float(f"{c_crepe:.2f}")} if v_crepe else None,
            "RMVPE": {"midi": v_rmvpe, "conf": float(f"{c_rmvpe:.2f}")} if v_rmvpe else None
        }

        crepe_thresh = self.cfg["council"]["crepe"]["confidence"]

        # --- 2. THE VETO ---
        if c_crepe < crepe_thresh:
            if v_fcpe and v_rmvpe and abs(v_fcpe - v_rmvpe) < 0.8:
                log_entry["logic"] = "VETO_OVERRIDDEN_BY_MAJORITY"
            else:
                log_entry["status"] = "VETOED"
                log_entry["logic"] = "CREPE_VETO_SILENCE"
                self.council_report.append(log_entry)
                return None

        # --- 3. FILTER VALID VOTES ---
        votes = {}
        if v_fcpe: votes["fcpe"] = v_fcpe
        if v_crepe and c_crepe >= crepe_thresh: votes["crepe"] = v_crepe
        if v_rmvpe: votes["rmvpe"] = v_rmvpe

        if not votes:
            log_entry["logic"] = "NO_VALID_VOTES"
            self.council_report.append(log_entry)
            return None
        
        # --- 4. OCTAVE FOLDING ---
        folded = False
        if len(votes) >= 2:
            anchor = None
            if "crepe" in votes: anchor = votes["crepe"]
            elif "rmvpe" in votes: anchor = votes["rmvpe"]
            else: anchor = votes["fcpe"]

            for k in votes:
                if votes[k] == anchor: continue
                diff = votes[k] - anchor
                shift = int(round(diff / 12.0))
                if shift != 0:
                    votes[k] -= (shift * 12)
                    folded = True
        
        vals = list(votes.values())

        # --- 5. CONSENSUS LOGIC ---
        final_res = None
        logic_tag = "UNKNOWN"

        if len(vals) >= 2:
            count = 0
            for v1 in vals:
                local_count = 0
                for v2 in vals:
                    if abs(v1 - v2) < 0.8: 
                        local_count += 1
                if local_count >= 2:
                    final_res = int(round(v1))
                    logic_tag = "UNANIMOUS" if len(vals) == 3 and local_count == 3 else "MAJORITY"
                    break
            
            if final_res is None:
                pairs = [("fcpe", "crepe"), ("fcpe", "rmvpe"), ("crepe", "rmvpe")]
                for m1, m2 in pairs:
                    if m1 in votes and m2 in votes and abs(votes[m1] - votes[m2]) < 0.8:
                        final_res = int(round((votes[m1] + votes[m2])/2))
                        logic_tag = f"SPLIT_DECISION_{m1.upper()}_{m2.upper()}"
                        break

        # Fallbacks
        if final_res is None:
            if "crepe" in votes: 
                final_res = int(round(votes["crepe"]))
                logic_tag = "CREPE_TRUST_FALLBACK"
            elif "rmvpe" in votes: 
                final_res = int(round(votes["rmvpe"]))
                logic_tag = "RMVPE_LAST_RESORT"
        
        # --- 6. FINALIZE ---
        if final_res is not None:
            log_entry["status"] = "ACCEPTED"
            log_entry["final_midi"] = final_res
            log_entry["logic"] = f"{logic_tag}{'_FOLDED' if folded else ''}"
            self.council_report.append(log_entry)
            return final_res, logic_tag.lower()
        else:
            log_entry["logic"] = "NO_AGREEMENT"
            self.council_report.append(log_entry)
            return None

    def _harvest_inst_pyin(self, source, sensitivity):
        env = self.envs[source]
        y = self.audio_data[source]
        sr = 44100
        hop = 512
        p_cfg = self.cfg["audio"]["pyin"]
        
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=5)
        notes = []
        busy_until = 0.0
        
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0 or t < busy_until: continue
            
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
            dur = self._measure_duration(source, f)
            if dur > 0: busy_until = t + dur + 0.05
            
            notes.append({
                "time": t, "midi": midi, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": 0
            })
        return notes

    def _harvest_drums(self, sensitivity):
        source = "drums"
        env = self.envs[source]
        sr = 44100
        frames = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=sensitivity, wait=4)
        notes = []
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0: continue
            notes.append({"time": t, "midi": 36, "dur": 0.0, "source": source, "score": float(env[f]), "voice_id": 0})
        return notes

    def _apply_coincidence_voting(self, pool):
        print("[GEN] Applying Psychoacoustic Voting...")
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
            
            j = i - 1
            while j >= 0:
                neighbor = pool[j]
                if t - neighbor["time"] > v_cfg["window"]: break 
                if neighbor["source"] != src:
                    w = priorities.get(neighbor["source"], 0.8)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                j -= 1
            k = i + 1
            while k < n_notes:
                neighbor = pool[k]
                if neighbor["time"] - t > v_cfg["window"]: break 
                if neighbor["source"] != src:
                    w = priorities.get(neighbor["source"], 0.8)
                    support_score += w * neighbor["score"]
                    masking_energy += neighbor["score"]
                k += 1

            if support_score > v_cfg["min_support"]:
                current["score"] *= (1.0 + support_score * v_cfg["boost_scale"])
            elif current["score"] < v_cfg["mask_threshold"]:
                if src == "vocals": pass
                elif masking_energy > (current["score"] * v_cfg["mask_ratio"]):
                    current["score"] *= 0.1
        return pool

    def _measure_duration(self, source, start_frame, pmap=None, start_pitch=None):
        if not self.use_holds: return 0.0
        allowed = self.cfg["holds"]["allowed_stems"]
        if not any(a in source for a in allowed): return 0.0
        
        rms = self.rms[source]
        h_cfg = self.cfg["holds"]
        sr = 44100
        hop = 512
        
        if start_frame >= len(rms): return 0.0
        threshold = rms[start_frame] * h_cfg["energy_decay"]
        
        curr = start_frame + 1
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        end = min(len(rms), start_frame + max_dist)
        
        while curr < end:
            if rms[curr] < threshold: break
            curr += 1
            
        dur = librosa.frames_to_time(curr - start_frame, sr=sr, hop_length=hop)
        return dur if dur >= h_cfg["tap_threshold"] else 0.0

    def _sample_f0_detailed(self, data, t_start, t_end, conf_key=None):
        if data is None: return None, 0.0
        f0_arr = data["f0"]
        step = data["time_step"]
        idx_s = int(t_start / step)
        idx_e = min(int(t_end / step), len(f0_arr))
        
        if idx_s >= len(f0_arr): return None, 0.0
        
        segment = f0_arr[idx_s:idx_e]
        conf_val = 1.0
        if conf_key:
            c_seg = data[conf_key][idx_s:idx_e]
            if len(c_seg) > 0: conf_val = np.mean(c_seg)
            else: conf_val = 0.0
            mask = c_seg > 0.1
            segment = segment[mask]
        else:
            segment = segment[segment > 40]

        if len(segment) == 0: return None, 0.0
        
        hz = np.median(segment)
        if hz < 40: return None, 0.0
        
        return int(round(librosa.hz_to_midi(hz))), conf_val

    def _export_council_report(self):
        if not self.council_report: return
        folder = "."
        for path in self.paths.values():
            if os.path.exists(path):
                folder = os.path.dirname(path)
                break
        base = os.path.basename(folder)
        fname = os.path.join(folder, f"{base}_Council_Report.json")
        print(f"[REPORT] Saving Council Minutes to {fname}...")
        with open(fname, 'w') as f:
            json.dump(self.council_report, f, indent=2)
            
    def generate_all(self):
        for diff in self.cfg["difficulty"]:
            self._generate_chart(diff)
        self._export_council_report()

    def _generate_chart(self, diff_name):
        print(f"[GEN] Processing {diff_name}...")
        d_cfg = self.cfg["difficulty"][diff_name]
        q = self._quantize_magnetic(self.master_pool, d_cfg["grids"])
        s = self._sieve_density(q, d_cfg)
        m = self._allocate_lanes(s, d_cfg["lanes"], d_cfg["chaos"])
        f = self._resolve_physics(m)
        self._export_json(f, diff_name)

    def _quantize_magnetic(self, pool, grids):
        beats = self.rhythm["beat_times"]
        out = []
        snap_strength = 0.6
        for n in pool:
            t = n["time"]
            if len(beats) < 2: out.append(n); continue
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            if idx < len(beats)-1: b_dur = beats[idx+1] - beat_t
            else: b_dur = beat_t - beats[idx-1]
            
            best_t = t
            min_dist = 999.0
            for div in grids:
                step = b_dur / (div/4.0)
                cand = beat_t + round((t-beat_t)/step)*step
                d = abs(cand - t)
                if d < min_dist: min_dist = d; best_t = cand
            
            if min_dist < 0.07:
                n["time"] = t + (best_t - t) * snap_strength
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
                target = int(norm * 2.5)
            else:
                low, high = r_cfg.get(src, r_cfg["other"])
                norm = np.clip((n["midi"]-low)/(high-low), 0.0, 1.0)
                target = int(norm*(lanes-1))
            
            if chaos>0 and np.random.rand()<chaos: target = np.random.randint(0, lanes)
            elif target==last_lane and "vocals" not in src: target=(target+1)%lanes
            
            base_vol = m_cfg["stem_vol"].get(src, 0.8)
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
                for c in [0,1,2,3]:
                    if c!=l and n["time"] >= lane_end[c]+0.05:
                        n["lane"]=c; lane_end[c]=n["time"]+n["dur"]
                        cleaned.append(n); break
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
        sr = 44100
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
                "source": n["source"], "vol": float(f"{n['vol']:.3f}"),
                "vote": n.get("vote_type", "raw")
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
        "drums":  os.path.join(args.folder, "drums.wav"),
        "bass":   os.path.join(args.folder, "bass.wav"),
        "piano":  os.path.join(args.folder, "piano.wav"),
        "guitar": os.path.join(args.folder, "guitar.wav"),
        "other":  os.path.join(args.folder, "other.wav"),
    }
    
    try:
        gen = MapGenerator(stems, use_holds=args.holds)
        gen.generate_all()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FATAL] {e}")