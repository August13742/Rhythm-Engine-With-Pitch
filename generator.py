"""
V204 - PROJECT ZERO ERROR
Architecture: Dual-Engine
    1. Vocals: V203.1 (Crepe Viterbi + Diarization)
    2. Instruments: V108 (Onset Trigger + Pyin Chunking)
Objective: Best-in-class handling for both fluid vocals and percussive instruments.
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
        "global_offset_sec": -0.015,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "audio": {
        "sr": 44100,
        "hop_length": 512,
        # V108 Pyin Settings for Instruments
        "pyin": {
            "frame_length": 4096, # Short window for transient accuracy
            "fmin": 40,
            "fmax": 2000,
            "confidence": 0.15 # Lower confidence allowed for percussive hits
        },
        # V203 Crepe Settings for Vocals
        "crepe": {
            "model": "full",
            "hop_length": 160, # 10ms
            "confidence": 0.40
        },
        # V203 Filters (Crucial for clean harvesting)
        "filters": {
            "default": {"hp": 60, "lp": 14000, "order": 8},
            "bass":    {"hp": 25, "lp": 8000,  "order": 8},
            "drums":   {"hp": 25, "lp": 18000, "order": 8},
        }
    },
    # [RESTORED] V203 Vocal Consensus Logic
    "consensus": {
        "attack_skip": 0.030,   # Ignore consonant noise
        "sustain_win": 0.120,   # Max sampling window
        "min_valid_ratio": 0.3, # Reject breath/noise notes
    },
    "harvest": {
        # V108 Sensitivities (Loose harvesting)
        "sens": {
            "vocals": 0.05, "drums": 0.10, "bass": 0.10,
            "piano": 0.05, "guitar": 0.04, "other": 0.08
        }
    },
    "diarization": {
        "reset_time": 4.0,
        "max_jump": 7,
        "backing_gate": 0.3
    },
    "holds": {
        "energy_decay": 0.60,
        "tap_threshold": 0.75,
        "max_dur": 2.0,
        "gap_buffer": 0.10
    },
    "mixing": {
        "stem_vol": {
            "vocals": 1.25, "drums": 0.90, "bass": 0.85,
            "piano": 0.90, "guitar": 0.85, "other": 0.60
        },
        "priorities": {
            "vocals": 2.0, "drums": 1.5, "bass": 1.2,
            "piano": 1.1, "guitar": 1.1, "other": 0.8
        },
        "dynamic_range": 0.3
    },
    "visuals": {
        "ranges": {
            "vocals": (48, 84), "piano": (48, 88), "guitar": (40, 76),
            "bass": (36, 60), "other": (48, 88)
        }
    },
    "difficulty": {
        "EASY":   {"lanes": 4, "grids": [4],      "poly": 1, "density": 2.0, "min_score": 0.60, "chaos": 0.0},
        "NORMAL": {"lanes": 4, "grids": [4, 8],   "poly": 2, "density": 4.0, "min_score": 0.50, "chaos": 0.0},
        "HARD":   {"lanes": 4, "grids": [4, 8, 12, 16], "poly": 3, "density": 6.0, "min_score": 0.35, "chaos": 0.1},
        "INSANE": {"lanes": 4, "grids": [4, 8, 12, 16, 24], "poly": 4, "density": 8.0, "min_score": 0.20, "chaos": 0.2},
    }
}
DIFF_CONFIGS = CONSTANTS["difficulty"]
class MapGenerator:
    def __init__(self, stems_path, use_holds=True):
        print(f"[INIT] V204 Frankenstein Engine (V203 Vocals + V108 Inst)...")
        self.paths = stems_path
        self.use_holds = use_holds
        self.cfg = CONSTANTS
        self.manifest = self._load_manifest()
        
        # 1. Load & Clean Audio
        self.audio_data = self._load_audio()
        
        # 2. Pre-process Envelopes
        self.envs = {}
        self.rms = {}
        self._generate_envelopes()
        
        # 3. Vocal Pitch Inference (Crepe Only)
        self.vocal_map = None
        if self._check_stem("vocals"):
            self.vocal_map = self._infer_vocal_pitch()
            
        # 4. Rhythm Analysis
        self.rhythm = self._analyze_rhythm()
        
        # 5. Harvest & Pipeline
        self.raw_notes = self._harvest_all()
        self.master_pool = self._score_and_sort(self.raw_notes)
        
        print(f"[INIT] Ready. Master Pool: {len(self.master_pool)} events.")

    # =========================================================
    #   PHASE 1: DATA INGESTION (V203 CLEANING)
    # =========================================================
    
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
                
            y, _ = librosa.load(path, sr=sr, mono=True)
            
            # Context-Aware Filtering (Keep this! It helps Pyin too)
            f_set = filters.get(name, filters["default"])
            sos_hp = scipy.signal.butter(f_set["order"], f_set["hp"], 'hp', fs=sr, output='sos')
            sos_lp = scipy.signal.butter(f_set["order"], f_set["lp"], 'lp', fs=sr, output='sos')
            y = scipy.signal.sosfilt(sos_hp, y)
            y = scipy.signal.sosfilt(sos_lp, y)
            
            # Normalize
            peak = np.max(np.abs(y))
            if peak > 0: y /= peak
            
            loaded[name] = y
            max_len = max(max_len, len(y))
            
        # Pad
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
            
            # Hybrid Envelopes
            if name == "vocals":
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

    def _infer_vocal_pitch(self):
        """V203: Run Crepe ONLY on vocals."""
        print(f"[AI] Running Crepe Viterbi on Vocals...")
        y = self.audio_data["vocals"]
        sr = self.cfg["audio"]["sr"]
        
        # Resample
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        dev = self.cfg["system"]["device"]
        audio = torch.tensor(y_16k, device=dev).float().unsqueeze(0)
        
        c_cfg = self.cfg["audio"]["crepe"]
        
        pitch = torchcrepe.predict(
            audio, sample_rate=16000, hop_length=c_cfg["hop_length"],
            fmin=50, fmax=1200, model=c_cfg["model"],
            batch_size=2048, device=dev, decoder=torchcrepe.decode.viterbi
        )
        _, periodicity = torchcrepe.predict(
            audio, sample_rate=16000, hop_length=c_cfg["hop_length"],
            fmin=50, fmax=1200, model=c_cfg["model"],
            batch_size=2048, device=dev, return_periodicity=True,
            decoder=torchcrepe.decode.weighted_argmax
        )
        
        return {
            "f0": pitch.squeeze(0).cpu().numpy(),
            "conf": periodicity.squeeze(0).cpu().numpy(),
            "time_step": c_cfg["hop_length"] / 16000.0
        }

    # =========================================================
    #   PHASE 2: DUAL ENGINE HARVESTING
    # =========================================================

    def _harvest_all(self):
        pool = []
        sens = self.cfg["harvest"]["sens"]
        
        # 1. Vocals -> V203 Engine (Crepe)
        if self._check_stem("vocals"):
            pool.extend(self._harvest_vocals_v203(sens["vocals"]))
            
        # 2. Instruments -> V108 Engine (Pyin/Onset)
        # Includes Bass, because V108 Bass logic is surprisingly rhythmic
        for inst in ["bass", "piano", "guitar", "other"]:
            if self._check_stem(inst):
                pool.extend(self._harvest_inst_v108(inst, sens[inst]))
                
        # 3. Drums -> V203/108 (Same logic)
        if self._check_stem("drums"):
            pool.extend(self._harvest_drums(sens["drums"]))
            
        return pool

    def _harvest_vocals_v203(self, sensitivity):
        """The Crepe-based Harvester from V203."""
        source = "vocals"
        env = self.envs[source]
        sr = self.cfg["audio"]["sr"]
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=2)
        
        notes = []
        pmap = self.vocal_map
        c_cfg = self.cfg["consensus"]
        
        voices = [None, None]
        last_times = [-999.0, -999.0]
        
        for i, f in enumerate(frames):
            t_onset = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t_onset < 0: continue
            
            # Adaptive Window
            t_next = 9999.0
            if i + 1 < len(frames):
                t_next = librosa.frames_to_time(frames[i+1], sr=sr) + self.cfg["system"]["global_offset_sec"]
            
            t_start = t_onset + c_cfg["attack_skip"]
            max_end = t_start + c_cfg["sustain_win"]
            safe_end = t_next - 0.010 
            t_end = min(max_end, safe_end)
            if t_end <= t_start: t_end = t_start + 0.02
            
            midi_val, conf = self._sample_crepe_map(pmap, t_start, t_end)
            
            if midi_val is None or conf < self.cfg["audio"]["crepe"]["confidence"]:
                continue
                
            voice_id, voices, last_times = self._solve_diarization(midi_val, t_onset, voices, last_times)
            dur = self._measure_duration(source, f)
            
            notes.append({
                "time": t_onset, "midi": midi_val, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": voice_id
            })
        return notes

    def _harvest_inst_v108(self, source, sensitivity):
        """The Legacy Harvester from V108. Better for percussive pitch."""
        env = self.envs[source]
        y = self.audio_data[source]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        
        # Pyin Settings
        p_cfg = self.cfg["audio"]["pyin"]
        
        # V108 Peak Picking (Slightly looser wait time for instruments)
        frames = librosa.util.peak_pick(env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=sensitivity, wait=4)
        
        notes = []
        
        for f in frames:
            t = librosa.frames_to_time(f, sr=sr) + self.cfg["system"]["global_offset_sec"]
            if t < 0: continue
            
            # V108 Pitch Detection: Chunk Analysis on Onset
            start_samp = int(f * hop)
            end_samp = start_samp + p_cfg["frame_length"]
            if end_samp > len(y): continue
            
            chunk = y[start_samp:end_samp]
            
            # Fast Pyin on small chunk
            f0, _, prob = librosa.pyin(
                chunk, fmin=p_cfg["fmin"], fmax=p_cfg["fmax"], 
                sr=sr, frame_length=p_cfg["frame_length"], fill_na=np.nan
            )
            
            # Filter low confidence
            mask = prob > p_cfg["confidence"]
            valid_f0 = f0[mask]
            valid_f0 = valid_f0[~np.isnan(valid_f0)]
            
            if len(valid_f0) == 0: continue
            
            hz = np.median(valid_f0)
            midi = int(round(librosa.hz_to_midi(hz)))
            
            dur = self._measure_duration(source, f)
            
            notes.append({
                "time": t, "midi": midi, "dur": dur, 
                "source": source, "score": float(env[f]), "voice_id": 0
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

    # =========================================================
    #   UTILITIES
    # =========================================================

    def _sample_crepe_map(self, pmap, t_start, t_end):
        idx_s = int(t_start / pmap["time_step"])
        idx_e = int(t_end / pmap["time_step"])
        if idx_s >= len(pmap["f0"]): return None, 0.0
        idx_e = max(idx_e, idx_s + 1)
        
        f0 = pmap["f0"][idx_s:idx_e]
        conf = pmap["conf"][idx_s:idx_e]
        
        mask = (conf > self.cfg["audio"]["crepe"]["confidence"]) & (~np.isnan(f0))
        if np.sum(mask) == 0: return None, 0.0
        
        # Weighted Median
        v_f0 = f0[mask]
        v_conf = conf[mask]
        sort_idx = np.argsort(v_f0)
        
        cumsum = np.cumsum(v_conf[sort_idx])
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2.0)
        
        midi = int(round(librosa.hz_to_midi(v_f0[sort_idx][median_idx])))
        return midi, np.mean(v_conf)

    def _solve_diarization(self, midi, t, voices, last_times):
        d_cfg = self.cfg["diarization"]
        silence = [t - last_times[0], t - last_times[1]]
        stale = [s > d_cfg["reset_time"] for s in silence]
        dist = [
            abs(midi - voices[0]) if voices[0] is not None else 999,
            abs(midi - voices[1]) if voices[1] is not None else 999
        ]
        chosen = 0
        if stale[0] and stale[1]: chosen = 0
        elif not stale[0] and stale[1]:
            chosen = 1 if dist[0] > d_cfg["max_jump"] else 0
        elif not stale[1] and stale[0]:
            chosen = 0 if dist[1] > d_cfg["max_jump"] else 1
        else:
            chosen = 0 if dist[0] <= dist[1] else 1
            
        voices[chosen] = midi
        last_times[chosen] = t
        return chosen, voices, last_times

    def _measure_duration(self, source, start_frame):
        if not self.use_holds: return 0.0
        rms = self.rms[source]
        h_cfg = self.cfg["holds"]
        sr = self.cfg["audio"]["sr"]
        hop = self.cfg["audio"]["hop_length"]
        
        if start_frame >= len(rms): return 0.0
        threshold = rms[start_frame] * h_cfg["energy_decay"]
        
        curr = start_frame + 1
        max_dist = int(h_cfg["max_dur"] * sr / hop)
        end = min(len(rms), start_frame + max_dist)
        
        while curr < end:
            if rms[curr] < threshold: break
            curr += 1
            
        dur = librosa.frames_to_time(curr - start_frame, sr=sr, hop_length=hop)
        if dur < h_cfg["tap_threshold"]: return 0.0
        return dur

    # =========================================================
    #   PHASE 3: EXPORT (V203 PHYSICS)
    # =========================================================

    def generate_all(self):
        for diff in self.cfg["difficulty"]:
            self._generate_chart(diff)

    def _generate_chart(self, diff_name):
        print(f"[GEN] Processing {diff_name}...")
        d_cfg = self.cfg["difficulty"][diff_name]
        
        quantized = self._quantize(self.master_pool, d_cfg["grids"])
        sieved = self._sieve_density(quantized, d_cfg)
        mapped = self._allocate_lanes(sieved, d_cfg["lanes"], d_cfg["chaos"])
        final = self._resolve_physics(mapped)
        
        self._export_json(final, diff_name)

    def _quantize(self, pool, grids):
        beats = self.rhythm["beat_times"]
        out = []
        for n in pool:
            t = n["time"]
            if len(beats) < 2: out.append(n); continue
            idx = (np.abs(beats - t)).argmin()
            beat_t = beats[idx]
            
            if idx < len(beats)-1: b_dur = beats[idx+1]-beat_t
            else: b_dur = beat_t - beats[idx-1]
            
            best_t = t
            min_dist = 999.0
            
            for div in grids:
                step = b_dur/(div/4.0)
                cand = beat_t + round((t - beat_t)/step)*step
                dist = abs(cand - t)
                if dist < min_dist: min_dist = dist; best_t = cand
            
            if min_dist < 0.06:
                n["time"] = best_t
                if n["dur"] > 0:
                     sixteenth = b_dur / 4.0
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
            window = [c for c in candidates if s <= c["time"] < s+1]
            window = [c for c in window if c["score"] >= d_cfg["min_score"]]
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
            midi = n["midi"]
            low, high = r_cfg.get(src, r_cfg["other"])
            norm = np.clip((midi - low) / (high - low), 0.0, 1.0)
            target = 0
            
            if src == "vocals":
                if n["voice_id"] == 0: target = int(norm * 2.5)
                else: target = min(3, 1 + int(norm * 2.5))
            else:
                target = int(norm * (lanes - 1))
            
            if chaos > 0 and np.random.rand() < chaos: target = np.random.randint(0, lanes)
            elif target == last_lane and src != "vocals": target = (target+1)%lanes
            
            vol = m_cfg["stem_vol"].get(src, 0.8) * (0.8 + n["score"]*0.3)
            n["vol"] = np.clip(vol, 0.0, 1.0)
            n["lane"] = target
            n["type"] = "hold" if n["dur"] > 0 else "tap"
            out.append(n)
            last_lane = target
        return out

    def _resolve_physics(self, notes):
        notes.sort(key=lambda x: x["time"])
        cleaned = []
        lane_end = {i: -1.0 for i in range(4)}
        for n in notes:
            l = n["lane"]
            if n["time"] < lane_end[l] + 0.05:
                # Collision: Try move
                moved = False
                for c in [0,1,2,3]:
                    if c!=l and n["time"] >= lane_end[c] + 0.05:
                        n["lane"] = c; lane_end[c] = n["time"]+n["dur"]
                        cleaned.append(n); moved=True; break
            else:
                lane_end[l] = n["time"]+n["dur"]
                cleaned.append(n)
        
        cleaned.sort(key=lambda x: x["time"])
        # Truncate
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
            n["score"] *= prio.get(n["source"], 1.0)
            if len(beat_arr)>0 and np.min(np.abs(beat_arr - n["time"])) < 0.05:
                n["score"] *= 1.25
            if n["dur"] > 0: n["score"] *= 1.1
            if n["source"] == "vocals" and n["voice_id"] == 1:
                if n["score"] < self.cfg["diarization"]["backing_gate"]: continue
        return sorted(pool, key=lambda x: x["time"])

    def _analyze_rhythm(self):
        y = np.zeros_like(self.audio_data["vocals"])
        if self._check_stem("drums"): y += self.audio_data["drums"]
        if self._check_stem("bass"): y += self.audio_data["bass"]
        sr = self.cfg["audio"]["sr"]
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
        return {"beat_times": librosa.frames_to_time(beats, sr=sr)}

    def _check_stem(self, name):
        return (name in self.audio_data and 
                self.audio_data[name] is not None and 
                self.manifest.get(name, {}).get("exists", False))

    def _load_manifest(self):
        d = os.path.dirname(self.paths["vocals"])
        p = os.path.join(d, "stems_manifest.json")
        if os.path.exists(p):
            with open(p, 'r') as f: return json.load(f)
        return {k: {"exists": True} for k in self.paths}

    def _export_json(self, notes, diff_name):
        folder = os.path.dirname(self.paths["vocals"])
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