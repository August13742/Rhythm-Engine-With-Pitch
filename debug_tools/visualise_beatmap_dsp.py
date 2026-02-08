import argparse
import sys
import time
import numpy as np
import librosa
import pygame
import warnings
import random

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

warnings.filterwarnings('ignore')

# ==========================================
#               SCULPTOR CONFIG
# ==========================================

# --- CONFIG ---
SFX_TARGET_DIFFICULTY = "INSANE" 
QUANTIZE_TO_GRID = True
DEFAULT_MUSIC_VOL = 0.4
DEFAULT_SFX_VOL = 0.9

# --- AUDIO CONSTANTS ---
ANALYSIS_FMIN = librosa.note_to_hz('C2') 
ANALYSIS_OCTAVES = 6 
QUANTIZE_WEIGHT = 0.90 

# --- GATING & FILTERING ---
# Hard Floor: Signals quieter than this (relative to peak) are ignored.
DB_FLOOR = -35.0 

# Spectral Flatness Threshold:
# 0.0 = Pure Sine Wave, 1.0 = White Noise
# If flatness > 0.4, it's likely a Snare/Shaker (Noisy), not a Melody Note.
NOISE_THRESHOLD = 0.35 

# --- DIFFICULTY PRESETS ---
DIFF_CONFIGS = {
    "EASY":   { "lanes": 4, "density": 0.20, "chord_prob": 0.0, "grid": 4, "max_fingers": 2 }, 
    "NORMAL": { "lanes": 4, "density": 0.40, "chord_prob": 0.1, "grid": 8, "max_fingers": 2 },
    "HARD":   { "lanes": 5, "density": 0.60, "chord_prob": 0.3, "grid": 12, "max_fingers": 3 }, 
    "INSANE": { "lanes": 6, "density": 0.80, "chord_prob": 0.5, "grid": 16, "max_fingers": 4 }, 
}

# ==========================================
#           DSP ANALYZER
# ==========================================

class SculptorAnalyzer:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
        self.data = self._analyze()

    def _analyze(self):
        print("[1/4] Separation (Harmonic/Percussive)...")
        y_harm, y_perc = librosa.effects.hpss(self.y, margin=(3.0, 1.0))
        
        print("[2/4] Spectral Flatness & Energy...")
        # Compute Flatness to detect "Noisy" instruments (Snares)
        flatness = librosa.feature.spectral_flatness(y=self.y)
        
        # Compute RMS Energy (Loudness) in dB
        rms = librosa.feature.rms(y=self.y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        print("[3/4] Rhythm & Timing...")
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        print("[4/4] Pitch Contour...")
        cqt = np.abs(librosa.cqt(y_harm, sr=self.sr, fmin=ANALYSIS_FMIN, 
                                n_bins=12*ANALYSIS_OCTAVES, bins_per_octave=12))
        cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt = (cqt + 80) / 80 
        
        return {
            "onset_env": onset_env,
            "flatness": flatness[0], # Flatten array
            "rms_db": rms_db,
            "beat_times": beat_times,
            "cqt": cqt,
            "tempo": tempo if np.isscalar(tempo) else tempo[0],
            "duration": librosa.get_duration(y=self.y, sr=self.sr),
            "sr": self.sr
        }

class NoteGenerator:
    def __init__(self, analyzer):
        self.ana = analyzer
        self.midi_base = 36 # C2

    def _snap_time(self, t, beats, subdivisions):
        if len(beats) < 2: return t
        idx = (np.abs(beats - t)).argmin()
        
        if idx < len(beats) - 1: beat_dur = beats[idx+1] - beats[idx]
        else: beat_dur = 0.5
            
        grid_dur = beat_dur / (subdivisions / 4) 
        diff = t - beats[idx]
        snapped_diff = round(diff / grid_dur) * grid_dur
        return beats[idx] + snapped_diff

    def _get_pitch(self, frame_idx):
        cqt = self.ana.data["cqt"]
        if frame_idx >= cqt.shape[1]: frame_idx = cqt.shape[1] - 1
        col = cqt[:, frame_idx]
        max_idx = np.argmax(col)
        return int(self.midi_base + max_idx)

    def process_difficulty(self, diff_name, dens_mod=0.0):
        cfg = DIFF_CONFIGS[diff_name]
        data = self.ana.data
        sr = self.ana.sr
        
        # 1. Onset Detection
        onset_env = data["onset_env"]
        onset_frames = librosa.util.peak_pick(onset_env, 
                                            pre_max=3, post_max=3, 
                                            pre_avg=3, post_avg=5, 
                                            delta=0.02, wait=2)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        candidates = []
        for i, frame in enumerate(onset_frames):
            strength = onset_env[frame]
            
            # --- CHISEL 1: HARD DB FLOOR (Anti-Spam) ---
            # Map frame to RMS index (approximate)
            rms_idx = min(frame, len(data["rms_db"])-1)
            loudness = data["rms_db"][rms_idx]
            
            # If it's quieter than -35dB (or custom floor), it's silence.
            if loudness < DB_FLOOR: 
                continue

            t = onset_times[i]
            final_t = self._snap_time(t, data["beat_times"], cfg["grid"])
            final_t = (final_t * QUANTIZE_WEIGHT) + (t * (1.0 - QUANTIZE_WEIGHT))
            
            # --- CHISEL 2: SPECTRAL FLATNESS (Snare Killer) ---
            flat_idx = min(frame, len(data["flatness"])-1)
            is_noisy = data["flatness"][flat_idx] > NOISE_THRESHOLD
            
            midi = self._get_pitch(frame)
            
            # Bass Detection
            is_bass = midi < 48 

            candidates.append({
                "time": final_t, 
                "midi": midi, 
                "strength": strength,
                "is_bass": is_bass,
                "is_noisy": is_noisy, # Tag it for the mapper logic
                "lane": 0, 
                "type": "tap"
            })
            
        # 2. Density Filtering
        eff_density = max(0.1, min(1.0, cfg["density"] + dens_mod))
        candidates.sort(key=lambda x: x["strength"], reverse=True)
        target_cnt = int(len(candidates) * eff_density)
        
        filtered = sorted(candidates[:target_cnt], key=lambda x: x["time"])
        if not filtered: return [], cfg["lanes"], data["tempo"]

        # 3. MAPPING PHASE (The Sculptor Logic)
        lanes = cfg["lanes"]
        midis = [c["midi"] for c in filtered if not c["is_bass"] and not c["is_noisy"]]
        if not midis: midis = [60]
        
        p_min, p_max = np.percentile(midis, 5), np.percentile(midis, 95)
        spread = max(1, p_max - p_min)
        
        game_notes = []
        last_t = -10
        last_lane = lanes // 2
        last_midi = 60
        
        for i, c in enumerate(filtered):
            if c["time"] < last_t + 0.05: continue
            
            # --- PATTERN LOGIC ---
            
            # RULE A: Bass/Noise Anchor
            # If it's a Kick (Bass) or Snare (Noisy), ignore pitch.
            if c["is_bass"]:
                # Alternating Outer Lanes (0, Max)
                target_lane = 0 if last_lane > lanes/2 else lanes - 1
            elif c["is_noisy"]:
                # Snares/Hi-Hats usually stay in the center or repeat current lane
                target_lane = lanes // 2
            
            # RULE B: Directional Melodic Flow
            else:
                # Calculate Delta
                pitch_delta = c["midi"] - last_midi
                
                if abs(pitch_delta) < 2:
                    # Same note? Stay close to current lane
                    target_lane = last_lane + random.choice([-1, 0, 1])
                elif pitch_delta > 0:
                    # Pitch goes UP -> Lane goes RIGHT
                    step = 1 if pitch_delta < 5 else 2
                    target_lane = last_lane + step
                else:
                    # Pitch goes DOWN -> Lane goes LEFT
                    step = -1 if pitch_delta > -5 else -2
                    target_lane = last_lane + step
                    
                # Standard mapping as a fallback/anchor to keep us centered
                # We blend the "Relative Flow" with the "Absolute Pitch"
                abs_ratio = (c["midi"] - p_min) / spread
                abs_lane = int(abs_ratio * lanes)
                
                # Weight: 60% Flow, 40% Absolute
                # This ensures we don't drift off screen, but we follow the melody curve
                target_lane = int((target_lane * 0.6) + (abs_lane * 0.4))
                
                # Update history for next note
                last_midi = c["midi"]

            # Clamp
            target_lane = max(0, min(lanes-1, target_lane))

            # Duration Logic
            dur = filtered[i+1]["time"] - c["time"] if i < len(filtered)-1 else 1.0
            type_n = "hold" if dur > 0.5 and random.random() > 0.7 else "tap"
            hold_len = min(dur, 2.0) if type_n == "hold" else 0
            
            vol = 0.5 + (0.5 * min(1.0, c["strength"] * 2))
            
            game_notes.append({
                "time": c["time"], "lane": target_lane, "midi": c["midi"],
                "type": type_n, "dur": hold_len, "vol": vol
            })
            
            last_t = c["time"]
            last_lane = target_lane
            
        return game_notes, lanes, data["tempo"]

# ==========================================
#           VISUALIZER
# ==========================================

COLORS = { "bg": (20, 20, 25), "EASY": (100, 255, 100), "NORMAL": (100, 200, 255), "HARD": (255, 200, 50), "INSANE": (255, 50, 50) }

class Visualizer:
    def __init__(self, audio_path):
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        pygame.mixer.set_num_channels(64) 
        
        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Rhythm Sculptor V36 - Press M to toggle Synth/Keysound")
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 24)
        self.playback_mode = "synth"  # synth or keysound
        self.keysounds = {}
        
        print(f"Loading {audio_path}...")
        self.y, self.sr = librosa.load(audio_path, sr=None) 
        
        self.analyzer = SculptorAnalyzer(self.y, self.sr)
        self.generator = NoteGenerator(self.analyzer)
        
        self.dens_mod = 0.0 
        self.music_vol = DEFAULT_MUSIC_VOL
        self.sfx_vol = DEFAULT_SFX_VOL
        self.beatmaps = {}
        self.metadata = {}
        self.played_indices = {}
        self.scroll_speed = 600
        self.hit_line_y = self.height - 100
        
        # Interactive state
        self.diff = "INSANE"  # Current selected difficulty
        self.playing = True
        self.start_time = time.time()
        self.pause_time = 0
        
        # Pre-synthesize tones for synth mode
        print("Synthesizing SFX Bank...")
        self.synth_sounds = {}
        for midi in range(24, 108):
            self.synth_sounds[midi] = self._gen_tone(midi)
        
        # Pre-slice keysounds for keysound mode
        print("Preparing Keysound Bank...")
        self.keysounds = {}

        self.regenerate(restart=False)
        
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(self.music_vol)
        pygame.mixer.music.play()
        self.playing = True
        self.start_time = time.time()
        self.pause_time = 0
        self.scroll_speed = 600
        self.hit_line_y = self.height - 100
        self.played_indices = {} 

    def _gen_tone(self, midi):
        freq = 440.0 * (2.0**((midi-69)/12.0))
        t = np.linspace(0, 0.2, int(44100*0.2), False)
        wave = 0.5 * np.sign(np.sin(2*np.pi*freq*t)) 
        wave = np.convolve(wave, np.ones(5)/5, mode='same') 
        wave = (wave * np.exp(-10*t) * 32767 * 0.5).astype(np.int16)
        return pygame.sndarray.make_sound(np.stack([wave, wave], axis=1))

    def regenerate(self, restart=True):
        print("Generating maps...")
        diffs = ["EASY", "NORMAL", "HARD", "INSANE"]
        for diff in diffs:
            notes, lanes, tempo = self.generator.process_difficulty(diff, self.dens_mod)
            self.beatmaps[diff] = notes
            nps = len(notes) / self.analyzer.data["duration"]
            self.metadata[diff] = {"count": len(notes), "lanes": lanes, "nps": nps}
            
            # Pre-synthesize keysounds for this difficulty
            if self.playback_mode == "keysound":
                self._synthesize_keysounds(diff, notes)
        
        if restart and self.playing:
            pygame.mixer.music.play()
            self.start_time = time.time()
            self.played_indices = {d: 0 for d in diffs}

    def run(self):
        clock = pygame.time.Clock()
        self.played_indices = {d: 0 for d in self.beatmaps}
        
        while True:
            clock.tick(60)
            
            if self.playing:
                curr = time.time() - self.start_time
                target_diff = self.diff
                if target_diff not in self.beatmaps: target_diff = "INSANE"
                
                notes = self.beatmaps[target_diff]
                idx = self.played_indices[target_diff]
                
                while idx < len(notes):
                    n = notes[idx]
                    if n["time"] <= curr:
                        if self.playback_mode == "synth":
                            s = self.synth_sounds.get(n["midi"])
                            if s: 
                                chan = pygame.mixer.find_channel()
                                if chan:
                                    v = n.get("vol", 1.0) * self.sfx_vol
                                    l_ratio = n["lane"] / max(1, self.metadata[target_diff]["lanes"]-1)
                                    chan.set_volume((1-l_ratio)*0.7*v, (0.3+l_ratio*0.7)*v)
                                    chan.play(s)
                        else:  # keysound mode
                            key_id = (target_diff, idx)
                            if key_id in self.keysounds:
                                chan = pygame.mixer.find_channel()
                                if chan:
                                    v = n.get("vol", 1.0) * self.sfx_vol
                                    l_ratio = n["lane"] / max(1, self.metadata[target_diff]["lanes"]-1)
                                    chan.set_volume((1-l_ratio)*0.7*v, (0.3+l_ratio*0.7)*v)
                                    chan.play(self.keysounds[key_id])
                        self.played_indices[target_diff] += 1
                        idx += 1
                    else: break
            else:
                curr = self.pause_time

            for e in pygame.event.get():
                if e.type == pygame.QUIT: return
                if e.type == pygame.KEYDOWN:
                    # Difficulty picker
                    if e.key == pygame.K_1: self.diff = "EASY"
                    if e.key == pygame.K_2: self.diff = "NORMAL"
                    if e.key == pygame.K_3: self.diff = "HARD"
                    if e.key == pygame.K_4: self.diff = "INSANE"
                    
                    # Pause/Play
                    if e.key == pygame.K_SPACE:
                        if self.playing:
                            pygame.mixer.music.pause()
                            self.pause_time = time.time() - self.start_time
                            self.playing = False
                        else:
                            pygame.mixer.music.unpause()
                            self.start_time = time.time() - self.pause_time
                            self.playing = True
                    
                    # Restart
                    if e.key == pygame.K_r:
                        pygame.mixer.music.rewind()
                        pygame.mixer.music.play()
                        self.start_time = time.time()
                        self.played_indices = {d: 0 for d in self.beatmaps}
                        self.playing = True
                    
                    changed = False
                    if e.key == pygame.K_UP: self.dens_mod += 0.05; changed = True 
                    if e.key == pygame.K_DOWN: self.dens_mod -= 0.05; changed = True 
                    if changed: self.regenerate(restart=True)
                    
                    if e.key == pygame.K_LEFTBRACKET: 
                        self.music_vol = max(0, self.music_vol - 0.1)
                        pygame.mixer.music.set_volume(self.music_vol)
                    if e.key == pygame.K_RIGHTBRACKET: 
                        self.music_vol = min(1, self.music_vol + 0.1)
                        pygame.mixer.music.set_volume(self.music_vol)
                    if e.key == pygame.K_MINUS: self.sfx_vol = max(0, self.sfx_vol - 0.1)
                    if e.key == pygame.K_EQUALS: self.sfx_vol = min(1, self.sfx_vol + 0.1)
                    
                    if e.key == pygame.K_m:
                        if self.playback_mode == "synth":
                            self.playback_mode = "keysound"
                            # Synthesize keysounds for current target difficulty
                            if self.diff not in self.keysounds:
                                self._synthesize_keysounds(self.diff, self.beatmaps[self.diff])
                        else:
                            self.playback_mode = "synth"
                        print(f"Playback mode: {self.playback_mode.upper()}")

            self.draw(curr)
            pygame.display.flip()

    def draw(self, curr):
        self.screen.fill(COLORS["bg"])
        col_w = self.width // 4
        diffs = ["EASY", "NORMAL", "HARD", "INSANE"]
        
        for i, diff in enumerate(diffs):
            x_off = i * col_w
            col = COLORS.get(diff, (255,255,255))
            notes = self.beatmaps[diff]
            meta = self.metadata[diff]
            is_selected = (diff == self.diff)
            lanes = meta["lanes"]
            
            pygame.draw.rect(self.screen, col if is_selected else (60,60,65), (x_off, 0, col_w, self.height), 2 if is_selected else 1)
            if i > 0: pygame.draw.rect(self.screen, (0,0,0), (x_off-2, 0, 4, self.height))
            
            lane_w = col_w / lanes
            for l in range(lanes):
                lx = x_off + l * lane_w
                pygame.draw.line(self.screen, (35,35,45), (lx, 0), (lx, self.height))
                
            title = self.big_font.render(f"{diff}", True, col if is_selected else (col[0]//2, col[1]//2, col[2]//2))
            self.screen.blit(title, (x_off + 10, 20))
            stats = self.font.render(f"Notes: {meta['count']} | NPS: {meta['nps']:.1f}", True, (200,200,200))
            self.screen.blit(stats, (x_off + 10, 50))
            
            if i == 0:
                tune = self.font.render(f"Density: {self.dens_mod:+.2f} | DB Floor: {DB_FLOOR}", True, (100,200,255))
                self.screen.blit(tune, (x_off + 10, self.height - 40))
                mode_text = "[SYNTH]" if self.playback_mode == "synth" else "[KEYSOUND]"
                mode_display = self.font.render(f"Mode: {mode_text} (M)", True, (100,200,255))
                self.screen.blit(mode_display, (x_off + 10, self.height - 70))                
                controls = self.font.render("1-4: Diff | SPACE: Pause | R: Restart | [/]: Music Vol | -/+: SFX Vol", True, (150,150,150))
                self.screen.blit(controls, (x_off + 10, self.height - 100))                
                controls = self.font.render("1-4: Diff | SPACE: Pause | R: Restart | [/]: Music Vol | -/+: SFX Vol", True, (150,150,150))
                self.screen.blit(controls, (x_off + 10, self.height - 100))

            pygame.draw.line(self.screen, (100, 255, 100), (x_off, self.hit_line_y), (x_off + col_w, self.hit_line_y), 2)
            
            for n in notes:
                if n["time"] < curr - 0.2: continue
                if n["time"] > curr + 1.5: break
                
                y = self.hit_line_y - (n["time"] - curr) * self.scroll_speed
                x = x_off + n["lane"] * lane_w
                
                base_c = col if is_selected else (col[0]//2, col[1]//2, col[2]//2)
                if n["vol"] < 0.7: base_c = tuple(max(20, c-80) for c in base_c)
                
                if n["type"] == "hold":
                    h = n["dur"] * self.scroll_speed
                    pygame.draw.rect(self.screen, (base_c[0]//3, base_c[1]//3, base_c[2]//3), (x+4, y-h, lane_w-8, h))
                
                pygame.draw.rect(self.screen, base_c, (x+2, y-10, lane_w-4, 20))
    
    def _resynthesize_audio(self, raw_audio):
        """Spectral resynthesis V41: FFT -> harmonic extraction -> additive synthesis with random phase"""
        N = len(raw_audio)
        if N == 0: return np.zeros(100)
        
        # FFT Analysis
        windowed = raw_audio * np.hanning(N)
        spectrum = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(N, 1/self.sr)
        magnitudes = np.abs(spectrum)
        
        # Filtering
        mask = (frequencies > 80) & (frequencies < 9000)
        magnitudes = magnitudes * mask
        
        # Peak picking (top 7 frequencies)
        num_peaks = 7
        peak_indices = np.argpartition(magnitudes, -num_peaks)[-num_peaks:]
        top_freqs = frequencies[peak_indices]
        top_mags = magnitudes[peak_indices]
        
        if np.max(top_mags) > 0:
            top_mags /= np.max(top_mags)
            top_mags = np.minimum(top_mags, 1.0)

        # Additive synthesis with random phase (V41 upgrade)
        out_dur = 0.2
        t = np.linspace(0, out_dur, int(self.sr * out_dur), False)
        new_wave = np.zeros_like(t)
        
        for f, m in zip(top_freqs, top_mags):
            if f == 0: continue
            
            # Random phase prevents "laser zap" attacks
            phase_offset = np.random.uniform(0, 2 * np.pi)
            new_wave += 0.3 * m * np.sin(2 * np.pi * f * t + phase_offset)

        # Envelope with slower attack (0.01s fade-in) + exponential decay
        attack_len = int(0.01 * self.sr)
        envelope = np.exp(-12 * t)
        
        if attack_len < len(envelope):
            envelope[:attack_len] *= np.linspace(0, 1, attack_len)
            
        new_wave *= envelope
        
        # Soft limiting
        max_val = np.max(np.abs(new_wave))
        if max_val > 0:
            new_wave = 0.95 * new_wave / max_val
        
        return new_wave.astype(np.float32)
    
    def _synthesize_keysounds(self, diff, notes):
        """Extract and resynthesize audio for each note"""
        try:
            synth_count = 0
            for idx, n in enumerate(notes):
                key_id = (diff, idx)
                if key_id in self.keysounds:
                    continue
                
                # Extract raw audio segment
                st = int(n["time"] * self.sr)
                et = int((n["time"] + max(0.1, n.get("dur", 0.15))) * self.sr)
                st = max(0, st - int(0.1 * self.sr))
                et = min(len(self.y), et + int(0.2 * self.sr))
                raw_seg = self.y[st:et]
                
                if len(raw_seg) < 100:
                    continue
                
                # Resynthesize
                resynth = self._resynthesize_audio(raw_seg)
                
                # Convert to pygame sound (stereo format)
                resynth_int16 = np.clip(resynth * 32767, -32768, 32767).astype(np.int16)
                # Stack mono to stereo for pygame mixer
                stereo_sound = np.stack([resynth_int16, resynth_int16], axis=1)
                sound = pygame.sndarray.make_sound(stereo_sound)
                self.keysounds[key_id] = sound
                synth_count += 1
            
            if synth_count > 0:
                print(f"[DSP] Synthesized {synth_count} keysounds for {diff}")
        except Exception as e:
            print(f"[DSP] Keysound synthesis error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    args = parser.parse_args()

    app = Visualizer(args.audio_file)
    app.run()