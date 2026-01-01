'''visualizer.py'''

import argparse
import os
import sys
import time
import json
import pygame
import numpy as np
import librosa
from generator import MapGenerator # Import logic module

# --- CONFIG ---
COLORS = { "bg": (20, 20, 25), "EASY": (100, 255, 100), "NORMAL": (100, 200, 255), "HARD": (255, 200, 50), "INSANE": (255, 50, 50) }

class Visualizer:
    def __init__(self, audio_path, folder_path):
        # Initialize Mixer with high buffer to prevent skipping
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        # Allocate enough channels for keysounding (polyphony)
        pygame.mixer.set_num_channels(64)
        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Rhythm Engine - Press M to toggle Synth/Keysound")
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 24)
        
        # Get stem paths from folder
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(folder_path, f"{base_name}_vocals.wav")
        other_path = os.path.join(folder_path, f"{base_name}_other.wav")
        rhythm_path = os.path.join(folder_path, f"{base_name}_rhythm.wav")
        
        # 1. GENERATE MAP
        print("[VIS] Running Generator...")
        self.gen = MapGenerator(vocals_path, other_path, rhythm_path)
        self.beatmaps = {}
        self.metadata = {}
        
        for d in ["EASY", "NORMAL", "HARD", "INSANE"]:
            self.beatmaps[d] = self.gen.generate(d)
            duration = self.gen.data["duration"]
            nps = len(self.beatmaps[d]) / duration if duration > 0 else 0
            lanes_map = {"EASY": 4, "NORMAL": 4, "HARD": 5, "INSANE": 6}
            self.metadata[d] = {"count": len(self.beatmaps[d]), "lanes": lanes_map[d], "nps": nps}
        
        # 2a. SYNTH MODE: Pre-synthesize tones
        print("[VIS] Synthesizing SFX Bank...")
        self.synth_sounds = {}
        for midi in range(24, 108):
            self.synth_sounds[midi] = self._gen_tone(midi)
        
        # 2b. KEYSOUND MODE: Load raw audio for slicing
        print("[VIS] Loading Raw Audio for Keysounding...")
        self.raw_voc, _ = librosa.load(vocals_path, sr=44100, mono=False)
        self.raw_oth, _ = librosa.load(other_path, sr=44100, mono=False)
        
        # Handle Mono/Stereo
        if len(self.raw_voc.shape) == 1: self.raw_voc = np.stack([self.raw_voc, self.raw_voc])
        if len(self.raw_oth.shape) == 1: self.raw_oth = np.stack([self.raw_oth, self.raw_oth])
        
        # Pre-slice keysounds for each difficulty
        self.keysounds = {}
        for d in ["EASY", "NORMAL", "HARD", "INSANE"]:
            print(f"[VIS] Slicing Keysounds for {d}...")
            self._slice_keysounds(d, self.beatmaps[d])
        
        # 2c. AUDIO: Load both full audio and rhythm stem
        pygame.mixer.music.load(audio_path)
        self.full_audio_path = audio_path
        self.rhythm_path = rhythm_path  # Not used anymore since we keep full audio playing
        
        # 3. STATE
        self.mode = "synth"  # Start with synth mode
        self.music_vol = 0.2
        self.sfx_vol = 0.9
        pygame.mixer.music.set_volume(self.music_vol)
        
        # 3. STATE
        self.diff = "INSANE"
        self.playing = True
        self.start_time = time.time()
        self.pause_time = 0
        self.scroll_speed = 600
        self.hit_line_y = self.height - 100
        self.played_indices = {}
        pygame.mixer.music.play()

    def _gen_tone(self, midi):
        freq = 440.0 * (2.0**((midi-69)/12.0))
        t = np.linspace(0, 0.2, int(44100*0.2), False)
        wave = 0.5 * np.sign(np.sin(2*np.pi*freq*t))
        wave = np.convolve(wave, np.ones(5)/5, mode='same')
        wave = (wave * np.exp(-10*t) * 32767 * 0.5).astype(np.int16)
        return pygame.sndarray.make_sound(np.stack([wave, wave], axis=1))

    def _slice_keysounds(self, diff, notes):
        """Resynthesize audio notes using spectral analysis for keysounding mode"""
        print(f"[VIS] Resynthesizing SFX for {diff} (Spectral Analysis)...")
        self.keysounds[diff] = []
        
        for i, n in enumerate(notes):
            # 1. EXTRACT RAW AUDIO SLICE
            src = self.raw_voc if n["source"] == "vocal" else self.raw_oth
            start_sample = int(n["time"] * 44100)
            # Use a short, punchy duration for analysis window (150ms is enough to catch the tone)
            analyze_dur = int(0.15 * 44100) 
            
            if start_sample >= src.shape[1]:
                self.keysounds[diff].append(None)
                continue
                
            raw_slice = src[:, start_sample : start_sample + analyze_dur]
            
            # Handle Stereo -> Mono for FFT analysis
            mono_slice = np.mean(raw_slice, axis=0)
            
            # 2. RESYNTHESIZE
            # We pass the mono slice to get the "Spectral DNA"
            synth_wave = self._resynthesize_audio(mono_slice)
            
            # 3. CREATE PYGAME SOUND
            # Convert back to stereo for the game engine
            stereo_wave = np.stack([synth_wave, synth_wave], axis=1)
            stereo_wave = (stereo_wave * 32767).astype(np.int16)
            
            try:
                sound = pygame.sndarray.make_sound(np.ascontiguousarray(stereo_wave))
                self.keysounds[diff].append(sound)
            except Exception as e:
                print(f"[VIS] Warning: Could not resynthesize note {i}: {e}")
                self.keysounds[diff].append(None)

    def _resynthesize_audio(self, raw_audio):
        """
        Takes a raw audio buffer, finds the dominant frequencies (Fundamental + Harmonics),
        and rebuilds the sound using pure Sine waves with a percussion envelope.
        Filters out high-freq artifacts and zapping peaks for smooth, clean playback.
        """
        N = len(raw_audio)
        if N == 0: return np.zeros(100)

        # --- STEP A: FFT ANALYSIS ---
        # Windowing reduces spectral leakage
        windowed = raw_audio * np.hanning(N)
        spectrum = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(N, 1/44100)
        magnitudes = np.abs(spectrum)

        # --- STEP B: FILTERING ---
        # 1. Remove Low End Rumble (< 80Hz) and High End Hiss (> 9kHz for cleaner but brighter sound)
        mask = (frequencies > 80) & (frequencies < 9000)
        magnitudes = magnitudes * mask
        
        # 2. Light spectral smoothing to reduce artifacts but preserve tone
        from scipy.ndimage import gaussian_filter1d
        magnitudes = gaussian_filter1d(magnitudes, sigma=1)  # Reduced from sigma=2 for less muddiness
        
        # 3. Find Top K Strongest Frequencies (The "Chord" or "Timbre")
        # We take the top 7 peaks for a richer tone
        num_peaks = 7  # Increased from 6 to capture more character
        # Get indices of top peaks
        peak_indices = np.argpartition(magnitudes, -num_peaks)[-num_peaks:]
        
        top_freqs = frequencies[peak_indices]
        top_mags = magnitudes[peak_indices]
        
        # Normalize magnitudes so the sound isn't too quiet or loud
        if np.max(top_mags) > 0:
            top_mags /= np.max(top_mags)
            # Apply soft limiting to prevent zapping peaks
            top_mags = np.minimum(top_mags, 1.0)

        # --- STEP C: ADDITIVE SYNTHESIS ---
        # Generate a new 200ms buffer for the game SFX
        out_dur = 0.2
        t = np.linspace(0, out_dur, int(44100 * out_dur), False)
        new_wave = np.zeros_like(t)
        
        for f, m in zip(top_freqs, top_mags):
            if f == 0: continue
            # Add a sine wave for this frequency with better amplitude balance
            new_wave += 0.3 * m * np.sin(2 * np.pi * f * t)  # Increased from 0.25 for brightness

        # --- STEP D: ENVELOPE SHAPING (The "Game Feel") ---
        # Apply a sharp "Pluck" envelope: Instant Attack, Exponential Decay.
        # This makes it sound like a key being hit, not a continuous drone.
        envelope = np.exp(-10 * t)  # Adjusted decay for better sustain
        new_wave *= envelope
        
        # --- STEP E: SOFT LIMITING ---
        # Prevent clipping and harshness
        max_val = np.max(np.abs(new_wave))
        if max_val > 0:
            new_wave = 0.95 * new_wave / max_val  # Normalize to 95% of max for more presence
        
        return new_wave

    def run(self):
        clock = pygame.time.Clock()
        self.played_indices = {d: 0 for d in self.beatmaps}
        
        while True:
            clock.tick(60)
            
            if self.playing:
                curr = time.time() - self.start_time
                target_diff = self.diff
                notes = self.beatmaps[target_diff]
                idx = self.played_indices[target_diff]
                
                while idx < len(notes):
                    n = notes[idx]
                    if n["time"] <= curr:
                        if self.mode == "synth":
                            # SYNTH MODE: Play synthesized tones
                            s = self.synth_sounds.get(n["midi"])
                            if s: 
                                chan = pygame.mixer.find_channel()
                                if chan:
                                    v = n.get("vol", 1.0) * self.sfx_vol
                                    l_ratio = n["lane"] / max(1, self.metadata[target_diff]["lanes"]-1)
                                    chan.set_volume((1-l_ratio)*0.7*v, (0.3+l_ratio*0.7)*v)
                                    chan.play(s)
                        elif self.mode == "keysound":
                            # KEYSOUND MODE: Play resynthesized audio
                            sounds = self.keysounds[target_diff]
                            if idx < len(sounds) and sounds[idx]:
                                sounds[idx].set_volume(self.sfx_vol)
                                sounds[idx].play()
                        
                        self.played_indices[target_diff] += 1
                        idx += 1
                    else: break
            else:
                curr = self.pause_time

            for e in pygame.event.get():
                if e.type == pygame.QUIT: return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_1: self.diff = "EASY"
                    if e.key == pygame.K_2: self.diff = "NORMAL"
                    if e.key == pygame.K_3: self.diff = "HARD"
                    if e.key == pygame.K_4: self.diff = "INSANE"
                    if e.key == pygame.K_m: self._toggle_mode()
                    if e.key == pygame.K_SPACE:
                        if self.playing:
                            pygame.mixer.music.pause()
                            self.pause_time = time.time() - self.start_time
                            self.playing = False
                        else:
                            pygame.mixer.music.unpause()
                            self.start_time = time.time() - self.pause_time
                            self.playing = True
                    if e.key == pygame.K_r: 
                        pygame.mixer.music.rewind()
                        pygame.mixer.music.play()
                        self.start_time = time.time()
                        self.played_indices = {d: 0 for d in self.beatmaps}
                        self.playing = True
                    if e.key == pygame.K_LEFTBRACKET: 
                        self.music_vol = max(0, self.music_vol - 0.1)
                        pygame.mixer.music.set_volume(self.music_vol)
                    if e.key == pygame.K_RIGHTBRACKET: 
                        self.music_vol = min(1, self.music_vol + 0.1)
                        pygame.mixer.music.set_volume(self.music_vol)
                    if e.key == pygame.K_MINUS: self.sfx_vol = max(0, self.sfx_vol - 0.1)
                    if e.key == pygame.K_EQUALS: self.sfx_vol = min(1, self.sfx_vol + 0.1)

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
            lanes = meta["lanes"]
            is_selected = (diff == self.diff)
            
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

            pygame.draw.line(self.screen, (100, 255, 100), (x_off, self.hit_line_y), (x_off + col_w, self.hit_line_y), 2)
            
            for n in notes:
                if n["time"] < curr - 0.2: continue
                if n["time"] > curr + 1.5: break
                
                y = self.hit_line_y - (n["time"] - curr) * self.scroll_speed
                x = x_off + n["lane"] * lane_w
                
                base_c = col if is_selected else (col[0]//2, col[1]//2, col[2]//2)
                
                if n.get("dur", 0) > 0:
                    h = n["dur"] * self.scroll_speed
                    pygame.draw.rect(self.screen, (base_c[0]//3, base_c[1]//3, base_c[2]//3), (x+4, y-h, lane_w-8, h))
                
                # Color coding based on source (V37 Debug Feature)
                if n.get("source") == "vocal":
                    note_color = (255, 100, 255) # Pink for Vocals
                elif n.get("source") == "rhythm":
                    note_color = (100, 255, 255) # Cyan for Drums
                else:
                    note_color = base_c # Default difficulty color

                pygame.draw.rect(self.screen, note_color, (x+2, y-10, lane_w-4, 20))
        
        # UI Panel
        panel_y = self.height - 60
        pygame.draw.line(self.screen, (50, 50, 50), (0, panel_y), (self.width, panel_y))
        
        mode_name = "[SYNTH]" if self.mode == "synth" else "[KEYSOUND]"
        info = self.font.render(f"1-4: Diff | M: Toggle Mode {mode_name} | SPACE: Pause | R: Restart | [/]: Music Vol {self.music_vol:.1f} | -/+: SFX Vol {self.sfx_vol:.1f}", True, (150,150,150))
        self.screen.blit(info, (10, panel_y + 15))
        
        status = "PAUSED" if not self.playing else f"Playing {self.diff}"
        status_col = (255,100,100) if not self.playing else (100,255,100)
        status_txt = self.font.render(status, True, status_col)
        self.screen.blit(status_txt, (self.width - 150, panel_y + 15))

    def _toggle_mode(self):
        """Toggle between synth and keysound modes without stopping music"""
        # Simply toggle the mode - music continues uninterrupted
        if self.mode == "synth":
            print("[VIS] Switching to KEYSOUND mode...")
            self.mode = "keysound"
        else:
            print("[VIS] Switching to SYNTH mode...")
            self.mode = "synth"
        
        # Stop all SFX channels (but not music)
        pygame.mixer.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py audio_file")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Derive folder path from audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = base_name
    
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Stems folder not found: {folder_path}")
        sys.exit(1)
    
    app = Visualizer(audio_file, folder_path)
    app.run()