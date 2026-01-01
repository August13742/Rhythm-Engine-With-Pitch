'''visualizer.py'''

import argparse
import os
import sys
import time
import json
import pygame
import numpy as np
from generator import MapGenerator # Import logic module

# --- CONFIG ---
COLORS = { "bg": (20, 20, 25), "EASY": (100, 255, 100), "NORMAL": (100, 200, 255), "HARD": (255, 200, 50), "INSANE": (255, 50, 50) }

class Visualizer:
    def __init__(self, audio_path, folder_path):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        pygame.mixer.set_num_channels(64)
        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Rhythm Engine - AI (Demucs) Mode")
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
            
        # 2. SYNTH SOUNDS
        print("Synthesizing SFX Bank...")
        self.synth_sounds = {}
        for midi in range(24, 108):
            self.synth_sounds[midi] = self._gen_tone(midi)
        
        # 3. AUDIO
        pygame.mixer.music.load(audio_path)
        self.music_vol = 0.2
        self.sfx_vol = 0.9
        pygame.mixer.music.set_volume(self.music_vol)
        
        # 4. STATE
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
                        s = self.synth_sounds.get(n["midi"])
                        if s: 
                            chan = pygame.mixer.find_channel()
                            if chan:
                                v = n.get("vol", 1.0) * self.sfx_vol
                                l_ratio = n["lane"] / max(1, self.metadata[target_diff]["lanes"]-1)
                                chan.set_volume((1-l_ratio)*0.7*v, (0.3+l_ratio*0.7)*v)
                                chan.play(s)
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
        
        info = self.font.render(f"1-4: Diff | SPACE: Pause | R: Restart | [/]: Music Vol {self.music_vol:.1f} | -/+: SFX Vol {self.sfx_vol:.1f}", True, (150,150,150))
        self.screen.blit(info, (10, panel_y + 15))
        
        status = "PAUSED" if not self.playing else f"Playing {self.diff}"
        status_col = (255,100,100) if not self.playing else (100,255,100)
        status_txt = self.font.render(status, True, status_col)
        self.screen.blit(status_txt, (self.width - 150, panel_y + 15))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualizer.py song.mp3 song_melody.wav song_rhythm.wav")
        sys.exit(1)
    app = Visualizer(sys.argv[1], sys.argv[2], sys.argv[3])
    app.run()