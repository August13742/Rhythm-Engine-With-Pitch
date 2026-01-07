'''visualizer.py'''

import argparse
import os
import sys
import time
import json
import pygame
import numpy as np
import librosa
from synthbank import SynthBank

# --- CONFIG ---
COLORS = { "bg": (20, 20, 25), "EASY": (100, 255, 100), "NORMAL": (100, 200, 255), "HARD": (255, 200, 50), "INSANE": (255, 50, 50) }

# Stem Colors (for different note sources)
STEM_COLORS = {
    "vocals": (255, 100, 255),      # Magenta - VOCALOID
    "drums": (100, 255, 255),       # Cyan
    "bass": (150, 100, 50),         # Brown/Gold
    "piano": (100, 200, 255),       # Light Blue
    "guitar": (150, 255, 100),      # Light Green
    "other": (200, 200, 200)        # Light Gray
}

class Visualizer:
    def __init__(self, audio_path, folder_path):
        # Initialize Mixer
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        pygame.mixer.set_num_channels(64)
        
        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Rhythm Engine V84 - 6-Stem Audio")
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 24)
        
        # Audio Setup
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        self.stems = {
            "vocals": os.path.join(folder_path, "vocals.wav"),
            "other":  os.path.join(folder_path, "other.wav"),
            "bass":   os.path.join(folder_path, "bass.wav"),
            "drums":  os.path.join(folder_path, "drums.wav"),
            "piano":  os.path.join(folder_path, "piano.wav"),
            "guitar": os.path.join(folder_path, "guitar.wav")
        }
        
        self.beatmaps = {}
        self.metadata = {}
        
        # LOAD BEATMAPS FROM FILES
        print("[VIS] Loading beatmaps from files...")
        
        # Load manifest to find a non-empty track for duration
        manifest_path = os.path.join(folder_path, "stems_manifest.json")
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        
        # Find a non-empty track for duration calculation
        duration_track = None
        for stem_name in ["drums", "bass", "piano", "guitar", "vocals", "other"]:
            if not manifest.get(stem_name, {}).get("is_silent", False):
                duration_track = stem_name
                break
        
        if duration_track is None:
            duration_track = "drums"  # Fallback to drums
        
        duration = librosa.get_duration(path=os.path.join(folder_path, f"{duration_track}.wav"))
        from generator import DIFF_CONFIGS
        for d in ["EASY", "NORMAL", "HARD", "INSANE"]:
            beatmap_file = os.path.join(folder_path, f"{base_name}_{d}.json")
            with open(beatmap_file, 'r') as f:
                self.beatmaps[d] = json.load(f)
            
            nps = len(self.beatmaps[d]) / duration if duration > 0 else 0
            lanes = DIFF_CONFIGS[d]["lanes"]
            self.metadata[d] = {"count": len(self.beatmaps[d]), "lanes": lanes, "nps": nps}
            print(f"[VIS] Loaded {d}: {len(self.beatmaps[d])} notes")
        
        # 2. SYNTHESIZE BANKS
        print("[VIS] Synthesizing Sound Banks...")
        self.square_bank = {}  # 8-bit square wave
        self.vocal_bank = {}   # VOCALOID choir
        self.synth_banks = {}  # Stem-specific synths
        
        # Use smart banks for vocals and other (only bake what's used)
        vocal_bank_raw = SynthBank.gen_smart_vocal_bank(self.beatmaps)
        other_bank_raw = SynthBank.gen_smart_other_bank(self.beatmaps)
        
        print(f"[VIS] Vocal bank has {len(vocal_bank_raw)} combinations")
        if vocal_bank_raw:
            for key in list(vocal_bank_raw.keys())[:3]:
                print(f"[VIS]   Example: {key}, shape={vocal_bank_raw[key].shape}")
        
        print(f"[VIS] Other bank has {len(other_bank_raw)} combinations")
        if other_bank_raw:
            for key in list(other_bank_raw.keys())[:3]:
                print(f"[VIS]   Example: {key}, shape={other_bank_raw[key].shape}")
        
        # Convert raw audio to pygame Sound objects
        self.synth_banks["vocals"] = {}
        for (midi, bucket), audio in vocal_bank_raw.items():
            self.synth_banks["vocals"][(midi, bucket)] = pygame.sndarray.make_sound(
                np.ascontiguousarray(audio)
            )
        self.vocal_bank = self.synth_banks["vocals"]
        
        self.synth_banks["other"] = {}
        for (midi, bucket), audio in other_bank_raw.items():
            self.synth_banks["other"][(midi, bucket)] = pygame.sndarray.make_sound(
                np.stack([audio, audio], axis=1)
            )
        
        # For other stems (drums, bass, piano, guitar), still generate all MIDI values with default duration
        for midi in range(24, 108):
            # Square wave (8-bit)
            square_audio = SynthBank.gen_square_tone(midi)
            self.square_bank[midi] = pygame.sndarray.make_sound(
                np.stack([square_audio, square_audio], axis=1)
            )
            
            # Stem-specific synths (drums, bass, piano, guitar with default durations)
            self.synth_banks["drums"] = self.synth_banks.get("drums", {})
            drums_audio = SynthBank.gen_drums_tone_midi(midi)
            self.synth_banks["drums"][midi] = pygame.sndarray.make_sound(
                np.stack([drums_audio, drums_audio], axis=1)
            )
            
            self.synth_banks["bass"] = self.synth_banks.get("bass", {})
            bass_audio = SynthBank.gen_bass_tone(midi)
            self.synth_banks["bass"][midi] = pygame.sndarray.make_sound(
                np.stack([bass_audio, bass_audio], axis=1)
            )
            
            self.synth_banks["piano"] = self.synth_banks.get("piano", {})
            piano_audio = SynthBank.gen_piano_tone(midi)
            self.synth_banks["piano"][midi] = pygame.sndarray.make_sound(
                np.stack([piano_audio, piano_audio], axis=1)
            )
            
            self.synth_banks["guitar"] = self.synth_banks.get("guitar", {})
            guitar_audio = SynthBank.gen_guitar_tone(midi)
            self.synth_banks["guitar"][midi] = pygame.sndarray.make_sound(
                np.stack([guitar_audio, guitar_audio], axis=1)
            )
        
        # 3. SETUP
        pygame.mixer.music.load(audio_path)
        self.mode = "vocal"  # Default to vocal mode
        self.music_vol = 0.2
        self.sfx_vol = 0.8
        pygame.mixer.music.set_volume(self.music_vol)
        
        self.diff = "INSANE"
        self.playing = True
        self.start_time = time.time()
        self.pause_time = 0
        self.scroll_speed = 600
        self.hit_line_y = self.height - 100
        self.played_indices = {}
        pygame.mixer.music.play()

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
                        sound = None
                        source = n.get("source", "other")
                        midi = n["midi"]
                        dur = n.get("dur", 0)
                        bucket = SynthBank.get_bucket(dur)
                        
                        # --- SYNTH MODES ---
                        if self.mode == "classic":
                            # All square wave (pure 8-bit)
                            sound = self.square_bank.get(midi)
                            
                        elif self.mode == "vocal":
                            # VOCALOID vocals + default other synth for non-vocals (both with duration)
                            if source == "vocals":
                                sound = self.synth_banks["vocals"].get((midi, bucket))
                            elif source == "other":
                                sound = self.synth_banks["other"].get((midi, bucket))
                            else:
                                # Fallback to square wave if source is unknown
                                sound = self.square_bank.get(midi)
                                
                        elif self.mode == "experimental":
                            # VOCALOID vocals + stem-specific synths (vocals and other use smart banks with duration)
                            if source == "vocals":
                                sound = self.synth_banks["vocals"].get((midi, bucket))
                            elif source == "other":
                                sound = self.synth_banks["other"].get((midi, bucket))
                            else:
                                sound = self.synth_banks.get(source, {}).get(midi)

                        if sound: 
                            chan = pygame.mixer.find_channel()
                            if chan:
                                v = n.get("vol", 1.0) * self.sfx_vol
                                l_ratio = n["lane"] / max(1, self.metadata[target_diff]["lanes"]-1)
                                
                                # Vocals centered
                                if source == "vocals" and self.mode != "classic":
                                    chan.set_volume(v, v)
                                else:
                                    chan.set_volume((1-l_ratio)*0.7*v, (0.3+l_ratio*0.7)*v)
                                chan.play(sound)
                        elif self.played_indices[target_diff] < 1:
                            # Debug: show why sound wasn't found
                            key = (midi, bucket)
                            if source == "other":
                                print(f"[VIS] No sound found for other: key={key} in bank={list(self.synth_banks['other'].keys())[:3]}")
                            elif source == "vocals":
                                print(f"[VIS] No sound found for vocals: key={key} in bank={list(self.synth_banks['vocals'].keys())[:3]}")
                            else:
                                print(f"[VIS] No sound found for {source}: midi={midi}")
                        
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
                    if e.key == pygame.K_m: self._cycle_mode()
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

    def _cycle_mode(self):
        modes = ["classic", "vocal", "experimental"]
        curr_idx = modes.index(self.mode)
        self.mode = modes[(curr_idx + 1) % len(modes)]
        print(f"[VIS] Mode switched to: {self.mode.upper()}")
        for i in range(pygame.mixer.get_num_channels()):
            pygame.mixer.Channel(i).stop()

    def draw(self, curr):
        self.screen.fill(COLORS["bg"])
        col_w = self.width // 4
        diffs = ["EASY", "NORMAL", "HARD", "INSANE"]
        
        for i, diff in enumerate(diffs):
            x_off = i * col_w
            col = COLORS.get(diff, (255,255,255))
            notes = self.beatmaps[diff]
            meta = self.metadata[diff]
            lanes = self.metadata[diff]["lanes"]
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
                
                # Get stem-specific color
                source = n.get("source", "other")
                note_color = STEM_COLORS.get(source, (200, 200, 200))
                
                # Dim if not selected
                if not is_selected:
                    note_color = (note_color[0]//2, note_color[1]//2, note_color[2]//2)

                pygame.draw.rect(self.screen, note_color, (x+2, y-10, lane_w-4, 20))
        
        # UI Panel
        panel_y = self.height - 60
        pygame.draw.line(self.screen, (50, 50, 50), (0, panel_y), (self.width, panel_y))
        
        info = self.font.render(f"1-4: Diff | M: {self.mode.upper()} | SPACE: Pause | R: Restart | [/]: Music | -/+: SFX", True, (150,150,150))
        self.screen.blit(info, (10, panel_y + 15))
        
        status = "PAUSED" if not self.playing else f"Playing {self.diff}"
        status_col = (255,100,100) if not self.playing else (100,255,100)
        status_txt = self.font.render(status, True, status_col)
        self.screen.blit(status_txt, (self.width - 150, panel_y + 15))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py audio_file")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"[ERROR] Audio file not found: {audio_file}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = base_name
    
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Stems folder not found: {folder_path}")
        sys.exit(1)
    
    app = Visualizer(audio_file, folder_path)
    app.run()