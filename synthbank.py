import numpy as np

# ==========================================
#      SYNTH BANK (SOUND DESIGN)
# ==========================================
# Each method matches a stem from generator: vocals, drums, bass, piano, guitar, other

class SynthBank:
    @staticmethod
    def gen_square_tone(midi):
        """Classic square wave for fallback."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.15
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        wave = np.sign(np.sin(2 * np.pi * freq * t))
        window = np.ones(3) / 3
        wave = np.convolve(wave, window, mode='same')
        env = np.exp(-12 * t) 
        wave = wave * env * 0.5 
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_vocaloid_tone(midi):
        """Vocal: VOCALOID-like choir synth with shimmer and reverb."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.5
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Multi-layer choir: harmonic richness
        saw1 = 2 * (t * freq % 1) - 1
        saw2 = 2 * (t * (freq * 1.003) % 1) - 1
        saw3 = 2 * (t * (freq * 0.997) % 1) - 1
        harmonic = 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        wave = 0.25 * saw1 + 0.25 * saw2 + 0.25 * saw3 + harmonic
        
        # Smooth envelope
        attack_len = int(0.02 * sr)
        envelope = np.ones_like(t)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        decay_start = int(0.15 * sr)
        envelope[decay_start:] = np.exp(-3 * t[:len(t)-decay_start])
        wave = wave * envelope * 0.55
        
        # Reverb tail
        reverb = np.zeros_like(wave)
        reverb[int(0.05*sr):] = wave[:-int(0.05*sr)] * 0.3
        wave = wave + reverb
        
        return (wave * 32767).astype(np.int16)
    
    @staticmethod
    def gen_vocal_tone(midi):
        """Alias for VOCALOID tone."""
        return SynthBank.gen_vocaloid_tone(midi)

    @staticmethod
    def gen_drums_tone(beat_pos):
        """Drums: Punchy 8-bit percussion."""
        duration = 0.1
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        if beat_pos % 2 == 0:  # Kick
            freq_sweep = 150 * np.exp(-60 * t)
            wave = np.sin(2 * np.pi * freq_sweep * t)
            wave *= np.exp(-10 * t)
        else:  # Snare
            noise = np.random.uniform(-1, 1, len(t))
            wave = noise * np.exp(-30 * t)
            
        return (wave * 0.6 * 32767).astype(np.int16)

    @staticmethod
    def gen_drums_tone_midi(midi):
        """Drums: Alternative MIDI version."""
        return SynthBank.gen_drums_tone(midi)

    @staticmethod
    def gen_bass_tone(midi):
        """Bass: Tight triangle wave, 8-bit style."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.2
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        env = np.exp(-10 * t) 
        wave = wave * env * 0.6
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_piano_tone(midi):
        """Piano: FM percussion, 8-bit compatible."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.3
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        mod_index = 2.0 * np.exp(-15 * t)
        wave = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * freq * 2 * t))
        env = np.exp(-5 * t)
        wave = wave * env * 0.5
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_guitar_tone(midi):
        """Guitar: Bright twangy lead, 8-bit style."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.2
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        phase = (t * freq) % 1
        wave = np.where(phase < 0.4, 1.0, -1.0).astype(np.float64)
        wave = wave + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        env = np.exp(-12 * t)
        wave = wave * env * 0.5
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_other_tone(midi):
        """Other: Metallic percussive, 8-bit style."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.15
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        wave = np.sign(np.sin(2 * np.pi * freq * 1.5 * t))
        env = np.exp(-20 * t)
        wave = wave * env * 0.4
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_drum_sound(beat_pos):
        """Legacy alias for drums."""
        return SynthBank.gen_drums_tone(beat_pos)