import numpy as np

# ==========================================
#      SYNTH BANK (SOUND DESIGN)
# ==========================================

class SynthBank:
    @staticmethod
    def gen_square_tone(midi):
        """Lead: Instant attack Square wave."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.15 # Shorter for tighter feel
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Square wave
        wave = np.sign(np.sin(2 * np.pi * freq * t))
        
        # Low-pass to remove ear-piercing high frequencies
        # But keep it snappy
        window = np.ones(3) / 3
        wave = np.convolve(wave, window, mode='same')
        
        # ENVELOPE FIX: Instant Attack (0ms)
        # Immediate exponential decay
        env = np.exp(-12 * t) 
        wave = wave * env * 0.5 
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_vocal_tone(midi):
        """Vocal: DETUNED SAW with FAST ATTACK."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.3
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # SuperSaw
        detune = 1.005
        saw1 = 2 * (t * freq % 1) - 1
        saw2 = 2 * (t * (freq * detune) % 1) - 1
        raw_wave = 0.3 * saw1 + 0.3 * saw2
        
        # ENVELOPE FIX: 
        # Old: 40ms attack (Too slow!)
        # New: 5ms attack (Perceptually instant but no click)
        attack_len = int(0.005 * sr) 
        envelope = np.ones_like(t)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        
        decay_start = int(0.1 * sr)
        envelope[decay_start:] = np.exp(-5 * t[:len(t)-decay_start])
        
        wave = raw_wave * envelope * 0.6
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_piano_tone(midi):
        """NEW: Piano Synth (FM Synthesis Approximation)."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.3
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # FM Synthesis for metallic/stringy sound
        # Carrier: freq, Modulator: freq * 2 (1 octave up)
        mod_index = 2.0 * np.exp(-15 * t) # Modulation dies out fast (the "hammer" hit)
        wave = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * freq * 2 * t))
        
        # Percussive Envelope
        env = np.exp(-5 * t)
        wave = wave * env * 0.5
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_bass_tone(midi):
        """Bass: Tight Triangle."""
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.2
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        
        # Tighter decay for rhythm precision
        env = np.exp(-10 * t) 
        wave = wave * env * 0.6
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_drum_sound(beat_pos):
        """Drums: Punchy click."""
        duration = 0.1
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        if beat_pos % 2 == 0: # Kick
            # Fast sweep: 150Hz -> 0Hz in 50ms
            freq_sweep = 150 * np.exp(-60 * t)
            wave = np.sin(2 * np.pi * freq_sweep * t)
            wave *= np.exp(-10 * t)
        else: # Snare
            noise = np.random.uniform(-1, 1, len(t))
            # Very tight gate
            wave = noise * np.exp(-30 * t) 
            
        return (wave * 0.6 * 32767).astype(np.int16)