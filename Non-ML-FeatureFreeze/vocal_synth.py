import numpy as np
import pygame
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Formant:
    freq: float
    bw: float
    gain_db: float = 0.0

@dataclass
class VocalProfile:
    name: str
    vowels: Dict[str, List[Formant]]
    formant_shift: float = 1.0
    breathiness: float = 0.05   # Greatly reduced default
    tension: float = 0.5
    master_gain: float = 1.0

class VocalSynth:
    
    @staticmethod
    def get_presets() -> Dict[str, VocalProfile]:
        # Refined Formants (Standard Japanese Soprano)
        BASE = {
            'A': [Formant(800, 80, 0), Formant(1200, 100, -6), Formant(2800, 120, -12)],
            'I': [Formant(300, 50, -6), Formant(2300, 90, -12), Formant(3000, 120, -18)],
            'U': [Formant(350, 60, -5), Formant(800, 80, -10), Formant(2500, 120, -18)],
            'E': [Formant(500, 70, -4), Formant(1800, 90, -10), Formant(2600, 120, -16)],
            'O': [Formant(500, 70, -2), Formant(900, 90, -8), Formant(2600, 120, -16)]
        }

        return {
            "PURE_MIKU": VocalProfile(
                name="Pure Tone",
                vowels=BASE,
                formant_shift=1.12,
                breathiness=0.02,    # Almost no noise (prevents static sound)
                tension=0.4,         # Soft tone, less buzz
                master_gain=1.0
            ),
            "SOFT_LUKA": VocalProfile(
                name="Soft/Breathy",
                vowels=BASE,
                formant_shift=0.95,
                breathiness=0.15,    # Gentle air
                tension=0.1,         # Very sine-like
                master_gain=1.1
            ),
            "POWER_RIN": VocalProfile(
                name="Power",
                vowels=BASE,
                formant_shift=1.08,
                breathiness=0.01,
                tension=0.8,         # Bright harmonics
                master_gain=0.9
            ),
            "CRYSTAL_IA": VocalProfile(
                name="Crystal",
                vowels=BASE,
                formant_shift=1.02,
                breathiness=0.05,
                tension=0.5,
                master_gain=1.0
            ),
            "FLOWER_GOTH": VocalProfile(
                name="Androgynous",
                vowels=BASE,
                formant_shift=0.92,  # Lower/Boyish
                breathiness=0.2,
                tension=0.6,         # Crunchy
                master_gain=1.2
            ),
            "GUMI_WHISPER": VocalProfile(
                name="Sweet Whisper",
                vowels=BASE,
                formant_shift=1.02,
                breathiness=0.7,     # Extremely breathy
                tension=0.1,
                master_gain=1.8      # Needs huge boost
            ),
            "YUKARI_DEEP": VocalProfile(
                name="Deep Resonance",
                vowels=BASE,
                formant_shift=0.88,  # Very deep female
                breathiness=0.1,
                tension=0.4,
                master_gain=1.3
            )
        }

    @staticmethod
    def _klatt_gain(freq, f_c, bw):
        # Optimized Resonator
        if freq <= 1.0 or f_c <= 1.0: return 0.0
        x = freq / f_c
        d = x * x * (bw / f_c)
        denom = (1.0 - x*x)**2 + d*d
        if denom == 0: return 0.0
        return 1.0 / np.sqrt(denom)

    @staticmethod
    def gen_tone(midi_note, vowel='A', duration=0.15, sr=44100, profile_name="POWER_RIN"):
        # --- FIX 1: Enforce Minimum Duration ---
        # Ensure the vocal is at least 0.45s (Tap length) so it sings clearly
        duration = max(duration, 0.45)

        presets = VocalSynth.get_presets()
        profile = presets.get(profile_name, presets["POWER_RIN"])
        
        # 1. Setup
        f0 = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        total_len = int(sr * duration)
        t = np.linspace(0, duration, total_len, False)
        nyquist = sr / 2

        # 2. Pitch Envelope (Subtle Humanization)
        vib = np.sin(2 * np.pi * 5.5 * t) * 0.08 
        freq_mod = f0 * (2.0 ** (vib/12.0))
        phase = 2 * np.pi * np.cumsum(freq_mod) / sr

        # 3. Additive Synthesis
        wave = np.zeros_like(t)
        target_vowel = profile.vowels.get(vowel, profile.vowels['A'])
        
        shifted_formants = [(fmt.freq * profile.formant_shift, fmt.bw, fmt.gain_db) for fmt in target_vowel]

        n = 1
        max_freq_limit = 10000 
        while True:
            freq_n = f0 * n
            if freq_n >= nyquist or freq_n > max_freq_limit: break
            
            slope = -12.0 + (profile.tension * 4.0)
            source_amp = n ** (slope / 6.0)
            
            filt_amp = 0.0
            for f_c, bw, db in shifted_formants:
                filt_amp += VocalSynth._klatt_gain(freq_n, f_c, bw) * (10 ** (db / 20.0))
                
            final_amp = source_amp * filt_amp
            if final_amp > 0.0001:
                wave += final_amp * np.sin(n * phase)
            n += 1

        # 4. Breath
        if profile.breathiness > 0:
            noise = np.random.normal(0, 0.2, len(t))
            breath_mod = 0.5 + 0.5 * np.cos(phase)
            wave += noise * breath_mod * profile.breathiness

        # 5. Envelope (Duration Aware & Robust)
        env = np.ones_like(t)
        atk = int(0.01 * sr)
        rel = int(0.05 * sr)
        
        # Safe Attack
        atk_len = min(atk, len(t))
        env[:atk_len] = np.linspace(0, 1, atk_len)
        
        # --- FIX 2: Safe Release ---
        # Even if duration is very short, this prevents crashing/clicking
        if len(t) < rel:
            env[:] *= np.linspace(1, 0, len(t))
        else:
            env[-rel:] *= np.linspace(1, 0, rel)
            
        wave *= env

        # 6. Normalize & Scale
        peak = np.max(np.abs(wave))
        if peak > 0: wave /= peak
        wave *= (0.25 * profile.master_gain)
        
        # Stereo
        left = wave
        right = np.roll(wave, int(0.005 * sr))
        stereo = np.vstack([left, right]).T

        return (stereo * 32767).astype(np.int16)