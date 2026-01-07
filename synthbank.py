import numpy as np
from vocal_synth import VocalSynth
from typing import Dict, Tuple
class SynthBank:
    # Duration Buckets (Seconds)
    # 0.15 = Tap, 0.5 = Short Hold, 1.0+ = Long Holds
    DUR_BUCKETS = [0.25,0.50,0.75, 1.0,1.25,1.5,1.75, 2.0,2.25,2.5,3.0, 4.0]
    @staticmethod
    def get_bucket(dur: float) -> float:
        """Finds the closest duration bucket efficiently."""
        if dur <= 0.15: return 0.15
        return min(SynthBank.DUR_BUCKETS, key=lambda x: abs(x - dur))
    
    @staticmethod
    def gen_square_tone(midi, duration=0.15): # Added duration support optional
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        sr = 44100
        # Ensure minimum duration for audibility
        dur = max(duration, 0.15)
        t = np.linspace(0, dur, int(sr * dur), False)
        
        phase = 2 * np.pi * freq * t
        wave = np.where((phase % (2*np.pi)) < np.pi, 1.0, -1.0)
        env = np.exp(-12 * t) 
        wave = wave * env * 0.3
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_smart_vocal_bank(beatmaps: Dict, profile_name="POWER_RIN") -> Dict[Tuple[int, float], np.ndarray]:
        """
        Scans the beatmaps and ONLY generates vocal (midi, bucket) combinations actually used.
        Returns a Dict[(midi, bucket), numpy_array] with raw audio data.
        Only includes notes with source == "vocals".
        """
        required_keys = set()
        print("[SYNTH] Scanning beatmap for required vocal notes...")

        # 1. Scan Beatmaps - only collect vocal notes
        for diff_name, notes in beatmaps.items():
            for n in notes:
                # Only collect actual vocal notes
                if n.get("source", "other") in ["vocals", "vocals_lead"]:
                    midi = n["midi"]
                    dur = n.get("dur", 0.15)
                    bucket = SynthBank.get_bucket(dur)
                    required_keys.add((midi, bucket))

        print(f"[SYNTH] Unique vocal variants to bake: {len(required_keys)}")
        
        # 2. Bake Sounds
        bank = {}
        for midi, bucket in required_keys:
            # Generate specific duration
            audio = VocalSynth.gen_tone(midi_note=midi, duration=bucket, profile_name=profile_name)
            bank[(midi, bucket)] = np.ascontiguousarray(audio)
            
        return bank
    
    @staticmethod
    def gen_smart_other_bank(beatmaps: Dict) -> Dict[Tuple[int, float], np.ndarray]:
        """
        Scans the beatmaps and ONLY generates other (midi, bucket) combinations actually used.
        Returns a Dict[(midi, bucket), numpy_array] with raw audio data.
        Only includes notes with source == "other".
        """
        required_keys = set()
        print("[SYNTH] Scanning beatmap for required other notes...")

        # 1. Scan Beatmaps - collect other notes
        for diff_name, notes in beatmaps.items():
            for n in notes:
                # Only collect actual other notes
                if n.get("source", "other") == "other":
                    midi = n["midi"]
                    dur = n.get("dur", 0.15)
                    bucket = SynthBank.get_bucket(dur)
                    required_keys.add((midi, bucket))

        print(f"[SYNTH] Unique other variants to bake: {len(required_keys)}")
        
        # 2. Bake Sounds
        bank = {}
        for midi, bucket in required_keys:
            # Generate specific duration
            audio = SynthBank.gen_other_tone(midi, duration=bucket)
            bank[(midi, bucket)] = np.ascontiguousarray(audio)
            
        return bank
    
    @staticmethod
    def gen_smart_instrument_bank(beatmaps: Dict, instrument_name: str) -> Dict[Tuple[int, float], np.ndarray]:
        """
        Generic Smart Bank Generator for instruments (Piano, Guitar).
        Scans beatmaps for notes from `instrument_name` and generates exact (midi, bucket) combos.
        """
        required_keys = set()
        print(f"[SYNTH] Scanning beatmap for required {instrument_name} notes...")

        for diff_name, notes in beatmaps.items():
            for n in notes:
                if n.get("source", "") == instrument_name:
                    midi = n["midi"]
                    dur = n.get("dur", 0.15)
                    bucket = SynthBank.get_bucket(dur)
                    required_keys.add((midi, bucket))

        print(f"[SYNTH] Unique {instrument_name} variants to bake: {len(required_keys)}")
        
        bank = {}
        for midi, bucket in required_keys:
            if instrument_name == "guitar":
                audio = SynthBank.gen_guitar_ks(midi, duration=bucket)
            elif instrument_name == "piano":
                audio = SynthBank.gen_piano_fm(midi, duration=bucket)
            else:
                # Fallback to square
                audio = SynthBank.gen_square_tone(midi, duration=bucket)
                
            bank[(midi, bucket)] = np.ascontiguousarray(audio)
            
        return bank

    @staticmethod
    def gen_vocal_tone(midi, duration=None):
        """
        Vocal Generator V3 with Dynamic Duration Support.
        """
        # "PURE_MIKU", "SOFT_LUKA", "POWER_RIN", "CRYSTAL_IA", 
        # "FLOWER_GOTH", "GUMI_WHISPER", "YUKARI_DEEP"
        VOCAL_CHAR = "POWER_RIN" 
        
        play_len = 0.15 if duration is None else duration
            
        # 2. Vowel Selection Logic
        vowels = ['A', 'I', 'U', 'E', 'O']
        v_idx = (midi * 13 + 7) % 5 
        if midi > 80: 
            if v_idx == 1: v_idx = 0 
            if v_idx == 2: v_idx = 4 
        selected_vowel = vowels[v_idx]
        
        return VocalSynth.gen_tone(
            midi_note=midi, 
            vowel=selected_vowel, 
            duration=play_len, 
            profile_name=VOCAL_CHAR
        )

    @staticmethod
    def gen_drums_tone(beat_pos):
        duration = 0.15
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        if beat_pos % 2 == 0:  # Kick
            freq_sweep = 150 * np.exp(-60 * t)
            wave = np.sin(2 * np.pi * freq_sweep * t)
            wave *= np.exp(-10 * t)
        else:  # Snare
            noise = np.random.uniform(-1, 1, len(t))
            wave = noise * np.exp(-30 * t)
            
        # Tuned down to 0.4
        return (wave * 0.4 * 32767).astype(np.int16)

    @staticmethod
    def gen_drums_tone_midi(midi):
        return SynthBank.gen_drums_tone(midi)

    @staticmethod
    def gen_bass_tone(midi):
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.2
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        env = np.exp(-10 * t) 
        wave = wave * env * 0.35 # Tuned down
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_piano_tone(midi):
        """Classic Piano Tone (kept for legacy support)"""
        return SynthBank.gen_piano_fm(midi, 0.3)

    @staticmethod
    def gen_guitar_tone(midi):
        """Classic Guitar Tone (kept for legacy support)"""
        return SynthBank.gen_guitar_ks(midi, 0.2)
        
    @staticmethod
    def gen_guitar_ks(midi: int, duration: float = 0.2) -> np.ndarray:
        """
        Karplus-Strong String Synthesis.
        Physically models a plucked string using a ring buffer delay line.
        """
        sr = 44100
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        
        # Buffer size = SampleRate / Frequency
        N = int(sr / freq)
        
        # Initialize buffer with noise (Pluck excitation)
        buf = np.random.uniform(-1, 1, N).astype(np.float32)
        
        # Output buffer
        n_samples = int(sr * duration)
        out = np.zeros(n_samples, dtype=np.float32)
        
        # Pointer for ring buffer
        ptr = 0
        
        # Karplus-Strong Loop
        for i in range(n_samples):
            out[i] = buf[ptr]
            
            # Lowpass filter feedback
            # avg = 0.5 * (current + previous)
            prev_ptr = (ptr - 1 + N) % N
            avg = 0.992 * 0.5 * (buf[ptr] + buf[prev_ptr]) # 0.992 decay factor ensures sustain
            
            buf[ptr] = avg
            ptr = (ptr + 1) % N
            
        return (out * 0.5 * 32767).astype(np.int16)
        
    @staticmethod
    def gen_piano_fm(midi: int, duration: float = 0.3) -> np.ndarray:
        """
        FM Synthesis E-Piano.
        """
        sr = 44100
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, False)
        
        # Modulator
        mod_freq = freq * 1.0 # Ratio 1:1 for bell-like tone
        # Decay envelop for modulator (brightness decay)
        mod_env = np.exp(-8 * t) 
        mod_amp = 3.0 * mod_env # Modulation index
        
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        
        # Carrier
        carrier = np.sin(2 * np.pi * freq * t + modulator)
        
        # Carrier Envelope (Amplitude decay)
        amp_env = np.exp(-2.5 * t)
        
        wave = carrier * amp_env * 0.4
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_other_tone(midi, duration=0.15):
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        # Ensure minimum duration for audibility
        dur = max(duration, 0.15)
        sr = 44100
        t = np.linspace(0, dur, int(sr * dur), False)
        
        wave = np.sign(np.sin(2 * np.pi * freq * 1.5 * t))
        
        # Calculate a decay rate that ensures the sound lasts the full 'dur'.
        decay_rate = 3.0 / dur
        env = np.exp(-decay_rate * t)

        wave = wave * env * 0.25 # Tuned down
        
        return (wave * 32767).astype(np.int16)