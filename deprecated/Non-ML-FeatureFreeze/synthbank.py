import numpy as np
from vocal_synth import VocalSynth
from typing import Dict, Tuple
class SynthBank:
    # Duration Buckets (Seconds)
    # 0.15 = Tap, 0.5 = Short Hold, 1.0+ = Long Holds
    DUR_BUCKETS = [0.15, 0.5, 1.0, 2.0, 4.0]
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
                if n.get("source", "other") == "vocals":
                    midi = n["midi"]
                    dur = n.get("dur", 0)
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
                    dur = n.get("dur", 0)
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
    def gen_vocal_tone(midi, duration=None):
        """
        Vocal Generator V3 with Dynamic Duration Support.
        """
        # "PURE_MIKU", "SOFT_LUKA", "POWER_RIN", "CRYSTAL_IA", 
        # "FLOWER_GOTH", "GUMI_WHISPER", "YUKARI_DEEP"
        VOCAL_CHAR = "POWER_RIN" 
        
        play_len = 0.25 if duration is None else duration
            
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
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.3
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        mod_index = 2.0 * np.exp(-15 * t)
        wave = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * freq * 2 * t))
        env = np.exp(-5 * t)
        wave = wave * env * 0.3 # Tuned down
        
        return (wave * 32767).astype(np.int16)

    @staticmethod
    def gen_guitar_tone(midi):
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        duration = 0.2
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), False)
        
        phase = (t * freq) % 1
        wave = np.where(phase < 0.4, 1.0, -1.0).astype(np.float64)
        wave = wave + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        env = np.exp(-12 * t)
        wave = wave * env * 0.3 # Tuned down
        
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
        # 3.0 ensures the volume drops to ~5% (exp(-3)) by the exact end of the note.
        # Short notes (0.15s) get rate ~20 (fast snappy decay).
        # Long notes (2.0s) get rate ~1.5 (slow sustain decay).
        decay_rate = 3.0 / dur
        env = np.exp(-decay_rate * t)

        wave = wave * env * 0.25 # Tuned down
        
        return (wave * 32767).astype(np.int16)