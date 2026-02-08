import os
import librosa
import numpy as np
import sys
from typing import List, Tuple, Any

# Add project root to path if needed (though relative imports are better if this is a package)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beatmap import NoteEvent

class AudioCache:
    """
    Simple cache to prevent re-loading the same audio file multiple times.
    Stores separate buffers for different sample rates.
    """
    _cache = {}

    @classmethod
    def get(cls, path: str, sr: int = 22050):
        key = (path, sr)
        if key not in cls._cache:
            if not os.path.exists(path):
                return None, None
            # print(f"[Cache] Loading {os.path.basename(path)} (sr={sr})...")
            y, s = librosa.load(path, sr=sr)
            cls._cache[key] = (y, s)
        else:
            # print(f"[Cache] Hit: {os.path.basename(path)} (sr={sr})")
            pass
        return cls._cache[key]

    @classmethod
    def clear(cls):
        cls._cache.clear()

class TimingCorrector:
    @staticmethod
    def ground_events(events: List[NoteEvent], audio_path: str, window: float = 0.05) -> List[NoteEvent]:
        """
        Uses DSP (librosa onset detection) to snap event times to the nearest true audio onset.
        This corrects latency/jitter from the ML transcription model.
        """
        try:
            import librosa
            import numpy as np
        except ImportError:
            return events
            
        if not os.path.exists(audio_path) or not events:
            return events
            
        print(f"[Timing] Grounding {len(events)} events with DSP ({os.path.basename(audio_path)})...")
        
        try:
            # Load audio (lightweight load)
            y, sr = AudioCache.get(audio_path, sr=22050)
            
            # Detect Onsets
            # backtracking=True helps find the precise start of the transient
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) == 0:
                return events
                
            # Snap events
            snapped_count = 0
            for e in events:
                # Find nearest onset
                # Search sorted array efficiently? Or just simple search for now (n*m) is slow if large.
                # Use numpy for speed if possible, but events is list of objects.
                # valid range: [e.time - window, e.time + window]
                
                # Simple linear scan optimized by knowing onsets are sorted?
                # Let's use numpy searchsorted
                idx = np.searchsorted(onset_times, e.time)
                
                candidates = []
                if idx < len(onset_times): candidates.append(onset_times[idx])
                if idx > 0: candidates.append(onset_times[idx - 1])
                
                best_onset = -1
                min_dist = window
                
                for t in candidates:
                    dist = abs(t - e.time)
                    if dist < min_dist:
                        min_dist = dist
                        best_onset = t
                        
                if best_onset != -1:
                    # Apply correction
                    e.time = float(best_onset)
                    snapped_count += 1
                    
            print(f"[Timing] Snapped {snapped_count}/{len(events)} events to DSP onsets.")
            return events
            
        except Exception as e:
            print(f"[Timing] DSP Grounding failed: {e}")
            return events

class TimingAlignment:
    @staticmethod
    def calculate_binary_onset_grid(bpm: float, duration: float, resolution: float = 0.01) -> np.ndarray:
        """
        Creates a binary grid of 'ideal' onsets for a given BPM.
        Resolution in seconds (default 10ms).
        """
        if bpm <= 0: return np.zeros(int(duration / resolution) + 1)
        
        beat_dur = 60.0 / bpm
        num_samples = int(duration / resolution) + 1
        grid = np.zeros(num_samples)
        
        # Mark beats on the grid
        # 1/4 notes for primary alignment
        t = 0.0
        while t < duration:
            idx = int(t / resolution)
            if idx < num_samples:
                grid[idx] = 1.0
            t += beat_dur
            
        return grid

    @staticmethod
    def find_best_offset(y: np.ndarray, sr: int, target_bpm: float, max_offset: float = 0.5, resolution: float = 0.01) -> float:
        """
        Finds the optimal offset to align audio onsets with a BPM-based grid.
        Uses cross-correlation of the onset envelope against a binary beat grid.
        """
        duration = len(y) / sr
        
        # 1. Get Onset Envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Resample/Quantize envelope to our resolution
        env_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        
        num_steps = int(duration / resolution) + 1
        quantized_env = np.zeros(num_steps)
        for i, val in enumerate(onset_env):
            t = env_times[i]
            idx = int(t / resolution)
            if idx < num_steps:
                quantized_env[idx] = max(quantized_env[idx], val)
        
        # 2. Create Target Grid
        target_grid = TimingAlignment.calculate_binary_onset_grid(target_bpm, duration, resolution)
        
        # 3. Cross-Correlate
        # We only care about a small window of offsets (e.g. +/- 0.5s)
        from scipy.signal import correlate
        
        max_shift = int(max_offset / resolution)
        
        # Correlation
        corr = correlate(quantized_env, target_grid, mode='same')
        
        # Find peak in the center +/- max_shift
        center = len(corr) // 2
        search_range = corr[center - max_shift : center + max_shift]
        
        if len(search_range) == 0: return 0.0
        
        peak_idx = np.argmax(search_range)
        offset_steps = peak_idx - max_shift
        
        return offset_steps * resolution
