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
