from dataclasses import dataclass
from typing import List, Optional

@dataclass
class NoteEvent:
    """Represents a single musical note event."""
    time: float       # Start time in seconds
    duration: float   # Duration in seconds
    pitch: float      # MIDI pitch (can be float for microtonal/bends)
    velocity: float   # 0.0 to 1.0
    source: str       # 'piano', 'vocals', etc.
    
    # Game-specific attributes (populated later)
    lane: int = 0
    is_slider: bool = False

@dataclass
class Beatmap:
    """Represents a full rhythm game chart."""
    metadata: dict
    notes: List[NoteEvent]
    bpm: float = 120.0

class EventFilter:
    @staticmethod
    def filter_ghost_notes(notes: List[NoteEvent], min_dur: float = 0.05, min_vel: float = 0.15) -> List[NoteEvent]:
        """Removes notes that are too short or too quiet. Exempts Percussion from duration check."""
        filtered = []
        for n in notes:
            # Velocity Check (Global)
            if n.velocity < min_vel: continue
            
            # Duration Check (Melodic Only)
            # Drums/Bass can be very short (transients)
            if n.source in ["drums", "bass"]:
                filtered.append(n)
            elif n.duration >= min_dur:
                filtered.append(n)
                
        return filtered

    @staticmethod
    def consolidate_rolls(notes: List[NoteEvent], gap_threshold: float = 0.03) -> List[NoteEvent]:
        """Merges notes of the same pitch/source that are remarkably close together."""
        if not notes: return []
        
        # Sort by source, then pitch, then time
        sorted_notes = sorted(notes, key=lambda x: (x.source, x.pitch, x.time))
        merged = []
        
        if not sorted_notes: return []
        
        current = sorted_notes[0]
        
        for i in range(1, len(sorted_notes)):
            next_note = sorted_notes[i]
            
            # Check if same source and pitch
            if (next_note.source == current.source and 
                abs(next_note.pitch - current.pitch) < 0.5):
                
                # Check for overlap or tiny gap
                gap = next_note.time - (current.time + current.duration)
                if gap < gap_threshold:
                    # Merge
                    new_dur = (next_note.time + next_note.duration) - current.time
                    # Max velocity
                    new_vel = max(current.velocity, next_note.velocity)
                    current.duration = new_dur
                    current.velocity = new_vel
                    continue
            
            merged.append(current)
            current = next_note
            
        merged.append(current)
        # Re-sort by time
        return sorted(merged, key=lambda x: x.time)

    @staticmethod
    def gate_silence(notes: List[NoteEvent], audio_path: str, threshold: float = 0.005) -> List[NoteEvent]:
        """Removes notes that occur during silent sections of the audio."""
        import librosa
        import numpy as np
        import os
        
        if not notes: return []
        if not os.path.exists(audio_path): return notes
        
        try:
            # Optimize: Load only if needed? No, we need it for all notes.
            # Using sr=None to preserve original quality for analysis, though lower sr is faster.
            # sr=22050 is fine for RMS.
            y, sr = librosa.load(audio_path, sr=22050)
        except Exception as e:
            print(f"[Filter] Error loading audio for gating: {e}")
            return notes
            
        filtered = []
        dropped = 0
        
        # Pre-calc squared energy for speed? iterating is O(N*W), valid for N~1000
        for n in notes:
            # Check window around note (0.15s window centered)
            sample = int(n.time * sr)
            if sample < 0 or sample >= len(y):
                continue # Out of bounds (drop?) - Let's keep it safe? No, if no audio, no note.
                
            win_size = int(0.15 * sr)
            start = max(0, sample - win_size // 2)
            end = min(len(y), sample + win_size // 2)
            
            segment = y[start:end]
            if len(segment) == 0: continue
            
            rms = np.sqrt(np.mean(segment**2))
            
            if rms >= threshold:
                filtered.append(n)
            else:
                dropped += 1
        
        if dropped > 0:
            print(f"[Filter] Gated {dropped} silent notes (RMS < {threshold})")
            
        return filtered

class Quantizer:
    def __init__(self, bpm: float):
        self.bpm = bpm if bpm > 0 else 120.0
        
    def snap_to_grid(self, events: List[NoteEvent], grids: List[int] = [12, 16]) -> List[NoteEvent]:
        """
        Snaps events to the nearest grid lines defined by `grids`.
        Grids are denominators (4 = quarter note, 8 = eighth, 12 = eighth triplet).
        """
        if self.bpm <= 0: return events
        
        quarter_dur = 60.0 / self.bpm
        
        # Calculate allowed intervals
        allowed_intervals = []
        for g in grids:
            if g == 0: continue
            # Interval = Quarter Duration / (Grid / 4)
            # e.g. Grid 4 -> Interval = Q / 1
            # e.g. Grid 12 -> Interval = Q / 3
            interval = quarter_dur / (g / 4.0)
            allowed_intervals.append(interval)
            
        snapped = []
        for e in events:
            best_time = e.time
            min_diff = float('inf')
            
            # Find closest grid point across all resolutions
            for interval in allowed_intervals:
                # Nearest multiple
                n_units = round(e.time / interval)
                grid_time = n_units * interval
                
                diff = abs(e.time - grid_time)
                if diff < min_diff:
                    min_diff = diff
                    best_time = grid_time
            
            # Create new note event to avoid mutating original list references
            new_n = NoteEvent(
                time=best_time,
                duration=e.duration,
                pitch=e.pitch,
                velocity=e.velocity,
                source=e.source
            )
            # Optional: Snap duration too? 
            # For now, let's keep duration logic simple to avoid broken holds
             
            snapped.append(new_n)
            
        return snapped
