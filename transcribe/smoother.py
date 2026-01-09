from typing import List
from beatmap import NoteEvent
import numpy as np

class VocalSmoother:
    """
    Smoothes vocal note events to remove vibrato, micro-jitters, and short fragments
    for better gameplay playability.
    """
    
    @staticmethod
    def smooth(events: List[NoteEvent], level: float = 0.5) -> List[NoteEvent]:
        """
        Applies smoothing to the event list.
        Level (0.0 - 1.0): Controls aggressiveness.
        """
        if level <= 0.01:
            return events
            
        # Tunable parameters based on level
        min_duration = 0.03 + (0.07 * level) # 30ms to 100ms
        vibrato_window = 0.15 + (0.2 * level) # 150ms to 350ms check
        pitch_tol = 1.5 # Semitones (Vibrato depth)
        
        # Pass 1: Vibrato Flattening
        # Detect alternating patterns A -> B -> A within window
        events = VocalSmoother._flatten_vibrato(events, vibrato_window, pitch_tol)
        
        # Pass 2: Short Note Fusion (Glueline)
        # Merge short notes into previous if close in pitch, or discard
        events = VocalSmoother._fuse_short_notes(events, min_duration)
        
        return events

    @staticmethod
    def _flatten_vibrato(events: List[NoteEvent], window: float, pitch_tol: float) -> List[NoteEvent]:
        if not events: return []
        
        smoothed = []
        i = 0
        n = len(events)
        
        while i < n:
            current = events[i]
            
            # Look ahead for vibrato pattern
            # A (current) -> B -> A ...
            # We want to merge [A, B, A] into a single A if they are close in time/pitch
            
            group = [current]
            j = i + 1
            
            # Form a candidate group
            while j < n:
                next_note = events[j]
                prev_note = events[j-1]
                
                # Time gap check (must be continuous melody)
                if next_note.time - (prev_note.time + prev_note.duration) > 0.05:
                    break
                    
                # Pitch check (must be within vibrato range of the Anchor/Start)
                if abs(next_note.pitch - current.pitch) > pitch_tol:
                    # Allow one deviant (B) if it returns to A?
                    # Simple heuristic: If it goes too far, break
                    break
                    
                # Duration check (vibrato notes are usually short)
                if next_note.duration > window:
                    break
                    
                group.append(next_note)
                j += 1
            
            # If group has > 1 notes, merge them
            if len(group) > 1:
                # Calculate merged properties
                start_time = group[0].time
                end_time = group[-1].time + group[-1].duration
                
                # Main Pitch: Weighted average or Mode?
                # Mode is safer for "trill" A-B-A-B -> A
                # Weighted Average might land in between (quarter tone)
                
                # Let's count pitch duration weight
                pitch_weights = {}
                for g in group:
                    p = round(g.pitch)
                    pitch_weights[p] = pitch_weights.get(p, 0) + g.duration
                
                best_pitch = max(pitch_weights, key=pitch_weights.get)
                
                smoothed.append(NoteEvent(
                    time=start_time,
                    duration=end_time - start_time,
                    pitch=best_pitch,
                    velocity=max(g.velocity for g in group),
                    source=group[0].source
                ))
            else:
                smoothed.append(current)
                
            i += len(group)
            
        return smoothed

    @staticmethod
    def _fuse_short_notes(events: List[NoteEvent], min_dur: float) -> List[NoteEvent]:
        if not events: return []
        
        fused = []
        for i, note in enumerate(events):
            if not fused:
                fused.append(note)
                continue
                
            last = fused[-1]
            
            # Check if current note is short "noise"
            if note.duration < min_dur:
                # Merge into previous if close in time and pitch
                time_gap = note.time - (last.time + last.duration)
                pitch_diff = abs(note.pitch - last.pitch)
                
                if time_gap < 0.05 and pitch_diff < 3.0:
                    # Extend previous note
                    last.duration = (note.time + note.duration) - last.time
                    continue
                # Else: It's a short isolated staccato or noise. 
                # If it's REALLY short (< 30ms), discard it?
                if note.duration < 0.03:
                    continue
            
            fused.append(note)
            
        return fused
