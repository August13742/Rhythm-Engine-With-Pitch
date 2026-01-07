import numpy as np
import librosa
try:
    from beatmap import NoteEvent
except ImportError:
    # Fallback if running from proper package context
    from ..beatmap import NoteEvent

def f0_to_note_events(f0: np.ndarray, audio: np.ndarray, sr: int, hop_length: int, min_dur_sec: float = 0.06, source_name: str = "vocals") -> list[NoteEvent]:
    """
    Converts a continuous F0 curve into discrete NoteEvents using hysteresis and energy gating.
    Adapted from f0_to_midi_robust.
    
    Args:
        f0: Numpy array of F0 values.
        audio: Audio time series.
        sr: Sample rate.
        hop_length: Hop length used for F0 extraction.
        min_dur_sec: Minimum duration for a note to be kept.
        source_name: Name of the source (for NoteEvent).
        
    Returns:
        List of NoteEvent objects.
    """
    frames = len(f0)
    if frames == 0:
        return []
    
    # 1. ENERGY GATE
    frame_len = 2048
    rms_hop = int(len(audio) / frames) if frames > 0 else 512
    # Ensure rms_hop is at least 1 to avoid ZeroDivision
    rms_hop = max(1, rms_hop)

    try:
        rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=rms_hop, center=True)[0]
        rms = librosa.util.fix_length(rms, size=frames)
        if rms.max() > 0: rms /= rms.max()
        
        # Kill anything below 12% volume (removes background hiss)
        f0[rms < 0.12] = 0
    except Exception as e:
        print(f"[Segmenter] RMS calc failed: {e}. proceeding without energy gate.")

    # 2. CONVERT TO MIDI FLOAT
    # Filter realistic range 45 (A1) to 90 (F#6) - Standard Pop Range
    # Mask invalid values
    f0_clean = np.where((f0 > 45) & (f0 < 1200), f0, np.nan)
    midi_float = 12 * np.log2(f0_clean / 440.0) + 69
    midi_float = np.nan_to_num(midi_float, nan=0.0)

    # 3. HYSTERESIS QUANTIZATION (The Vibrato Fix)
    midi_stable = np.zeros_like(midi_float, dtype=int)
    last_pitch = 0
    
    for i in range(frames):
        val = midi_float[i]
        if val <= 0:
            last_pitch = 0
            midi_stable[i] = 0
            continue
            
        if last_pitch == 0:
            # New note start: just round it
            curr = int(round(val))
        else:
            # Sustain: Only change if we drift > 0.6 semitones away
            diff = abs(val - last_pitch)
            if diff > 0.6:
                curr = int(round(val))
            else:
                curr = last_pitch
        
        midi_stable[i] = curr
        last_pitch = curr

    # 4. SEGMENTATION
    segments = []
    if len(midi_stable) > 0:
        curr_p = midi_stable[0]
        curr_start = 0
        for i in range(1, frames):
            if midi_stable[i] != curr_p:
                segments.append({"p": curr_p, "s": curr_start, "e": i})
                curr_p = midi_stable[i]
                curr_start = i
        segments.append({"p": curr_p, "s": curr_start, "e": frames})

    final_events = []
    for seg in segments:
        dur = (seg["e"] - seg["s"]) * hop_length / sr
        
        # Keep valid notes
        if seg["p"] > 0 and dur >= min_dur_sec:
            start_time = seg["s"] * hop_length / sr
            # Estimate velocity from RMS if available, else 0.8
            vel = 0.8 # Placeholder, could average RMS over segment
            
            final_events.append(NoteEvent(
                time=start_time,
                duration=dur,
                pitch=float(seg["p"]),
                velocity=vel,
                source=source_name
            ))

    return final_events
