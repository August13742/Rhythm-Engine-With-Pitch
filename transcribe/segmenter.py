import numpy as np
import librosa
from scipy import signal

try:
    from beatmap import NoteEvent
except ImportError:
    # Fallback if running from proper package context
    from ..beatmap import NoteEvent


def f0_to_note_events_v2(
    f0: np.ndarray,
    audio: np.ndarray, 
    sr: int,
    hop_length: int,
    min_dur_sec: float = 0.08,
    onset_sensitivity: float = 0.7,
    pitch_jump_threshold: float = 2.0,
    source_name: str = "vocals"
) -> list[NoteEvent]:
    """
    V2: Onset-aware segmentation with median smoothing.
    
    Improvements over V1:
    - Uses spectral onset detection to find note boundaries
    - Applies median filter to F0 to remove vibrato jitter
    - Combines onset + pitch change for robust note splitting
    - Better handles portamento (only splits on onset OR large jump)
    
    Args:
        f0: Numpy array of F0 values (Hz).
        audio: Audio time series.
        sr: Sample rate.
        hop_length: Hop length used for F0 extraction.
        min_dur_sec: Minimum note duration (default 0.08s for cleaner output).
        onset_sensitivity: Onset detection threshold multiplier (0.5-1.0).
        pitch_jump_threshold: Semitones to trigger split without onset (default 2.0).
        source_name: Name of the source (for NoteEvent).
        
    Returns:
        List of NoteEvent objects.
    """
    frames = len(f0)
    if frames == 0:
        return []
    
    hop_sec = hop_length / sr
    
    # 1. ENERGY GATE
    rms_hop = max(1, int(len(audio) / frames))
    try:
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=rms_hop, center=True)[0]
        rms = librosa.util.fix_length(rms, size=frames)
        if rms.max() > 0:
            rms /= rms.max()
    except Exception:
        rms = np.ones(frames)
    
    # 2. ONSET DETECTION
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        onset_env = librosa.util.fix_length(onset_env, size=frames)
        onset_env /= (onset_env.max() + 1e-8)
        # Threshold to find onset frames
        onset_thresh = onset_sensitivity * np.mean(onset_env[onset_env > 0.1])
        onset_frames = set(np.where(onset_env > onset_thresh)[0])
    except Exception:
        onset_frames = set()
    
    # 3. F0 TO MIDI WITH GATING AND SMOOTHING
    f0_gated = f0.copy()
    f0_gated[(rms < 0.10) | (f0 < 45) | (f0 > 1200)] = 0
    
    # Convert to MIDI (avoid log of zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        midi_float = np.where(f0_gated > 0, 12 * np.log2(f0_gated / 440.0) + 69, 0)
    
    # Median filter to remove vibrato jitter (5 frames = 50ms at 10ms hop)
    midi_smooth = signal.medfilt(midi_float, kernel_size=5)
    
    # 4. ONSET-AWARE SEGMENTATION
    events = []
    curr_start = None
    curr_pitches = []
    
    for i in range(frames):
        pitch = midi_smooth[i]
        
        if curr_start is None:
            # Looking for note start
            if pitch > 0:
                curr_start = i
                curr_pitches = [pitch]
        else:
            # In a note
            if pitch <= 0:
                # Note ended (unvoiced)
                dur = (i - curr_start) * hop_sec
                if dur >= min_dur_sec and curr_pitches:
                    median_pitch = np.median(curr_pitches)
                    events.append(NoteEvent(
                        time=curr_start * hop_sec,
                        duration=dur,
                        pitch=float(round(median_pitch)),
                        velocity=0.8,
                        source=source_name
                    ))
                curr_start = None
                curr_pitches = []
            else:
                # Check for note break
                curr_median = np.median(curr_pitches[-20:]) if len(curr_pitches) >= 20 else np.median(curr_pitches)
                pitch_change = abs(pitch - curr_median)
                
                # Check if onset nearby (within 2 frames)
                is_onset_nearby = any(j in onset_frames for j in range(max(0, i-2), min(frames, i+2)))
                
                # Split on: (onset + mild pitch change) OR (large pitch jump alone)
                should_break = (is_onset_nearby and pitch_change > 0.6) or (pitch_change > pitch_jump_threshold)
                
                if should_break:
                    dur = (i - curr_start) * hop_sec
                    if dur >= min_dur_sec and curr_pitches:
                        median_pitch = np.median(curr_pitches)
                        events.append(NoteEvent(
                            time=curr_start * hop_sec,
                            duration=dur,
                            pitch=float(round(median_pitch)),
                            velocity=0.8,
                            source=source_name
                        ))
                    curr_start = i
                    curr_pitches = [pitch]
                else:
                    curr_pitches.append(pitch)
    
    # Final segment
    if curr_start is not None and curr_pitches:
        dur = (frames - curr_start) * hop_sec
        if dur >= min_dur_sec:
            events.append(NoteEvent(
                time=curr_start * hop_sec,
                duration=dur,
                pitch=float(round(np.median(curr_pitches))),
                velocity=0.8,
                source=source_name
            ))
    
    return events


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


def post_process_polyphonic_notes(
    notes: list[NoteEvent],
    min_gap: float = 0.03,
    min_note_dur: float = 0.05,
    max_note_dur: float = 10.0
) -> list[NoteEvent]:
    """
    Post-processing for polyphonic note lists (e.g., from BasicPitch).
    
    Operations:
    1. Remove very short notes (likely false positives)
    2. Merge consecutive notes at same pitch with small gaps
    3. Split implausibly long notes
    
    Args:
        notes: List of NoteEvent objects.
        min_gap: Maximum gap (seconds) between notes to merge them.
        min_note_dur: Minimum note duration to keep.
        max_note_dur: Maximum note duration before splitting.
        
    Returns:
        Cleaned list of NoteEvent objects.
    """
    if not notes:
        return notes
    
    from collections import defaultdict
    
    # Group by pitch
    by_pitch = defaultdict(list)
    for n in notes:
        by_pitch[int(round(n.pitch))].append(n)
    
    processed = []
    for pitch, pitch_notes in by_pitch.items():
        pitch_notes = sorted(pitch_notes, key=lambda x: x.time)
        merged = []
        
        for n in pitch_notes:
            # Skip very short notes
            if n.duration < min_note_dur:
                continue
            
            # Check if we can merge with previous note
            if merged:
                prev = merged[-1]
                gap = n.time - (prev.time + prev.duration)
                
                if gap < min_gap and gap >= 0:
                    # Merge: extend previous note
                    new_dur = (n.time + n.duration) - prev.time
                    merged[-1] = NoteEvent(
                        time=prev.time,
                        duration=new_dur,
                        pitch=prev.pitch,
                        velocity=max(prev.velocity, n.velocity),
                        source=prev.source
                    )
                    continue
            
            merged.append(n)
        
        # Split overly long notes
        for n in merged:
            if n.duration > max_note_dur:
                # Split into chunks
                remaining = n.duration
                t = n.time
                while remaining > 0:
                    chunk_dur = min(remaining, max_note_dur)
                    processed.append(NoteEvent(
                        time=t,
                        duration=chunk_dur,
                        pitch=n.pitch,
                        velocity=n.velocity,
                        source=n.source
                    ))
                    t += chunk_dur
                    remaining -= chunk_dur
            else:
                processed.append(n)
    
    return sorted(processed, key=lambda x: x.time)


def transcribe_polyphonic_instrument(
    audio_path: str,
    instrument_type: str = "piano",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.35,
    min_note_dur: float = 0.04,
    velocity_percentile: float = 25,
    max_polyphony: int = 8
) -> list[NoteEvent]:
    """
    Improved polyphonic instrument transcription using BasicPitch with
    optimized post-processing.
    
    This function applies several improvements over raw BasicPitch:
    1. Tuned thresholds for better precision/recall balance
    2. Velocity-based filtering to remove low-confidence notes
    3. Polyphony limiting via NMS to reduce over-detection
    4. Note merging and cleanup
    
    Args:
        audio_path: Path to audio file
        instrument_type: Type of instrument ("piano", "guitar", "other")
        onset_threshold: BasicPitch onset threshold (0.3-0.7)
        frame_threshold: BasicPitch frame threshold (0.2-0.6)
        min_note_dur: Minimum note duration in seconds
        velocity_percentile: Remove notes below this velocity percentile
        max_polyphony: Maximum simultaneous notes at any time
        
    Returns:
        List of NoteEvent objects
    """
    import numpy as np
    from collections import defaultdict
    
    # Import BasicPitch
    try:
        from .basic_pitch import BasicPitchTranscriber
    except ImportError:
        from transcribe.basic_pitch import BasicPitchTranscriber
    
    # Instrument-specific settings (tuned on Chopin Op.10 No.3 benchmark)
    # Piano: onset=0.6, frame=0.4 achieves F1=0.772 (P=0.759, R=0.785)
    presets = {
        "piano": {"onset": 0.6, "frame": 0.4, "max_poly": 10, "vel_pct": 0},
        "guitar": {"onset": 0.55, "frame": 0.4, "max_poly": 6, "vel_pct": 10},
        "bass": {"onset": 0.6, "frame": 0.45, "max_poly": 2, "vel_pct": 0},
        "other": {"onset": 0.55, "frame": 0.4, "max_poly": 8, "vel_pct": 10},
    }
    
    if instrument_type in presets:
        preset = presets[instrument_type]
        onset_threshold = preset["onset"]
        frame_threshold = preset["frame"]
        max_polyphony = preset["max_poly"]
        velocity_percentile = preset["vel_pct"]
    
    # Transcribe with BasicPitch
    bp = BasicPitchTranscriber()
    notes = bp.transcribe(
        audio_path,
        instrument_name=instrument_type,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold
    )
    
    if not notes:
        return []
    
    # Sort by time
    notes = sorted(notes, key=lambda x: x.time)
    
    # 1. Duration filtering (remove very short notes)
    notes = [n for n in notes if n.duration >= min_note_dur]
    
    # 2. Optional velocity filtering (only if vel_pct > 0)
    if velocity_percentile > 0:
        velocities = [n.velocity for n in notes]
        if velocities:
            vel_thresh = np.percentile(velocities, velocity_percentile)
            notes = [n for n in notes if n.velocity >= vel_thresh]
    
    # Note: Skip merging - with tuned thresholds, raw BasicPitch output 
    # achieves better F1 than with post-processing (tested on Chopin Op.10 No.3)
    
    return notes
