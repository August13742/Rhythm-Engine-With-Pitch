"""
clean_midi.py - Forensic MIDI Cleanup
Verifies MIDI notes against the actual audio spectrogram.
Kills hallucinations, ghost notes, and "mosquito" noise.
"""
import argparse
import numpy as np
import librosa
import pretty_midi
import sys

def scrub_midi(midi_path, audio_path, output_path):
    print(f"[INIT] Loading Audio: {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"[INIT] Computing Spectrogram (CQT)...")
    # CQT is better than FFT for music because it maps to musical notes
    # hop_length=512 is ~11ms precision
    C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=88*3, bins_per_octave=36))
    # Normalize CQT to 0-1 range
    C = librosa.amplitude_to_db(C, ref=np.max)
    C = (C - C.min()) / (C.max() - C.min())

    print(f"[INIT] Loading MIDI: {midi_path}...")
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    total_notes = 0
    kept_notes = 0
    
    # We will create a new cleaned instrument
    new_inst = pretty_midi.Instrument(program=0)
    
    # TUNING KNOBS
    MIN_DURATION = 0.06  # 60ms (kills mosquito notes)
    ENERGY_THRESH = 0.15 # 0-1 Scale (Strictness of spectrogram verification)
    RMS_GATE = 0.02      # Global silence threshold

    # 1. Calculate Global RMS for Silence Gating
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms = (rms - rms.min()) / (rms.max() - rms.min())
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    for inst in pm.instruments:
        for note in inst.notes:
            total_notes += 1
            
            # FILTER 1: DURATION
            if (note.end - note.start) < MIN_DURATION:
                continue

            # FILTER 2: SILENCE GATE (Global RMS)
            # Check average RMS amplitude during note duration
            start_frame = int(note.start * sr / 512)
            end_frame = int(note.end * sr / 512)
            if start_frame >= len(rms): continue
            
            note_rms = np.mean(rms[start_frame : end_frame+1])
            if note_rms < RMS_GATE:
                continue

            # FILTER 3: SPECTROGRAM VERIFICATION (The "Truth" Check)
            # Does the audio actually have energy at this pitch?
            # Map MIDI pitch (0-127) to CQT bin
            # We used bins_per_octave=36 (3 bins per semitone for precision)
            # C1 is MIDI 24.
            fmin_midi = 24
            bin_idx = (note.pitch - fmin_midi) * 3
            
            # Check the specific frequency bin (+/- 1 neighbor for vibrato tolerance)
            if 0 <= bin_idx < C.shape[0]:
                # Extract the energy block for this note in the spectrogram
                # Time slice: start_frame to end_frame
                # Freq slice: bin_idx-1 to bin_idx+2
                cqt_slice = C[bin_idx-1 : bin_idx+2, start_frame : end_frame+1]
                avg_energy = np.mean(cqt_slice)
                
                if avg_energy < ENERGY_THRESH:
                    # Note exists in MIDI but Audio says it's empty -> HALLUCINATION
                    continue

            # If we survived all filters, add to new track
            new_inst.notes.append(note)
            kept_notes += 1

    # Save
    new_pm = pretty_midi.PrettyMIDI()
    new_pm.instruments.append(new_inst)
    new_pm.write(output_path)
    
    print(f"--- CLEANUP COMPLETE ---")
    print(f"Original Notes: {total_notes}")
    print(f"Cleaned Notes:  {kept_notes}")
    print(f"Removed:        {total_notes - kept_notes} (Ghosts/Noise)")
    print(f"Saved to:       {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", help="Input Messy MIDI")
    parser.add_argument("audio", help="Original Vocal Stem (WAV/MP3)")
    parser.add_argument("--out", default="cleaned.mid", help="Output MIDI")
    args = parser.parse_args()
    
    try:
        # Redirect all output to a log file
        with open("clean_midi.log", "w") as log:
            import sys as sys_module
            class LogWriter:
                def __init__(self, file):
                    self.file = file
                    self.terminal = sys.stdout
                def write(self, message):
                    self.file.write(message)
                    self.file.flush()
                    self.terminal.write(message)
                def flush(self):
                    self.file.flush()
            sys.stdout = LogWriter(log)
            sys.stderr = LogWriter(log)
            scrub_midi(args.midi, args.audio, args.out)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)