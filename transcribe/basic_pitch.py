import os
import sys
import logging
from typing import List, Optional

# Add vendorized repo to path
VENDOR_PATH = os.path.join(os.path.dirname(__file__), "basic_pitch_torch_repo")
sys.path.append(VENDOR_PATH)

try:
    from basic_pitch_torch.inference import predict
    BASIC_PITCH_AVAILABLE = True
except ImportError as e:
    BASIC_PITCH_AVAILABLE = False
    logging.warning(f"Basic Pitch Torch not found: {e}")

try:
    from beatmap import NoteEvent
except ImportError:
    from ..beatmap import NoteEvent

class BasicPitchTranscriber:
    def __init__(self):
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError("Basic Pitch library not found.")
        
    def transcribe(self, audio_path: str, instrument_name: str = "instrument", **kwargs) -> List[NoteEvent]:
        """
        Transcribes audio file to NoteEvents using Basic Pitch Torch.
        """
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file not found: {audio_path}")
            return []

        print(f"[BasicPitch] Transcribing {audio_path}...")
        
        # Check for empty/too short files to avoid model crash
        # Basic Pitch requires at least ~0.2s of audio
        import librosa
        try:
            # Check duration
            y_check, sr_check = librosa.load(audio_path, sr=None)
            if librosa.get_duration(y=y_check, sr=sr_check) < 1.0:
                print(f"[BasicPitch] Skipping {os.path.basename(audio_path)}: Duration < 1.0s.")
                return []
        except Exception:
            pass

        # predict returns: model_output, midi_data, note_events
        # note_events is a list of (start_time, end_time, pitch, amplitude, list of frames)
        try:
            # Construct absolute path to the weights in the vendorized repo
            model_path = os.path.join(VENDOR_PATH, "assets", "basic_pitch_pytorch_icassp_2022.pth")
            
            _, _, events = predict(
                audio_path,
                model_path=model_path,
                onset_threshold=kwargs.get("onset_threshold", 0.6), # Stricter default
                frame_threshold=kwargs.get("frame_threshold", 0.4), # Stricter default
                minimum_note_length=kwargs.get("minimum_note_length", 58.0),
                minimum_frequency=None,
                maximum_frequency=None
            )
        except Exception as e:
            logging.error(f"Basic Pitch inference failed: {e}")
            return []

        # Convert to NoteEvent
        note_events = []
        for start, end, pitch, vel, _ in events:
             note_events.append(NoteEvent(
                time=start,
                duration=end - start,
                pitch=pitch,
                velocity=vel,
                source=instrument_name
            ))
            
        print(f"[BasicPitch] Found {len(note_events)} notes for {instrument_name}.")
        
        # PASS 2: Multi-Pass Consensus (Time+Pitch Check)
        if kwargs.get("multipass_consensus", False):
            try:
                import librosa
                import soundfile as sf
                import numpy as np
                
                print(f"[BasicPitch] Running Multi-Pass Consensus (3-Pass: +12st, -12st)...")
                
                def run_pass(semitones):
                    if semitones == 0: return # Skip base
                    
                    # 1. Create Shifted Audio
                    y, sr = librosa.load(audio_path, sr=None)
                    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
                    
                    suffix = f"_p{semitones}" if semitones > 0 else f"_m{abs(semitones)}"
                    temp_path = audio_path.replace(".wav", "").replace(".mp3", "").replace(".ogg", "") + f"{suffix}.wav"
                    sf.write(temp_path, y_shifted, sr)
                    
                    # 2. Transcribe
                    _, _, evs = predict(
                        temp_path,
                        model_path=model_path,
                        onset_threshold=kwargs.get("onset_threshold", 0.6),
                        frame_threshold=kwargs.get("frame_threshold", 0.4),
                        minimum_note_length=kwargs.get("minimum_note_length", 58.0)
                    )
                    
                    if os.path.exists(temp_path): os.remove(temp_path)
                    
                    # 3. Shift Back
                    p_notes = []
                    for s, e, p, v, _ in evs:
                        p_notes.append(NoteEvent(
                            time=s, duration=e-s, pitch=p-semitones, velocity=v, source=instrument_name
                        ))
                    p_notes.sort(key=lambda x: x.time)
                    return p_notes

                # Run Passes
                pass_plus12 = run_pass(12)
                pass_minus12 = run_pass(-12)
                
                # Voting Consensus (2 out of 3)
                # We iterate through Base notes (Pass 0).
                # If a note has support from EITHER +12 or -12, we keep it.
                
                consensus_notes = []
                note_events.sort(key=lambda x: x.time)
                
                kept = 0
                for n0 in note_events:
                    votes = 1 # Base has it
                    
                    # Check +12
                    for n1 in pass_plus12:
                        if abs(n0.time - n1.time) < 0.1 and abs(n0.pitch - n1.pitch) < 0.5:
                            votes += 1
                            break
                        if n1.time > n0.time + 0.1: break
                            
                    # Check -12
                    for n2 in pass_minus12:
                        if abs(n0.time - n2.time) < 0.1 and abs(n0.pitch - n2.pitch) < 0.5:
                            votes += 1
                            break
                        if n2.time > n0.time + 0.1: break
                        
                    if votes >= 2:
                        consensus_notes.append(n0)
                        kept += 1
                
                print(f"[BasicPitch] 3-Pass Consensus: Kept {kept}/{len(note_events)} notes.")
                return consensus_notes
                
            except Exception as e:
                logging.error(f"[BasicPitch] Multi-Pass Failed: {e}")
                return note_events
        
        return note_events

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) < 2:
        print("Usage: python basic_pitch.py <audio_file>")
        sys.exit(1)
        
    fpath = sys.argv[1]
    if not os.path.exists(fpath):
        print("File not found.")
        sys.exit(1)
        
    try:
        t = BasicPitchTranscriber()
        notes = t.transcribe(fpath, "test")
        for n in notes[:5]:
            print(n)
    except ImportError:
        print("Basic Pitch not installed.")
