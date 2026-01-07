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
