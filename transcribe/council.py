import os
import sys
import torch
import numpy as np
import librosa
import logging
from typing import List, Literal

try:
    from torchfcpe import spawn_bundled_infer_model
except ImportError:
    spawn_bundled_infer_model = None

try:
    from .rmvpe_model import RMVPE_Infer
except ImportError:
    RMVPE_Infer = None

try:
    from beatmap import NoteEvent
    from transcribe.segmenter import f0_to_note_events
except ImportError:
    from ..beatmap import NoteEvent
    from .segmenter import f0_to_note_events

class CouncilV2:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        
    def _load_fcpe(self):
        if "fcpe" not in self.models:
            if spawn_bundled_infer_model:
                print(f"[Council] Loading FCPE on {self.device}...")
                self.models["fcpe"] = spawn_bundled_infer_model(device=self.device)
            else:
                logging.warning("torchfcpe not installed.")
                
    def _load_rmvpe(self):
        if "rmvpe" not in self.models:
            if RMVPE_Infer:
                print(f"[Council] Loading RMVPE on {self.device}...")
                # Assuming weights are at 'models/rmvpe.pt' or similar. 
                # User's rmvpe_model.py has a downloader.
                weight_path = os.path.join(os.path.dirname(__file__), "../models/rmvpe.pt")
                self.models["rmvpe"] = RMVPE_Infer(model_path=weight_path, device=self.device)
            else:
                logging.warning("RMVPE_Infer class not found.")



    def transcribe(self, audio_path: str, model_type: Literal["fcpe", "rmvpe", "basic_pitch"] = "fcpe", **kwargs) -> List[NoteEvent]:
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file not found: {audio_path}")
            return []

        print(f"[Council] Transcribing vocals with {model_type.upper()}...")
        
        if model_type == "basic_pitch":
             # Polyphonic Mode
             print("[Council] Using BasicPitch for Polyphonic Vocals...")
             # Lazy import
             from .basic_pitch import BasicPitchTranscriber
             
             if "basic_pitch" not in self.models:
                 self.models["basic_pitch"] = BasicPitchTranscriber()
             
             # Tuned high-recall thresholds for vocals
             return self.models["basic_pitch"].transcribe(
                 audio_path, 
                 instrument_name="vocals",
                 onset_threshold=kwargs.get("onset_threshold", 0.35), 
                 frame_threshold=kwargs.get("frame_threshold", 0.30),
                 multipass_consensus=kwargs.get("multipass_consensus", False)
             )

        # Monophonic Models (FCPE/RMVPE)
        # Load Audio (Shared)
        # Note: FCPE inputs tensor, RMVPE inputs numpy/tensor.
        sr = 16000 # Common denominator
        audio, _ = librosa.load(audio_path, sr=sr)
        
        f0 = None
        hop_length = 160 # Approx 10ms at 16k
        
        if model_type == "fcpe":
            self._load_fcpe()
            if "fcpe" in self.models:
                audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0).unsqueeze(-1)
                # FCPE Infer
                f0_tensor = self.models["fcpe"].infer(
                    audio_tensor, 
                    sr=sr, 
                    decoder_mode="local_argmax", 
                    threshold=0.06,
                    f0_min=50, 
                    f0_max=1100
                )
                f0 = f0_tensor.squeeze().cpu().numpy()
                hop_length = len(audio) / len(f0) # Recalculate exact hop
                
        elif model_type == "rmvpe":
            self._load_rmvpe()
            if "rmvpe" in self.models:
                f0 = self.models["rmvpe"].infer(audio, thred=0.05)
                # RMVPE hop is usually 160 (10ms)
                hop_length = 160 
        
        if f0 is None:
            logging.error(f"Model {model_type} failed to produce F0.")
            return []
            
        print(f"[Council] Extracting notes from F0 curve ({len(f0)} frames)...")
        events = f0_to_note_events(f0, audio, sr, hop_length)
        print(f"[Council] Found {len(events)} vocal notes.")
        return events

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--model", default="fcpe")
    args = parser.parse_args()
    
    c = CouncilV2()
    notes = c.transcribe(args.file, model_type=args.model)
    for n in notes[:5]:
        print(n)
