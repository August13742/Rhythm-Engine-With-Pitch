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
    import torchcrepe
except ImportError:
    torchcrepe = None

try:
    from beatmap import NoteEvent
    from transcribe.segmenter import f0_to_note_events, f0_to_note_events_v2
except ImportError:
    from ..beatmap import NoteEvent
    from .segmenter import f0_to_note_events, f0_to_note_events_v2

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



    def transcribe(self, audio_path: str, model_type: Literal["fcpe", "rmvpe", "basic_pitch", "hybrid"] = "hybrid", **kwargs) -> List[NoteEvent]:
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

        elif model_type == "crepe":
            # CREPE (High Quality Monophonic)
            print("[Council] Using CREPE for Monophonic Vocals...")
            if torchcrepe is None:
                logging.error("torchcrepe not installed.")
                return []
            
            # Load Audio
            sr = 16000
            audio, _ = librosa.load(audio_path, sr=sr)
            
            # Predict
            # CREPE expects audio tensor of shape (B, T)
            audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
            
            hop_length = 160 # 10ms
            fmin = 50
            fmax = 1100
            model = "full" # or "tiny"
            decoder = "viterbi" # or "argmax"
            
            print(f"[Council] Running CREPE ({model}/{decoder})...")
            
            # Compute pitch: [B, T_out]
            f0, harmonicity = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length,
                fmin, 
                fmax, 
                model, 
                batch_size=2048,
                device=self.device,
                return_harmonicity=True,
                decoder=torchcrepe.decode.viterbi
            )
            
            f0 = f0.squeeze().cpu().numpy()
            harmonicity = harmonicity.squeeze().cpu().numpy()
            
            # Filter low confidence
            f0[harmonicity < 0.3] = 0
            
            print(f"[Council] Extracting notes from F0 curve ({len(f0)} frames)...")
            return f0_to_note_events_v2(f0, audio, sr, hop_length)

        elif model_type == "hybrid":
            # --- HYBRID CONSENSUS (V300) ---
            # 1. Structure (BasicPitch)
            # 2. Tuning (RMVPE/FCPE)
            # 3. Fusion
            
            print("[Council] Starting Hybrid Consensus Pipeline...")
            
            # Step 1: Structure (Timing) from BasicPitch
            from .basic_pitch import BasicPitchTranscriber
            if "basic_pitch" not in self.models:
                self.models["basic_pitch"] = BasicPitchTranscriber()
                
            # Use 'instrument' settings for BasicPitch to get good segmentation?
            # Or use vocal settings? Vocal settings are usually loose.
            # We want *precise* timing. BasicPitch "vocals" config might be okay.
            structure_notes = self.models["basic_pitch"].transcribe(
                audio_path,
                instrument_name="vocals",
                onset_threshold=0.35, # Default reasonable value
                frame_threshold=0.30,
                min_len=58.0 # From engine.py config
            )
            print(f"  [Hybrid] Structure: Found {len(structure_notes)} note segments.")
            
            # Step 2: Tuning (Pitch Curve) from RMVPE
            # Fallback to FCPE if RMVPE fails?
            # Preference: RMVPE (Better for singing voice extraction typically)
            
            sr = 16000
            audio, _ = librosa.load(audio_path, sr=sr)
            f0 = None
            hop_length = 160 # 10ms
            
            # Try RMVPE first
            self._load_rmvpe()
            if "rmvpe" in self.models:
                print("  [Hybrid] Tuning: Using RMVPE...")
                f0 = self.models["rmvpe"].infer(audio, thred=0.03) # slightly lower text
            else:
                # Fallback FCPE
                self._load_fcpe()
                if "fcpe" in self.models:
                    print("  [Hybrid] Tuning: RMVPE missing, falling back to FCPE...")
                    audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0).unsqueeze(-1)
                    f0_tensor = self.models["fcpe"].infer(
                        audio_tensor, sr=sr, decoder_mode="local_argmax", threshold=0.06
                    )
                    f0 = f0_tensor.squeeze().cpu().numpy()
                    hop_length = len(audio) / len(f0)
            
            if f0 is None:
                print("  [Hybrid] Error: No pitch model available. Returning BasicPitch notes as is.")
                return structure_notes
                
            # Step 3: Fusion
            print("  [Hybrid] Fusing Structure + Tuning...")
            final_notes = []
            
            for note in structure_notes:
                # Map time to frames
                start_frame = int(note.time * sr / hop_length)
                end_frame = int((note.time + note.duration) * sr / hop_length)
                
                # Bounds check
                start_frame = max(0, start_frame)
                end_frame = min(len(f0), end_frame)
                
                if end_frame <= start_frame:
                    continue
                    
                segment = f0[start_frame:end_frame]
                # Filter unvoiced (0 or very low)
                voiced = segment[segment > 10] # 10Hz min
                
                if len(voiced) > 0:
                    # Median Pitch
                    median_pitch = np.median(voiced)
                    
                    # BasicPitch midi might be "65.4" (microtonal) or "65.0"
                    # We overwrite it with the F0 curve pitch
                    # Check if median_pitch is reasonable? 
                    # If BasicPitch says 60 and RMVPE says 72 (Octave error), who do we trust?
                    # RMVPE is pitch-specialized. Trust RMVPE.
                    
                    # Convert f0 to midi
                    import math
                    if median_pitch > 0:
                        midi_pitch = 69 + 12 * math.log2(median_pitch / 440.0)
                        note.pitch = midi_pitch
                        final_notes.append(note)
                else:
                    # Segment was purely unvoiced in RMVPE?
                    # Might be a breath or noise that BasicPitch picked up.
                    # Or a very quiet note.
                    # If we trust BasicPitch, keep it. 
                    # But often BP picks up breaths.
                    # Let's keep it but trust BP's pitch? No, BP pitch is notoriously bad for vocals.
                    # Let's Drop it. (Conservative)
                    pass
            
            print(f"  [Hybrid] Fusion Result: {len(final_notes)} notes.")
            return final_notes

        # Monophonic Models (FCPE/RMVPE) Legacy Direct
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
        # Use V2 onset-aware segmenter for better accuracy
        # Lower min_dur for vocals to catch shorter syllables (60ms vs 80ms default)
        events = f0_to_note_events_v2(f0, audio, sr, hop_length, min_dur_sec=0.06)
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
