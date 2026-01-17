import os
import sys
import json
import librosa
import numpy as np
from typing import List

from beatmap import NoteEvent, EventFilter
from transcribe.basic_pitch import BasicPitchTranscriber
from transcribe.council import CouncilV2
from utils import AudioCache, TimingCorrector
from chart_generator import ChartGenerator

# Configuration for Transcription (Per Stem)
# Tuned for high precision and musicality
STEM_TRANSCRIBE_CONFIG = {
    "vocals": {
        "onset": 0.35, "frame": 0.30, 
        "smoothing": 0.7, "multipass": False, # Vocals rely on FCPE/ConsensusEngine
        "min_len": 58.0
    },
    "piano": {
        "onset": 0.40, "frame": 0.30, # Slightly stricter onset
        "smoothing": 0.0, # NO SMOOTHING (Preserve fast runs)
        "multipass": True, # Fix pitch leaks
        "min_len": 30.0 # Allow shorter notes
    },
    "guitar": {
        "onset": 0.40, "frame": 0.30,
        "smoothing": 0.0, # NO SMOOTHING (Plucks are sharp)
        "multipass": True,
        "min_len": 40.0
    },
    "bass": {
        "onset": 0.50, "frame": 0.40, # Stricter (Bass is often muddy)
        "smoothing": 0.5, # Some smoothing for sustained bass
        "multipass": True,
        "min_len": 80.0
    },
    "other": {
        "onset": 0.55, "frame": 0.40,  # Stricter for composite tracks
        "smoothing": 0.0,
        "multipass": True,  # Enable consensus to reduce noise
        "min_len": 58.0,
        "velocity_gate": 0.25  # Remove low-confidence notes
    }
}

class ConsensusEngine:
    @staticmethod
    def fuse_vocals(lead_notes: List[NoteEvent], poly_notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Fuses High-Quality Monophonic Lead (FCPE) with Polyphonic Harmonies (BasicPitch).
        Advanced Logic:
        1. TRUST BP (Chords): If BP detects a chord (>=2 notes) and FCPE is in the middle (averaging error), 
           discard FCPE and use BP notes.
        2. LEAD FIRST: Otherwise, keep Lead.
        3. HARMONIES: Add non-overlapping BP notes as Harmonies/Fillers.
        """
        if not poly_notes: return lead_notes
        if not lead_notes: return poly_notes
        
        # Sort
        lead_notes.sort(key=lambda x: x.time)
        poly_notes.sort(key=lambda x: x.time)
        
        final_events = []
        valid_intervals = {3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 19, 24}
        
        # Track which BP notes are used
        used_poly_indices = set()
        
        # Iterate LEAD notes first to check for Chords/Conflicts
        for l_idx, l in enumerate(lead_notes):
            # Find overlapping BP notes
            overlaps = []
            overlap_indices = []
            
            for p_idx, p in enumerate(poly_notes):
                # Check overlap
                if (p.time < l.time + l.duration) and (p.time + p.duration > l.time):
                     overlaps.append(p)
                     overlap_indices.append(p_idx)
            
            # CONSENSUS CHECK
            # Case 1: BP detects Chord (>=2)
            if len(overlaps) >= 2:
                # Check if FCPE matches any of them
                match_found = False
                for p in overlaps:
                     if abs(p.pitch - l.pitch) < 1.0: # Close enough
                         match_found = True
                         break
                
                if not match_found:
                     # TRUST BP: FCPE is likely averaging. Discard Lead.
                     # Add all overlapping BP notes to final (Mark them as replacement?)
                     # We treat them as if they are the correct source.
                     # But we must ensure they aren't added twice (handled by used_poly_indices?)
                     # No, this loop is driving the Lead edition.
                     
                     print(f"[Consensus] Discarding FCPE note at {l.time:.2f}s (Pitch {l.pitch:.1f}) in favor of BP Chord.")
                     for idx in overlap_indices:
                         if idx not in used_poly_indices:
                             # Use BP note. Should it be Harmony? One should be lead.
                             # Let's keep them as is (source='vocals') or set is_harmony?
                             # Set closest to original lead as lead? Or just all harmony?
                             # Let's mark all as "is_harmony=False" (Lead) to ensure at least one is charted?
                             # Actually simplest is just append them.
                             final_events.append(poly_notes[idx])
                             used_poly_indices.add(idx)
                     continue # Skip adding the Lead 'l'

            # Case 2: Normal Lead Processing
            final_events.append(l) # Keep Lead
            
            # Check harmonies for this lead
            for idx, p in zip(overlap_indices, overlaps):
                if idx in used_poly_indices: continue
                
                diff = abs(p.pitch - l.pitch)
                semitone = round(diff)
                
                if semitone == 0:
                    used_poly_indices.add(idx) # Mark as used (absorbed by lead)
                elif semitone in valid_intervals:
                    # Good Harmony
                    p.is_harmony = True
                    final_events.append(p)
                    used_poly_indices.add(idx)
                elif semitone <= 2:
                    # Dissonance - Reject
                    used_poly_indices.add(idx) # Mark as processed (rejected)
        
        # Add remaining BP notes (Fillers)
        for i, p in enumerate(poly_notes):
            if i not in used_poly_indices:
                p.is_harmony = True # Filler is harmony/backing
                final_events.append(p)
                
        print(f"[Consensus] Fused {len(final_events)} notes from {len(lead_notes)} Lead + {len(poly_notes)} Poly.")
        return final_events

class RhythmEngine:
    def __init__(self, stems_folder: str, beatmap_folder: str, audio_engine_latency_offset: float = -0.02):
        self.stems_folder = stems_folder
        self.beatmap_folder = beatmap_folder
        self.base_name = os.path.basename(os.path.dirname(stems_folder)) if os.path.basename(stems_folder) in ["stems", "beatmap"] else os.path.basename(stems_folder)
        # Actually stems_folder is usually ".../stems/songname"
        if os.path.dirname(stems_folder).endswith("stems"):
             self.base_name = os.path.basename(stems_folder)
        
        # Ensure paths exist
        if not os.path.isdir(self.stems_folder):
            raise ValueError(f"Stems folder does not exist: {self.stems_folder}")
        os.makedirs(self.beatmap_folder, exist_ok=True)
        
        # Initialize Transcribers
        self.bp_transcriber = BasicPitchTranscriber()
        self.council = CouncilV2()
        
        # Estimate BPM or default
        self.bpm = self._detect_bpm() 
        print(f"[RhythmEngine] BPM set to: {self.bpm}")

        # Model Latency Compensation
        # Benchmark says BasicPitch is ~8ms EARLY (-0.008s).
        # However, we often perceive things as late due to audio output latency.
        # Let's add a configurable offset. 
        # Positive = shift notes LATER (to fix early notes)
        # Negative = shift notes EARLIER (to fix late notes)
        
        # Parameterized Latency (Constructor Argument)
        self.audio_engine_latency_offset = audio_engine_latency_offset
        print(f"[RhythmEngine] Latency Compensation: {self.audio_engine_latency_offset*1000:.1f}ms")
        
        self.generator = ChartGenerator(bpm=self.bpm)
        self.manifest = self._load_manifest()


    def _detect_bpm(self) -> float:
        """
        Detects BPM using Madmom (DBNBeatTracker) with Librosa fallback.
        Includes Sanity Check to prefer 100-180 BPM range.
        """
        target_path = os.path.join(self.stems_folder, "drums.wav")
        if not os.path.exists(target_path):
             for f in ["other.wav", "bass.wav", "vocals.wav"]:
                 p = os.path.join(self.stems_folder, f)
                 if os.path.exists(p):
                     target_path = p
                     break
        
        if not os.path.exists(target_path):
            print("[RhythmEngine] No audio files found for BPM detection. Defaulting to 120.0")
            return 120.0

        print(f"[RhythmEngine] Detecting BPM from {os.path.basename(target_path)}...")
        
        # 1. Try Madmom (DBNBeatTracker)
        try:
            # Output of madmom relies on 'collections', which removed MutableSequence in Py3.10+
            import collections
            if not hasattr(collections, 'MutableSequence'):
                import collections.abc
                collections.MutableSequence = collections.abc.MutableSequence
                collections.Iterable = collections.abc.Iterable
            
            # Madmom also relies on np.float, np.int, np.bool which were removed in Numpy 1.24+
            import numpy as np
            if not hasattr(np, 'float'):
                np.float = float
            if not hasattr(np, 'int'):
                np.int = int
            if not hasattr(np, 'bool'):
                np.bool = bool
            
            import madmom
            import madmom.features.beats
            print("  [BPM] Using Madmom DBNBeatTracker...")
            
            # Madmom proc handles loading internally effectively, but let's pass file path
            # DBNBeatTracker might be DBNBeatTrackingProcessor in this version
            if hasattr(madmom.features.beats, 'DBNBeatTracker'):
                proc = madmom.features.beats.DBNBeatTracker(fps=100)
            else:
                proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                
            act = madmom.features.beats.RNNBeatProcessor()(target_path)
            beats = proc(act)
            
            # Calculate BPM from beats (inter-beat interval)
            if len(beats) > 1:
                intervals = np.diff(beats)
                median_interval = np.median(intervals)
                bpm = 60.0 / median_interval
                print(f"  [BPM] Madmom Raw: {bpm:.2f}")
                return self._sanitize_bpm(bpm)
            
        except ImportError:
            print("  [BPM] Madmom not found. Falling back to Librosa.")
        except Exception as e:
            print(f"  [BPM] Madmom failed: {e}. Falling back to Librosa.")

        # 2. Fallback: Librosa
        try:
            # Use Cache for consistency (though Madmom used its own loader above)
            y, sr = AudioCache.get(target_path, sr=22050)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            if isinstance(tempo, np.ndarray): tempo = tempo.item()
            print(f"  [BPM] Librosa Raw: {tempo:.2f}")
            return self._sanitize_bpm(float(tempo))

        except Exception as e:
            print(f"[RhythmEngine] BPM Detection Error: {e}. Defaulting to 120.0")
            return 120.0

    def _sanitize_bpm(self, bpm: float) -> float:
        """
        Sanity Checker: Bias towards 100-180 BPM.
        Corrects for Double/Half time errors.
        """
        if bpm <= 0: return 120.0
        
        original = bpm
        # Logic: If < 90, try 2x. If > 180, try 0.5x.
        # This is a heuristic.
        while bpm < 90:
            bpm *= 2
        while bpm > 185:
            bpm /= 2
            
        if bpm != original:
            print(f"  [BPM] Sanity Check: {original:.2f} -> {bpm:.2f}")
        return bpm

    def _load_manifest(self):
        m_path = os.path.join(self.stems_folder, "stems_manifest.json")
        if os.path.exists(m_path):
            try:
                with open(m_path, 'r') as f:
                    return json.load(f)
            except: 
                return {}
        return {}

    def run(self):
        print(f"Starting Rhythm Engine V300 on: {self.stems_folder} (Focus: Dynamic)")
        
        # 0. BPM Detection (Already done in __init__)
        print(f"[RhythmEngine] Using BPM: {self.bpm}")
        # self.generator.bpm = self.bpm # Already set during init
        # self.generator.quantizer.bpm = self.bpm # Already set during init
        
        
        all_events: List[NoteEvent] = []
        
        # 1. Transcribe Instruments
        # Use Manifest to skip silent stems
        for stem in ["piano", "guitar", "bass", "other", "drums"]:
            # Check manifest first
            if stem in self.manifest:
                if self.manifest[stem].get("is_silent", False):
                    print(f"Skipping {stem} (Silent per manifest).")
                    continue
            
            path = os.path.join(self.stems_folder, f"{stem}.wav")
            if os.path.exists(path):
                print(f"Processing {stem}...")
                
                # TUNING:
                # Use STEM_TRANSCRIBE_CONFIG
                # Drums -> Librosa Onset (High Sensitivity, ignore pitch)
                
                if stem == "drums":
                    print(f"Processing {stem} with Librosa Onset Detection (Beat Feel)...")
                    notes = self._transcribe_drums_onset(path)
                else:
                    # BasicPitch for melodic instruments
                    cfg = STEM_TRANSCRIBE_CONFIG.get(stem, STEM_TRANSCRIBE_CONFIG["other"])
                    print(f"Processing {stem} with BasicPitch (Config: {cfg})...")
                    
                    params = {
                        "onset_threshold": cfg["onset"], 
                        "frame_threshold": cfg["frame"],
                        "multipass_consensus": cfg["multipass"],
                        "minimum_note_length": cfg["min_len"]
                    }
                    
                    notes = self.bp_transcriber.transcribe(path, instrument_name=stem, **params)
                    
                    # Velocity Gate: Remove low-confidence notes (especially for 'other')
                    vel_gate = cfg.get("velocity_gate", 0)
                    if vel_gate > 0 and notes:
                        import numpy as np
                        velocities = [n.velocity for n in notes]
                        thresh = np.percentile(velocities, vel_gate * 100)
                        before_count = len(notes)
                        notes = [n for n in notes if n.velocity >= thresh]
                        print(f"  [VelGate] {stem}: {before_count} -> {len(notes)} notes (thresh={thresh:.3f})")
                    
                    # Apply Smoothing if configured
                    if cfg["smoothing"] > 0:
                        from transcribe.smoother import VocalSmoother
                        notes = VocalSmoother.smooth(notes, level=cfg["smoothing"])
                
                # Drum Fix: Force Fixed Pitch (e.g., C4 = 60)
                if stem == "drums":
                    for n in notes: n.pitch = 60
                
                # DSP Grounding (New)
                # Ground instrument notes to audio transients
                # APPLY OFFSET BEFORE GROUNDING to help it find the right transient
                
                # Manual Offset Correction
                if hasattr(self, "audio_engine_latency_offset") and self.audio_engine_latency_offset != 0:
                     for n in notes: n.time += self.audio_engine_latency_offset
                
                # Skip grounding for Onset detected drums? 
                # Librosa Onset IS the ground truth. Grounding again might shift it to *neighboring* onset.
                # But TimingCorrector uses backtracking.
                # Let's Skip Grounding for drums if we used Onset Detection, as it IS onset detection.
                if stem != "drums":
                     notes = TimingCorrector.ground_events(notes, path, window=0.1)
                
                # Silence Gate (New)
                # Remove notes in silent sections (Hallucination removal)
                # Threshold 0.01 (~-40dB)
                notes = EventFilter.gate_silence(notes, path, threshold=0.01)
                    
                all_events.extend(notes)
                
        # 2. Transcribe Vocals
        # Prefer Lead > Mixed
        v_sources = ["vocals_lead", "vocals"]
        found_vocals = False
        for v_name in v_sources:
             # Check manifest
             if v_name in self.manifest:
                 if self.manifest[v_name].get("is_silent", False):
                     continue

             v_path = os.path.join(self.stems_folder, f"{v_name}.wav")
             if os.path.exists(v_path):
                print(f"Processing vocals ({v_name})...")
                
                # CHECK POLYPHONY MODE
                # If "choir" or "duet" in filename (heuristic) OR manifest flag
                use_polyphony = False
                if "choir" in self.base_name.lower() or "duet" in self.base_name.lower() or "poly" in self.base_name.lower():
                    use_polyphony = True
                
                v_notes = []
                
                if use_polyphony:
                    print(f"[Generator] Polyphonic Mode Enabled for {self.base_name}")
                    # 1. Get Lead (FCPE)
                    lead_events = self.council.transcribe(v_path, model_type="fcpe")
                    
                    # 2. Get Poly/Harmony (BasicPitch)
                    # Note: Defaults tuned in Council (onset=0.4, frame=0.3)
                    poly_events = self.council.transcribe(v_path, model_type="basic_pitch", multipass_consensus=True)
                    
                    # 3. Fuse
                    v_notes = ConsensusEngine.fuse_vocals(lead_events, poly_events)
                else:
                    # REVERT: Hybrid (BasicPitch) was "terrible/sparse". 
                    # Returning to FCPE (High Fidelity Frame-based)
                    # Note: We rely on "Smart Snapping" in ChartGenerator to fix the timing.
                    v_notes = self.council.transcribe(v_path, model_type="fcpe")

                print(f"DEBUG: Vocals BEFORE Smoothing: {len(v_notes)}")
                
                # Vocal Smoothing (Tunable)
                from transcribe.smoother import VocalSmoother
                # Lower level to 0.4 (Gentle) to preserve short notes (60ms)
                print(f"[Generator] Applying Vocal Smoothing (Level=0.4)...")
                v_notes = VocalSmoother.smooth(v_notes, level=0.4)
                
                print(f"DEBUG: Vocals AFTER Smoothing: {len(v_notes)}")
                
                # Patch source name (Force match to stem name for Layer Selector)
                for n in v_notes:
                    n.source = v_name 
                
                # DSP Grounding? 
                # Be careful grounding harmonies, they might shift onto lead transients.
                # But we have offset now. Let's ground them.
                all_events.extend(v_notes)
                found_vocals = True
                break
            
        # --- CACHE INJECTION ---
        cache_path = os.path.join(self.stems_folder, "events_cache.pkl")
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(all_events, f)
        print(f"[Debug] Cached {len(all_events)} events to {cache_path}")
        # -----------------------
            
        print(f"Total collected events: {len(all_events)}")
        
        # 3. Generate Charts
        print(f"[RhythmEngine] Saving beatmaps to: {self.beatmap_folder}")
        
        for diff in ["EASY", "NORMAL", "HARD", "ALT_HARD"]:
            chart_data = self.generator.generate(list(all_events), diff, manifest=self.manifest) # Pass copy & manifest
            
            out_file = os.path.join(self.beatmap_folder, f"{diff}.json")
            with open(out_file, "w") as f:
                json.dump(chart_data, f, indent=2)
            print(f"Saved {out_file}")

    def _transcribe_drums_onset(self, audio_path: str) -> List[NoteEvent]:
        import librosa
        # print(f"[Onset] Analyzing {os.path.basename(audio_path)} for transients...")
        y, sr = AudioCache.get(audio_path, sr=None)
        
        # 1. Onset Envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # 2. Pick Peaks (Adaptive threshold)
        # Increased delta (0.35 -> 0.45) for stricter picking
        # Increased wait (4 -> 6) to reduce rolls
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.45, wait=6)
        
        # 3. Convert to times
        times = librosa.frames_to_time(peaks, sr=sr)
        
        # 4. Get Energies (Velocity)
        energies = onset_env[peaks]
        if len(energies) > 0:
            max_e = energies.max()
            if max_e > 0: energies /= max_e
            
        events = []
        
        # --- GRID DECIMATOR ---
        # Goal: STRICT 1/4 note (or 1/2 note) feel.
        
        bpm = self.bpm if self.bpm > 0 else 120.0
        quarter_note_dur = 60.0 / bpm
        
        # Window: Decimate anything closer than a quarter note?
        # Target 80% of a quarter note to allow some breathing room but kill rolls.
        decim_window = quarter_note_dur * 0.85 
        
        print(f"[Onset] Decimating Drums closer than {decim_window:.3f}s (Targeting 1/4 notes at {bpm} BPM)")
        
        last_t = -1.0
        events_raw = []
        
        for t, e in zip(times, energies):
             events_raw.append(NoteEvent(
                time=float(t),
                duration=0.1, 
                pitch=60, 
                velocity=float(e),
                source="drums"
            ))
            
        # Run Decimation Pass
        final_events = []
        if events_raw:
            events_raw.sort(key=lambda x: x.time)
            
            curr = events_raw[0]
            # Initialize with first note
            
            for i in range(1, len(events_raw)):
                next_e = events_raw[i]
                
                # Logic:
                # 1. We have a 'current candidate' (curr).
                # 2. We look at 'next_e'.
                # 3. If next_e is too close to curr, we look at who is louder.
                #    If next_e is louder, it becomes the new candidate (curr = next_e).
                #    If curr is louder, we ignore next_e.
                # 4. If next_e is FAR enough, we commit 'curr' to final, and 'next_e' becomes new candidate.
                
                if next_e.time - curr.time < decim_window:
                    # Conflict! Keep louder.
                    if next_e.velocity > curr.velocity:
                        curr = next_e
                else:
                    # 'curr' is safe, push it.
                    final_events.append(curr)
                    curr = next_e
                    
            final_events.append(curr)
            
        print(f"[Onset] Found {len(events_raw)} -> Decimated {len(final_events)} drum hits.")
        return final_events
