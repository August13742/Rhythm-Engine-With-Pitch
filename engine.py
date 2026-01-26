import os
import sys
import json
import librosa
import numpy as np
import pickle
from typing import List, Dict

from beatmap import NoteEvent, EventFilter
from transcribe.basic_pitch import BasicPitchTranscriber
from transcribe.council import CouncilV2
from utils import AudioCache, TimingCorrector
from chart_generator import ChartGenerator, GENERATOR_CONFIG

# Configuration for Transcription (Per Stem)
# Tuned for high precision and musicality
# Configuration for Transcription (Per Stem)
# Moved to chart_generator.py as GENERATOR_CONFIG["transcription_filters"]

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
        
        self.manifest = self._load_manifest()

        # Estimate BPM or default
        # Check manifest first to avoid duplicated work
        if self.manifest.get("bpm"):
            self.bpm = float(self.manifest["bpm"])
            print(f"[RhythmEngine] Using BPM from manifest: {self.bpm}")
        else:
            self.bpm = self._detect_bpm() 
            print(f"[RhythmEngine] Detected BPM: {self.bpm}")
            # We'll save it to manifest during the run() phase

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
            
        # Snap to nearest 0.5 to prevent floating point drift in quantization
        bpm = round(bpm * 2) / 2
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

    def _extract_raw_events(self, rechart: bool) -> List[NoteEvent]:
        """
        Performs the Heavy Transcription Phase (Model Inference).
        Returns raw events (unfiltered, unsmoothed, ungrounded).
        Saves/Resumes from `stems/raw_events_cache.pkl`.
        """
        cache_path = os.path.join(self.stems_folder, "raw_events_cache.pkl")
        
        # Resume Check
        if rechart and os.path.exists(cache_path):
            print(f"[RhythmEngine] Loading RAW events from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load cache: {e}. Re-running extraction.")
        
        print("[RhythmEngine] Starting Extraction Phase...")
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
                    cfg = GENERATOR_CONFIG["transcription_params"].get(stem, GENERATOR_CONFIG["transcription_params"]["other"])

                    print(f"Processing {stem} with BasicPitch (Config: {cfg})...")
                    
                    params = {
                        "onset_threshold": cfg["onset"], 
                        "frame_threshold": cfg["frame"],
                        "multipass_consensus": cfg["multipass"],
                        "minimum_note_length": cfg["min_len"]
                    }
                    
                    notes = self.bp_transcriber.transcribe(path, instrument_name=stem, **params)
                    
                # Drum Fix: Force Fixed Pitch (e.g., C4 = 60)
                if stem == "drums":
                    for n in notes: n.pitch = 60
                
                all_events.extend(notes)
        
        # 2. Transcribe Vocals
        # Prefer Lead > Mixed
        v_sources = ["vocals_lead", "vocals"]
        found_vocals = False
        for v_name in v_sources:
             # Check manifest for silence
             if v_name in self.manifest:
                 if self.manifest[v_name].get("is_silent", False):
                     continue

             v_path = os.path.join(self.stems_folder, f"{v_name}.wav")
             if os.path.exists(v_path):
                print(f"Processing vocals ({v_name})...")
                
                # CHECK POLYPHONY MODE
                # Explicitly check manifest. Only enable if vocal_type is "polyphonic".
                use_polyphony = False
                if v_name in self.manifest and self.manifest[v_name].get("vocal_type", "") == "polyphonic":
                    use_polyphony = True
                
                v_notes = []
                
                if use_polyphony:
                    print(f"[Generator] Polyphonic Mode Enabled for {v_name} (Manifest Triggered)")
                    # 1. Get Lead (FCPE)
                    lead_events = self.council.transcribe(v_path, model_type="fcpe")
                    
                    # 2. Get Poly/Harmony (BasicPitch)
                    # Note: Defaults tuned in Council (onset=0.4, frame=0.3)
                    poly_events = self.council.transcribe(v_path, model_type="basic_pitch", multipass_consensus=True)
                    
                    # 3. Fuse
                    v_notes = ConsensusEngine.fuse_vocals(lead_events, poly_events)
                else:
                    # Monophonic / Default Mode
                    print(f"[Generator] Standard Monophonic Mode for {v_name}")
                    # REVERT: Hybrid (BasicPitch) was "terrible/sparse". 
                    # Returning to FCPE (High Fidelity Frame-based)
                    # Note: We rely on "Smart Snapping" in ChartGenerator to fix the timing.
                    v_notes = self.council.transcribe(v_path, model_type="fcpe")
                
                # Patch source name (Force match to stem name for Layer Selector)
                for n in v_notes:
                    n.source = v_name 
                
                all_events.extend(v_notes)
                found_vocals = True
                break
        
        # Save Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(all_events, f)
        print(f"[Debug] Cached {len(all_events)} RAW events to {cache_path}")
        
        return all_events

    def _refine_events(self, raw_events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Applies cleaning, smoothing, and grounding to raw events.
        FAST phase - runs every time (even on rechart).
        """
        print("[RhythmEngine] Refinement Phase (Latency & Grounding)...")
        refined_events = []
        
        # Group by Source
        events_by_source: Dict[str, List[NoteEvent]] = {}
        for n in raw_events:
            s = n.source
            if s not in events_by_source: events_by_source[s] = []
            events_by_source[s].append(n)
            
        for source, notes in events_by_source.items():
            path = os.path.join(self.stems_folder, f"{source}.wav")
            if not os.path.exists(path) and source == "vocals_lead":
                 path = os.path.join(self.stems_folder, "vocals.wav")
            
            # 1. Latency Compensation (Manual Offset)
            if hasattr(self, "audio_engine_latency_offset") and self.audio_engine_latency_offset != 0:
                 for n in notes: n.time += self.audio_engine_latency_offset
            
            # 2. Grounding (TimingCorrector)
            # Skip for Drums (Onset Detected)
            if source != "drums" and os.path.exists(path):
                 notes = TimingCorrector.ground_events(notes, path, window=0.1)

            # Filtering (Velocity/Silence/Smoothing) MOVED TO CHART GENERATOR (Stage 1)

            refined_events.extend(notes)
            
        print(f"Total refined events: {len(refined_events)}")
        return refined_events

    def run(self, rechart: bool = False):
        print(f"Starting Rhythm Engine V300 on: {self.stems_folder} (Focus: Dynamic)")
        if rechart:
            print("[RhythmEngine] Rechart Mode: Skipping Inference if possible.")
        
        # 0. BPM Detection (Already done in __init__)
        print(f"[RhythmEngine] Using BPM: {self.bpm}")
        
        # 1. Extract (Or Load Raw Cache)
        raw_events = self._extract_raw_events(rechart)
        
        # 2. Refine (Fast Processing)
        all_events = self._refine_events(raw_events)
        
        # 3. Generate Charts
        print(f"[RhythmEngine] Saving beatmaps to: {self.beatmap_folder}")
        
        for diff in ["EASY", "NORMAL", "HARD", "ALT_HARD"]:
            chart_data = self.generator.generate(list(all_events), diff, manifest=self.manifest, stems_folder=self.stems_folder) # Pass copy & manifest
            
            out_file = os.path.join(self.beatmap_folder, f"{diff}.json")
            with open(out_file, "w") as f:
                json.dump(chart_data, f, indent=2)
            print(f"Saved {out_file}")

        # Update Manifest with BPM if not already there
        if "bpm" not in self.manifest or self.manifest["bpm"] != self.bpm:
            self.manifest["bpm"] = self.bpm
            m_path = os.path.join(self.stems_folder, "stems_manifest.json")
            with open(m_path, "w") as f:
                json.dump(self.manifest, f, indent=2)
            print(f"[RhythmEngine] Persistent BPM {self.bpm} baked into: {m_path}")

    def _transcribe_drums_onset(self, audio_path: str) -> List[NoteEvent]:
        import librosa
        # print(f"[Onset] Analyzing {os.path.basename(audio_path)} for transients...")
        y, sr = AudioCache.get(audio_path, sr=None)
        
        # 1. Onset Envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Pick Peaks (Adaptive threshold)
        # LOOSENED for Stage 1 Charter Sifting
        # Captured more staccato/rolls, Charter will sieve based on difficulty budget.
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.20, wait=3)
        
        # 3. Convert to times
        times = librosa.frames_to_time(peaks, sr=sr)
        
        # 3.5 Rudimentary Quantization (Snap to 1/48)
        # Prevents tiny jitter from confusing the Charter's Quantizer later.
        # 1/48 grid at 120bpm is ~0.01s (very fine, keeps human feel but removes float noise)
        
        if self.bpm > 0:
            beat_dur = 60.0 / self.bpm
            snap_grid = beat_dur / 48.0 # 1/48 note
            times = [round(t / snap_grid) * snap_grid for t in times]
        else:
             # Fallback to 10ms generic snap if BPM read failed (unlikely)
             times = [round(t, 2) for t in times]
        
        # 4. Get Energies (Velocity)
        energies = onset_env[peaks]
        if len(energies) > 0:
            max_e = energies.max()
            if max_e > 0: energies /= max_e
            
        events_raw = []
        for t, e in zip(times, energies):
             events_raw.append(NoteEvent(
                time=float(t),
                duration=0.1, 
                pitch=60, 
                velocity=float(e),
                source="drums"
            ))
            
        print(f"[Onset] Found {len(events_raw)} drum hits. (Snapped to 1/48 grid)")
        return events_raw
