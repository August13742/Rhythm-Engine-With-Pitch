import os
import argparse
import sys
import json
import random
from typing import List

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beatmap import Beatmap, NoteEvent, EventFilter, Quantizer
from transcribe.basic_pitch import BasicPitchTranscriber
from transcribe.council import CouncilV2

# Configuration for Visualizer compatibility & Generator Logic
DIFF_CONFIGS = {
    "EASY":   {"lanes": 4, "nps": 2.0},
    "NORMAL": {"lanes": 4, "nps": 4.0},
    "HARD":   {"lanes": 4, "nps": 6.0},
    "INSANE": {"lanes": 4, "nps": 9.0}
}
# Removed hardcoded primary/support from DIFF_CONFIGS because it is now dynamic

GENERATOR_CONFIG = {
    "cleaning": {
        "min_duration": 0.03,
        "min_velocity": 0.1,
        "roll_consolidation_gap": 0.03
    },
    "holds": {
        "allowed_stems": ["vocals", "vocals_lead", "other"],
        "min_duration": 0.15, # seconds
        "max_vocal_duration": 1.5 # Break long vocals to prevent stale SFX pitch
    },
    "scoring": {
        "weights": {
            "vocals": 1.5, "vocals_lead": 1.6,
            "piano": 1.2, "guitar": 1.2,
            "drums": 1.1, "bass": 1.0,
            "other": 0.8
        },
        "coincidence_bonus": 0.2,
        "min_score_threshold": 0.2
    },
    "windowing": {
        "size": 1.0, 
    },
    "layers": {
        "primary_multiplier": 2.0,
        "support_multiplier": 0.8,
        "support_on_beat_bonus": 1.5,
        "shadow_window": 0.05
    }
}

class StemSelector:
    @staticmethod
    def select_layers(difficulty: str, manifest: dict, focus_mode: str = "main") -> dict:
        """
        Determines Primary/Support stems based on Difficulty and Focus Mode.
        focus_mode: "main" (Vocals/Melody) or "alt" (Instruments/Rhythm)
        """
        # 1. Analyze Manifest (What exists?)
        active_stems = []
        has_vocals = False
        
        if manifest:
            for k, v in manifest.items():
                if isinstance(v, dict) and not v.get("is_silent", True):
                    active_stems.append(k)
                    if k in ["vocals", "vocals_lead"]: has_vocals = True
        else:
             active_stems = ["vocals", "drums", "bass", "other"]
             has_vocals = True

        primary = []
        support = []
        
        # Difficulty Logic (Generalized)
        # Low Diffs always prefer Main Focus to avoid confusion
        actual_mode = focus_mode
        if difficulty in ["EASY", "NORMAL"]:
            actual_mode = "main"

        if actual_mode == "main":
            # MAIN FOCUS: Vocals > Lead > Rhythm
            if has_vocals:
                # Prefer Lead
                if "vocals_lead" in active_stems: primary.append("vocals_lead")
                elif "vocals" in active_stems: primary.append("vocals")
                
                # Hard/Insane adds Lead Guitar/Piano to primary?
                if difficulty in ["HARD", "INSANE"]:
                    if "guitar" in active_stems: primary.append("guitar")
            else:
                # Instrumental: Leads are Primary
                if "guitar" in active_stems: primary.append("guitar")
                if "piano" in active_stems: primary.append("piano")
                if not primary and "other" in active_stems: primary.append("other")

            # Support: Rhythm
            if difficulty != "EASY":
                if "drums" in active_stems: support.append("drums")
            if difficulty in ["HARD", "INSANE"]:
                if "bass" in active_stems: support.append("bass")

        elif actual_mode == "alt":
            # ALT FOCUS: Instruments (2nd Busiest) > Rhythm
            # Ignore Vocals
            primary_candidates = ["guitar", "piano", "other"]
            for s in primary_candidates:
                if s in active_stems: primary.append(s)
            
            # If no melody instruments, Drums become primary (Drum Chart)
            if not primary and "drums" in active_stems:
                primary.append("drums")
            elif "drums" in active_stems:
                # If we have melody, Drums are support? Or Primary for Insane?
                # For Alt focus, let's keep drums as Primary if it's Insane
                if difficulty == "INSANE":
                    primary.append("drums")
                else:
                    support.append("drums")
                    
            if "bass" in active_stems: support.append("bass")
            
        print(f"  [Layers] Difficulty: {difficulty} (Mode: {actual_mode})")
        print(f"    > Primary: {primary}")
        print(f"    > Support: {support}")
        
        return {"primary": primary, "support": support}

class ChartGenerator:
    def __init__(self, bpm: float = 120.0):
        self.quantizer = Quantizer(bpm)
        self.bpm = bpm
        
    def generate(self, events: List[NoteEvent], difficulty: str, manifest: dict = None, focus_mode: str = "main") -> dict:
        """
        Converts raw events into a playable chart using V300 pipeline.
        focus_mode: "main" or "alt"
        """
        print(f"[Generator] Starting {difficulty} chart generation (Mode: {focus_mode})...")
        
        cfg = DIFF_CONFIGS.get(difficulty, DIFF_CONFIGS["NORMAL"])
        target_nps = cfg["nps"]
        n_lanes = cfg["lanes"]
        
        # --- STAGE 0: LAYER SELECTION ---
        layers = StemSelector.select_layers(difficulty, manifest, focus_mode)
        primary_src = set(layers["primary"])
        support_src = set(layers["support"])
        
        # --- STAGE 1: THE CLEANER ---
        c_cfg = GENERATOR_CONFIG["cleaning"]
        clean_events = EventFilter.filter_ghost_notes(
            events, 
            min_dur=c_cfg["min_duration"], 
            min_vel=c_cfg["min_velocity"]
        )
        clean_events = EventFilter.consolidate_rolls(
            clean_events, 
            gap_threshold=c_cfg["roll_consolidation_gap"]
        )
        print(f"  [Cleaner] {len(events)} -> {len(clean_events)} events")
        
        # --- STAGE 2: ADAPTIVE GRID (Quantization) ---
        # Define Grids per Difficulty
        # 4=Quarter, 8=Eighth, 12=Triplet Eighth, 16=Sixteenth, 24=Triplet Sixteenth
        diff_grids = {
            "EASY":   [4],
            "NORMAL": [4, 8],
            "HARD":   [4, 8, 12, 16],
            "INSANE": [4, 8, 12, 16, 24, 32] 
        }
        allowed_grids = diff_grids.get(difficulty, [4, 8])
        
        # Define Stems that require Strict Grids (Backbone)
        # Even on Insane, a Bass/Drum backing track feels better if locked to standard grooves
        strict_stems = ["drums", "bass"]
        strict_grids = [4, 8]
        if difficulty in ["HARD", "INSANE"]: strict_grids.append(16) # Allow 16th kicks on Hard+
        
        # Split events
        group_strict = []
        group_free = []
        
        for e in clean_events:
            if e.source in strict_stems:
                group_strict.append(e)
            else:
                group_free.append(e)
                
        # Snap separately
        snapped_strict = self.quantizer.snap_to_grid(group_strict, grids=strict_grids)
        snapped_free = self.quantizer.snap_to_grid(group_free, grids=allowed_grids)
        
        quantized_events = snapped_strict + snapped_free
        
        # --- STAGE 3: THE SIEVE (Scoring & Selection) ---
        ranked_events = self._rank_events_layered(quantized_events, primary_src, support_src)
        final_events = self._filter_density_windowed(ranked_events, target_nps)
        print(f"  [Sieve] Selected {len(final_events)} notes (NPS Limit: {target_nps})")
        
        # --- STAGE 4: THE MAPPER (Lane Allocation) ---
        chart_notes = self._allocate_lanes(final_events, n_lanes)
        
        return {
            "metadata": {"difficulty": difficulty, "version": "V300", "bpm": self.bpm},
            "notes": chart_notes
        }

    def _rank_events_layered(self, events: List[NoteEvent], primary_src: set, support_src: set) -> List[NoteEvent]:
        """
        Rank events based on their Layer Assignment.
        Strict Mode: Non-Focus notes get 0 score.
        Support Mode: Must be on-beat.
        """
        if not events: return []
        
        scored_Events = []
        l_cfg = GENERATOR_CONFIG["layers"]
        s_cfg = GENERATOR_CONFIG["scoring"]
        
        beat_dur = 60.0 / self.bpm if self.bpm > 0 else 0.5
        
        for n in events:
            # Base Score = Velocity
            base_score = n.velocity
            is_valid = False
            
            # Layer Logic
            if n.source in primary_src:
                base_score *= l_cfg["primary_multiplier"]
                is_valid = True
                
            elif n.source in support_src:
                # Support: GRID LIMIT CHECK
                # Only allow support notes on main beats (1/4, 1/8)
                # We can check quantization grid or time
                # Ideally, we want "Tempo Setters".
                
                time_in_beats = n.time / beat_dur
                # Check for 1/2 beat (Eighth note) precision
                # E.g., 0.0, 0.5, 1.0, 1.5...
                
                # Allow 1/4 notes (Strongest) and 1/8 (Strong)
                # Reject 1/16 fills for Support
                
                beat_fraction = time_in_beats % 1.0
                # Close to 0.0 (Quarter) or 0.5 (Eighth)
                is_quarter = abs(beat_fraction) < 0.1 or abs(beat_fraction - 1.0) < 0.1
                is_eighth = abs(beat_fraction - 0.5) < 0.1
                
                if is_quarter:
                     base_score *= l_cfg["support_multiplier"] * l_cfg["support_on_beat_bonus"]
                     is_valid = True
                elif is_eighth:
                     base_score *= l_cfg["support_multiplier"]
                     is_valid = True
                else:
                     # Filter out complex fills (16ths) for Support layer
                     base_score = 0.0
                     is_valid = False
            else:
                # KILL NON-FOCUS
                base_score = 0.0
                is_valid = False
                
            if is_valid:
                setattr(n, "_score", base_score) 
                scored_Events.append(n)
            
        # Coincidence Bonus (Shadowing)
        # If a Support note coincides with a Primary note, suppress the Support note?
        scored_Events.sort(key=lambda x: x.time)
        coincidence_window = l_cfg["shadow_window"]
        
        # Since we filtered `scored_Events` to only include valid notes, we can loop through them
        # However, we need to be careful: if we removed a note, it can't shadow anything.
        # But Primary notes are always valid. Support notes might be removed if non-grid.
        
        for i, n in enumerate(scored_Events):
            # Check neighbors
            for j in range(max(0, i-5), min(len(scored_Events), i+5)):
                if i == j: continue
                other = scored_Events[j]
                if abs(n.time - other.time) < coincidence_window:
                    # Logic: If I am Support and Other is Primary -> I get nuked
                    if n.source in support_src and other.source in primary_src:
                        n._score = 0.0 # Shadowed by primary
                    
                    # Logic: If both are same layer -> Coincidence Bonus (Chord)
                    elif (n.source in primary_src and other.source in primary_src):
                        n._score += s_cfg["coincidence_bonus"]

        # Final Filter: remove zero score notes
        return [e for e in scored_Events if getattr(e, "_score", 0) > 0.001]

    def _filter_density_windowed(self, events: List[NoteEvent], target_nps: float) -> List[NoteEvent]:
        """
        Selects notes using a Sliding Window approach.
        NPS is treated as a LIMIT per window, not a target to fill.
        Also enforces a minimum quality threshold to avoid garbage.
        """
        if not events: return []
        
        # 1. Global Quality Filter
        min_score = GENERATOR_CONFIG["scoring"]["min_score_threshold"]
        valid_events = [e for e in events if getattr(e, "_score", 0) >= min_score]
        
        if not valid_events: return []
        
        valid_events.sort(key=lambda x: x.time)
        
        final_selection = []
        window_size = GENERATOR_CONFIG["windowing"]["size"]
        max_notes_per_window = int(target_nps * window_size)
        
        # 2. Windowed Sieve
        # Iterate through windows
        start_time = valid_events[0].time
        end_time = valid_events[-1].time
        
        curr_t = start_time
        event_idx = 0
        n_events = len(valid_events)
        
        while curr_t < end_time:
            next_t = curr_t + window_size
            
            # Collect notes in this window
            window_notes = []
            temp_idx = event_idx
            while temp_idx < n_events and valid_events[temp_idx].time < next_t:
                window_notes.append(valid_events[temp_idx])
                temp_idx += 1
            
            # Process window
            if len(window_notes) > max_notes_per_window:
                # Sieve inside the window by Score
                window_notes.sort(key=lambda x: x._score, reverse=True)
                selected = window_notes[:max_notes_per_window]
                final_selection.extend(selected)
            else:
                final_selection.extend(window_notes)
                
            # Move index
            event_idx = temp_idx
            curr_t = next_t
            
        final_selection.sort(key=lambda x: x.time)
        return final_selection

    def _allocate_lanes(self, events: List[NoteEvent], n_lanes: int) -> List[dict]:
        """
        Maps notes to lanes [0, n_lanes-1].
        """
        processed = []
        
        # State for anchoring
        last_pitch = events[0].pitch if events else 60
        last_lane = n_lanes // 2
        
        # Configuration
        allowed_holds = GENERATOR_CONFIG["holds"]["allowed_stems"]
        min_hold_dur = GENERATOR_CONFIG["holds"]["min_duration"]
        
        # Calculate dynamic hold threshold based on BPM (still useful as a relative minimum)
        beat_dur = 60.0 / self.bpm
        # Dynamic thresh should not be lower than absolute min_hold_dur
        dynamic_thresh = (beat_dur / 2.0) * 0.9 
        hold_thresh = max(min_hold_dur, dynamic_thresh)
        
        for ev in events:
            # Logic: Relative movements
            pitch_delta = ev.pitch - last_pitch
            
            # If large jump, absolute mapping
            if abs(pitch_delta) > 12: # Octave jump
                # Map 40-80 to 0-(n-1)
                norm = max(0.0, min(1.0, (ev.pitch - 48) / 36.0))
                lane = int(norm * (n_lanes - 0.01))
            else:
                # Relative move
                move = 0
                if pitch_delta > 2: move = 1
                elif pitch_delta < -2: move = -1
                
                lane = max(0, min(n_lanes - 1, last_lane + move))
            
            # Update state
            last_pitch = ev.pitch
            last_lane = lane
            
            # Check hold type
            # 1. Must be allowed stem
            # 2. Must exceed threshold
            is_hold = False
            if ev.source in allowed_holds:
                if ev.duration > hold_thresh:
                    is_hold = True
            
            processed.append({
                "time": float(ev.time),
                "lane": int(lane),
                "dur": float(ev.duration) if is_hold else 0.0, # Taps = 0.0 dur
                "type": "hold" if is_hold else "tap",
                "midi": int(ev.pitch),
                "vol": float(getattr(ev, "velocity", 0.8)), # Default vol if missing
                "source": str(ev.source)
            })
        
        # Post-process: Resolve conflicts (Hold overlaps)
        return self._resolve_conflicts(processed)

    def _resolve_conflicts(self, notes: List[dict]) -> List[dict]:
        """
        Prevents overlapping notes in the same lane.
        Trims holds to ensure a gap before the next note.
        Also merges notes that are too close (Minijacks) to avoid tap bursts.
        """
        # Group by lane
        lanes_dict = {}
        for n in notes:
            l = n["lane"]
            if l not in lanes_dict: lanes_dict[l] = []
            lanes_dict[l].append(n)
            
        final = []
        gap = 0.05 # 50ms gap
        minijack_thresh = 0.1 # 100ms threshold for merging close notes (approx 1/16 at 150BPM)
        allowed_holds = GENERATOR_CONFIG["holds"]["allowed_stems"]

        for l in lanes_dict:
            # Sort by time
            stack = sorted(lanes_dict[l], key=lambda x: x["time"])
            
            # Merged stack
            merged_stack = []
            if not stack: continue
            
            # Pass 1: Merge close notes
            curr = stack[0]
            for i in range(1, len(stack)):
                next_n = stack[i]
                
                # Check proximity
                if next_n["time"] - curr["time"] < minijack_thresh:
                    # Merge!
                    # Only create hold if allowed
                    can_hold = curr["source"] in allowed_holds
                    
                    if can_hold:
                        # Extend current duration to cover next note
                        new_end = max(curr["time"] + curr["dur"], next_n["time"] + next_n["dur"])
                        curr["dur"] = max(0.1, new_end - curr["time"]) # Make it a hold if merged
                        curr["type"] = "hold"
                    else:
                        # Cannot hold (e.g. guitar/drums). 
                        # Absorb the next note but keep as tap (dur=0).
                        # Essentially "de-jacking" the chart.
                        curr["dur"] = 0.0
                        curr["type"] = "tap"
                        
                    # Skip next_n (it's absorbed)
                else:
                    merged_stack.append(curr)
                    curr = next_n
            merged_stack.append(curr)
            
            # Pass 2: Overlap Prevention (Hold Escape)
            for i in range(len(merged_stack)):
                current = merged_stack[i]
                
                # Check against next note
                if i + 1 < len(merged_stack):
                    next_note = merged_stack[i+1]
                    
                    if current["type"] == "hold":
                        limit = next_note["time"] - gap
                        end_t = current["time"] + current["dur"]
                        
                        if end_t > limit:
                            # Trim duration
                            new_dur = max(0.0, limit - current["time"])
                            current["dur"] = new_dur
                            
                            # If too short, convert back to tap
                            if new_dur < 0.05:
                                current["dur"] = 0.0
                                current["type"] = "tap"
                
                final.append(current)
        
        # Re-sort all by time
        return sorted(final, key=lambda x: x["time"])

class TimingCorrector:
    @staticmethod
    def ground_events(events: List[NoteEvent], audio_path: str, window: float = 0.05) -> List[NoteEvent]:
        """
        Uses DSP (librosa onset detection) to snap event times to the nearest true audio onset.
        This corrects latency/jitter from the ML transcription model.
        """
        try:
            import librosa
            import numpy as np
        except ImportError:
            return events
            
        if not os.path.exists(audio_path) or not events:
            return events
            
        print(f"[Timing] Grounding {len(events)} events with DSP ({os.path.basename(audio_path)})...")
        
        try:
            # Load audio (lightweight load)
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Detect Onsets
            # backtracking=True helps find the precise start of the transient
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            if len(onset_times) == 0:
                return events
                
            # Snap events
            snapped_count = 0
            for e in events:
                # Find nearest onset
                # Search sorted array efficiently? Or just simple search for now (n*m) is slow if large.
                # Use numpy for speed if possible, but events is list of objects.
                # valid range: [e.time - window, e.time + window]
                
                # Simple linear scan optimized by knowing onsets are sorted?
                # Let's use numpy searchsorted
                idx = np.searchsorted(onset_times, e.time)
                
                candidates = []
                if idx < len(onset_times): candidates.append(onset_times[idx])
                if idx > 0: candidates.append(onset_times[idx - 1])
                
                best_onset = -1
                min_dist = window
                
                for t in candidates:
                    dist = abs(t - e.time)
                    if dist < min_dist:
                        min_dist = dist
                        best_onset = t
                        
                if best_onset != -1:
                    # Apply correction
                    e.time = float(best_onset)
                    snapped_count += 1
                    
            print(f"[Timing] Snapped {snapped_count}/{len(events)} events to DSP onsets.")
            
        except Exception as e:
            print(f"[Timing] DSP Grounding failed: {e}")
            
        return events

class RhythmEngine:
    def __init__(self, stems_folder: str):
        self.stems_folder = stems_folder
        self._ensure_paths()
        
        # Initialize Transcribers
        self.bp_transcriber = BasicPitchTranscriber()
        self.council = CouncilV2()
        
        # Estimate BPM or default
        self.bpm = 120.0 
        self.generator = ChartGenerator(bpm=self.bpm)
        self.manifest = self._load_manifest()

    def _ensure_paths(self):
        if not os.path.isdir(self.stems_folder):
            raise ValueError(f"Stems folder does not exist: {self.stems_folder}")

    def _load_manifest(self):
        m_path = os.path.join(self.stems_folder, "stems_manifest.json")
        if os.path.exists(m_path):
            try:
                with open(m_path, 'r') as f:
                    return json.load(f)
            except: 
                return {}
        return {}

    def run(self, focus_mode: str = "main"):
        print(f"Starting Rhythm Engine V300 on: {self.stems_folder} (Focus: {focus_mode})")
        
        # 0. BPM Detection (Basic)
        self._detect_bpm()
        self.generator.bpm = self.bpm 
        self.generator.quantizer.bpm = self.bpm
        
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
                
                # Tune parameters based on instrument
                # For melody instruments, we want high precision (fewer false positives)
                # For drums, we force fixed pitch anyway, so standard params are fine
                params = {}
                if stem in ["piano", "guitar"]:
                     params = {"onset_threshold": 0.6, "frame_threshold": 0.4}
                
                notes = self.bp_transcriber.transcribe(path, instrument_name=stem, **params)
                
                # Drum Fix: Force Fixed Pitch (e.g., C4 = 60)
                if stem == "drums":
                    for n in notes: n.pitch = 60
                
                # DSP Grounding (New)
                # Ground instrument notes to audio transients
                notes = TimingCorrector.ground_events(notes, path)
                    
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
                v_notes = self.council.transcribe(v_path, model_type="fcpe")
                # Patch source name
                for n in v_notes: n.source = v_name
                
                # DSP Grounding for Vocals? 
                # Vocals are softer, onsets might be unreliable.
                # But let's try it with a relaxed window? 
                # Or skip it. Let's skip for vocals for now to preserve flow.
                
                all_events.extend(v_notes)
                found_vocals = True
                break
            
        print(f"Total collected events: {len(all_events)}")
        
        # 3. Generate Charts
        output_dir = os.path.join(self.stems_folder, "beatmap")
        os.makedirs(output_dir, exist_ok=True)
        
        for diff in ["EASY", "NORMAL", "HARD", "INSANE"]:
            chart_data = self.generator.generate(list(all_events), diff, manifest=self.manifest, focus_mode=focus_mode) # Pass copy & manifest
            
            out_file = os.path.join(output_dir, f"{diff}.json")
            with open(out_file, "w") as f:
                json.dump(chart_data, f, indent=2)
            print(f"Saved {out_file}")

    def _detect_bpm(self):
        try:
            import librosa
            import numpy as np
            
            # Priority: Drums -> Other -> Vocals -> First available stem
            candidates = ["drums", "other", "vocals", "bass", "piano", "guitar"]
            
            bpm_found = 0.0
            
            for stem in candidates:
                path = os.path.join(self.stems_folder, f"{stem}.wav")
                if os.path.exists(path):
                    # Check if file has meaningful content (size > 10kb)
                    # This prevents loading silent/header-only wavs
                    if os.path.getsize(path) < 10000:
                        continue
                        
                    print(f"Detecting BPM from {stem}...")
                    try:
                        y, sr = librosa.load(path, sr=22050, duration=60)
                        if len(y) < sr * 5: # Skip if shorter than 5 seconds
                            continue
                            
                        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                        
                        if isinstance(tempo, np.ndarray):
                            tempo = tempo[0]
                        
                        val = float(tempo)
                        if val > 40 and val < 300: # Reasonable range
                            bpm_found = val
                            print(f"Detected BPM: {self.bpm:.2f}")
                            break
                    except Exception as sub_e:
                        print(f"Failed to detect BPM from {stem}: {sub_e}")
                        continue
            
            if bpm_found > 0:
                self.bpm = bpm_found
            else:
                print("No suitable audio for BPM detection, using default 120.")
                
        except Exception as e:
            print(f"BPM Detection failed: {e}. Using default 120.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rhythm Engine V300")
    parser.add_argument("folder", help="Path to the folder containing separated stems")
    args = parser.parse_args()

    engine = RhythmEngine(args.folder)
    engine.run()
