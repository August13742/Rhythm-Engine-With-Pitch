import copy
from typing import List
from beatmap import NoteEvent, EventFilter, Quantizer
from transcribe.smoother import VocalSmoother
import numpy as np
import os

# Configuration for Visualizer compatibility & Generator Logic
DIFF_CONFIGS = {
    "EASY":   {"lanes": 4, "nps": 2.5, "poly": 1, "min_interval": 0.25},
    "NORMAL": {"lanes": 4, "nps": 5.0, "poly": 2, "min_interval": 0.20},
    "HARD":   {"lanes": 4, "nps": 8.0, "poly": 2, "min_interval": 0.15},
    "ALT_HARD": {"lanes": 4, "nps": 8.0, "poly": 2, "min_interval": 0.15}
}

GENERATOR_CONFIG = {
    "cleaning": {
        "min_duration": 0.03,
        "min_velocity": 0.15,
        "roll_consolidation_gap": 0.03
    },
    "holds": {
        "allowed_stems": ["vocals", "vocals_lead", "other"],
        "min_duration": 0.35,
        "max_vocal_duration": 10.0,
        "min_reaction_time": 0.10 
    },
    "sieving": {
        "burst_limit": 0.4, 
        "min_global_interval": 0.05, 
        "support_grid": 4  # Strict denominator for Support Grid selection (e.g. 8 = 1/8 notes)
    },
    "strict_grids": {
        "enforce_pick_from_grid": False,
        "main_layer_grid": [24,32],
        "support_layer_grid": [4]
    },
    "scoring": {
        "weights": {
            "vocals": 1.5, "vocals_lead": 1.6,
            "piano": 1.2, "guitar": 1.2,
            "drums": 1.1, "bass": 1.0,
            "other": 0.6  # Lower weight for noisy composite track
        },
        "rhythm_weights": {
            "quarter": 1.30,   # Strong bias for downbeats (Red)
            "eighth": 1.15,    # Moderate bias for upbeats (Blue)
            "sixteenth": 1.0,  # Neutral (Yellow)
            "complex": 0.9    # Slight penalty for off-grid/triplets (Purple)
        },
        "coincidence_bonus": 0.2,
        "min_score_threshold": 0.25
    },
    "windowing": {
        "size": 1.0, 
    },
    "layers": {
        "primary_multiplier": 2.0,
        "support_multiplier": 0.8,
        "support_on_beat_bonus": 1.5,
        "shadow_window": 0.05
    },
    "volume": {
        "primary_boost": 1.2,       # +20% for Primary Layer
        "support_penalty": 1.0,     # -10% for Support Layer
        "instrument_boosts": {
            "drums": 1.1,          # Drums need punch
            "bass": 1.10,           # Bass needs presence
            "vocals": 1.05,         # Vocals are key
            "vocals_lead": 1.05,
            "guitar": 1.0,          # Standard
            "piano": 1.0,
            "other": 0.9            # Background stuff slightly quieter
        },
        "global_limit": 1.0         # Hard clip at 1.0
    },
    "layer_sifting": {
        "melodic_support_gate": 0.35,   # Min velocity for Melodic Support notes
        "melodic_support_weight": 0.5,  # Score penalty (Deprioritize in density)
        "percussive_stems": ["drums", "bass"], # Stems EXEMPT from sifting
        "smoothing_window": 3.0 # if x second silent: enable alternative stem to takeover
    },
    "transcription_filters": {
        "vocals": {
            "source_match": ["vocals", "vocals_lead"],
            "velocity_gate": 0.0, # Handled by Council/FCPE usually
            "smoothing": 0.7,
            "silence_threshold": 0.01 
        },
        "piano": {
            "source_match": ["piano"],
            "velocity_gate": 0.0,
            "smoothing": 0.0,
            "silence_threshold": 0.01
        },
        "guitar": {
            "source_match": ["guitar"],
            "velocity_gate": 0.0,
            "smoothing": 0.0,
            "silence_threshold": 0.01
        },
        "bass": {
            "source_match": ["bass"],
            "velocity_gate": 0.0,
            "smoothing": 0.5,
            "silence_threshold": 0.01
        },
        "other": {
            "source_match": ["other"],
            "velocity_gate": 0.25, # Remove low-confidence notes
            "smoothing": 0.0,
            "silence_threshold": 0.01
        },
        "drums": {
            "source_match": ["drums"],
            "velocity_gate": 0.0, # Onset detection handles this?
            "smoothing": 0.0,
            "silence_threshold": 0.00 # Drums often sharp
        }
    },
    "transcription_params": {
        "vocals": {"onset": 0.35, "frame": 0.30, "multipass": False, "min_len": 58.0},
        "piano": {"onset": 0.40, "frame": 0.30, "multipass": True, "min_len": 30.0},
        "guitar": {"onset": 0.40, "frame": 0.30, "multipass": True, "min_len": 40.0},
        "bass": {"onset": 0.50, "frame": 0.40, "multipass": True, "min_len": 80.0},
        "other": {"onset": 0.55, "frame": 0.40, "multipass": True, "min_len": 58.0}
    }
}

class ActivityAnalyzer:
    """Calculates stem activity density over time to drive dynamic selection."""
    @staticmethod
    def get_activity_map(events: List[NoteEvent], window_size: float = 5.0, resolution: float = 1.0) -> dict:
        """
        Creates a map of {stem: activity_curve} where activity_curve is a list of scores.
        resolution: Step size in seconds for the activity timeline.
        window_size: Period in seconds for the smoothing window.
        """
        if not events: return {}
        
        max_time = max(e.time + e.duration for e in events)
        num_steps = int(max_time / resolution) + 2
        
        # 1. Initialize buckets
        activity = {} # stem -> [sum_velocity]
        stems = set(e.source for e in events)
        for s in stems:
            activity[s] = [0.0] * num_steps
            
        # 2. Fill atomic buckets
        for e in events:
            step = int(e.time / resolution)
            if step < num_steps:
                # Use velocity * duration (energy) as activity metric
                # For drums (dur=0.1), velocity is usually enough.
                energy = e.velocity * (e.duration if e.duration > 0.1 else 0.5)
                activity[e.source][step] += energy
                
        # 3. Apply Sliding Window Smoothing (Forward & Backward)
        # This provides the "Long Context" hysteresis.
        smoothed = {}
        half_win = int(window_size / (2 * resolution))
        
        for s, curve in activity.items():
            s_curve = [0.0] * num_steps
            for i in range(num_steps):
                start = max(0, i - half_win)
                end = min(num_steps, i + half_win + 1)
                window = curve[start:end]
                s_curve[i] = sum(window) / len(window) if window else 0.0
            smoothed[s] = s_curve
            
        return {
            "resolution": resolution,
            "max_time": max_time,
            "activity": smoothed
        }

class LayerTimeline:
    """Queryable object for dynamic primary/support layers."""
    def __init__(self, timeline: List[dict], resolution: float):
        self.timeline = timeline # list of {"primary": set, "support": set}
        self.resolution = resolution
        
    def get_layers(self, time: float) -> tuple:
        idx = int(time / self.resolution)
        if idx < 0: idx = 0
        if idx >= len(self.timeline):
            idx = len(self.timeline) - 1
            
        return self.timeline[idx]["primary"], self.timeline[idx]["support"]

class StemSelector:
    PRIORITY_HIERARCHY = [
        ["vocals_lead", "vocals"],
        ["guitar", "piano", "synth", "other"],
        ["bass"],
        ["drums"]
    ]

    @staticmethod
    def get_dynamic_timeline(events: List[NoteEvent], difficulty: str, focus_mode: str = "main") -> LayerTimeline:
        """Determines layers for every second of the song based on activity and hierarchy."""
        resolution = 1.0
        
        # Pull smoothing window from config
        smoothing_window = GENERATOR_CONFIG["layer_sifting"].get("smoothing_window", 5.0)
        
        analyzer = ActivityAnalyzer()
        act_map = analyzer.get_activity_map(events, window_size=smoothing_window, resolution=resolution)
        
        activity = act_map["activity"]
        num_steps = len(next(iter(activity.values()))) if activity else 0
        
        timeline_data = []
        
        # Fixed Mode Override
        actual_mode = focus_mode
        if difficulty in ["EASY", "NORMAL", "HARD"]:
            actual_mode = "main"
        elif difficulty == "ALT_HARD":
            actual_mode = "alt"
            
        # Sticky Selection State / Thresholds
        activity_threshold = 0.1 # Minimum activity to be considered "active"

        # 1. Identify "Ideal" Primary (Focus Stem)
        focus_stem = None
        if actual_mode == "main":
            # Use Vocals if they exist anywhere
            vocal_stems = ["vocals_lead", "vocals"]
            for s in vocal_stems:
                if s in activity:
                    focus_stem = s
                    break
        else: # alt
            # Top melodic non-vocal
            melodic = ["guitar", "piano", "synth", "other"]
            candidates = [s for s in melodic if s in activity]
            if candidates:
                # Pick busiest overall melodic
                candidates.sort(key=lambda s: sum(activity[s]), reverse=True)
                
                # Instrumental Divergence:
                # If song has no vocals, HARD defaults to Candidate[0] (Busiest).
                # To force ALT to be different, we pick Candidate[1] (2nd Busiest) if available.
                has_vocals = any(v in activity for v in ["vocals", "vocals_lead"])
                
                if not has_vocals and len(candidates) > 1:
                    print(f"    [StemSelector] Instrumental detected (ALT Mode): Picking 2nd best stem ({candidates[1]}) to diverge from HARD.")
                    focus_stem = candidates[1]
                else:
                    focus_stem = candidates[0]

        # Use Vocals as fallback if no melodic in Alt mode? 
        # Actually user said "vocal is playing when main instrument stops"
        
        # 2. Analyze Gaps in Focus Stem (Raw Activity vs Smoothed)
        # We use RAW activity to find the start/end of true silence gaps.
        # But we use SMOOTHED activity for fallback selection to avoid flicker.
        raw_map = ActivityAnalyzer.get_activity_map(events, window_size=0, resolution=resolution)
        raw_activity = raw_map["activity"]
        
        min_gap = GENERATOR_CONFIG["layer_sifting"].get("min_switch_gap", 5.0)
        gap_steps = int(min_gap / resolution)
        
        is_focus_silent_raw = [True] * num_steps
        if focus_stem and focus_stem in raw_activity:
            is_focus_silent_raw = [raw_activity[focus_stem][i] <= 0.05 for i in range(num_steps)]
            
        # Identify 5s+ Gaps
        fallback_allowed = [False] * num_steps
        i = 0
        while i < num_steps:
            if is_focus_silent_raw[i]:
                start = i
                while i < num_steps and is_focus_silent_raw[i]:
                    i += 1
                end = i
                if (end - start) >= gap_steps:
                    # Sustained Gap! Allow fallback in this entire range.
                    for j in range(start, end):
                        fallback_allowed[j] = True
            else:
                i += 1
                
        # 3. Fill Gaps with Best Candidate (Window-Aggregated)
        # For each gap, find the stem with the highest TOTAL activity within that gap.
        final_primary_timeline = [None] * num_steps
        
        # Default: Fill with focus stem
        for i in range(num_steps):
            final_primary_timeline[i] = focus_stem
            
        # Overwrite Gaps
        # We need to reconstruct the gaps from fallback_allowed
        i = 0
        while i < num_steps:
            if fallback_allowed[i]:
                start = i
                while i < num_steps and fallback_allowed[i]:
                    i += 1
                end = i
                
                # Analyze this gap [start, end]
                best_stem = None
                best_score = -1.0
                
                # Candidates: All melodic/harmonic stems + Vocals (for Alt mode fallback)
                candidates = ["vocals", "vocals_lead", "guitar", "piano", "synth", "other", "bass"] 
                
                # Calculate Global Activity for Bias
                global_activity_sum = {s: sum(curve) for s, curve in activity.items()}
                max_global = max(global_activity_sum.values()) if global_activity_sum else 1.0
                
                scores = {}
                for s in candidates:
                    if s in activity:
                        # Sum activity in this window
                        window_sum = sum(activity[s][start:end])
                        
                        # Apply Global Dominance Bias
                        # Bonus: Up to +100% score for being the most dominant stem in the song
                        global_score = global_activity_sum.get(s, 0.0)
                        dominance_factor = global_score / max_global if max_global > 0 else 0.0
                        bias_multiplier = 1.0 + (1.0 * dominance_factor)
                        
                        final_score = window_sum * bias_multiplier
                        
                        scores[s] = final_score
                        if final_score > best_score and window_sum > 0.1:
                            best_score = final_score
                            best_stem = s
                            
                # Summary of scores for this window
                if best_stem:
                    score_strs = [f"{s}:{scr:.2f}" for s, scr in scores.items() if scr > 0]
                    print(f"    [GapFiller] Range {start*resolution:.1f}-{end*resolution:.1f}s | Winner: {best_stem} | Scores: {', '.join(score_strs)}")
                
                # If we found a good fallback, use it for the WHOLE gap
                # This ensures stability (Sticky for the window duration)
                if best_stem:
                    for k in range(start, end):
                        final_primary_timeline[k] = best_stem
            else:
                i += 1

        # 4. Build Final Timeline Objects
        for i in range(num_steps):
            primary = []
            support = []
            
            p_stem = final_primary_timeline[i]
            if p_stem:  
                # Check if it's active at this specific step?
                # User request: "active instrument of the specific absent window"
                # We selected based on sum, but maybe it has a quiet moment inside the window.
                # Should we silence it? No, "fill it in".
                primary.append(p_stem)
                
            # Support
            if difficulty != "EASY":
                active_at_step = [s for s, curve in activity.items() if curve[i] > activity_threshold]
                drums_active = "drums" in active_at_step
                
                # Logic: Blacklist support if it's primary
                # If primary is drums (unlikely), don't add drums.
                if "drums" not in primary and drums_active:
                    support.append("drums")
                else:
                    # Fallback support
                    if "bass" in active_at_step and "bass" not in primary:
                        support.append("bass")
                            
            timeline_data.append({"primary": set(primary), "support": set(support)})
        
        # Summary logging
        if timeline_data:
            switches = []
            last_p = None
            for idx, entry in enumerate(timeline_data):
                p = list(entry["primary"])[0] if entry["primary"] else "None"
                if p != last_p:
                    switches.append(f"{idx*resolution}s:{p}")
                    last_p = p
            print(f"    [Layers] Dynamic Timeline ({difficulty}): {' -> '.join(switches[:10])}{'...' if len(switches)>10 else ''}")
            
        return LayerTimeline(timeline_data, resolution)

    @staticmethod
    def get_stem_energy(stem_name: str, manifest: dict) -> float:
        """Returns the peak energy or RMS from manifest."""
        if not manifest or stem_name not in manifest:
            return 0.0
        
        info = manifest[stem_name]
        return info.get("peak_energy", 0.0)



class VocalCorrector:
    """
    Sieve to stop vocal charts from jumping wildly (Octave Errors).
    Runs BEFORE Lane Allocation.
    """
    @staticmethod
    def apply(events: List[NoteEvent], difficulty: str) -> List[NoteEvent]:
        # Only apply to vocals
        vocals = [e for e in events if e.source in ["vocals", "vocals_lead"]]
        others = [e for e in events if e.source not in ["vocals", "vocals_lead"]]
        
        if not vocals: return events
        
        vocals.sort(key=lambda x: x.time)
        
        # Logic: Rolling Median Sieve
        # If a note deviates by exactly +/- 12 semitones from median, and is short, snap it back.
        
        window_size = 5
        half_window = window_size // 2
        
        cleaned_vocals = []
        # We process linearly but need context.
        # Actually, let's just do a pass.
        
        for i in range(len(vocals)):
            curr = vocals[i]
            
            # Context window
            start = max(0, i - half_window)
            end = min(len(vocals), i + half_window + 1)
            context = vocals[start:end]
            
            if not context:
                cleaned_vocals.append(curr)
                continue
                
            # Calculate Median Pitch
            pitches = sorted([n.pitch for n in context])
            median = pitches[len(pitches)//2]
            
            diff = curr.pitch - median
            
            # Check for Octave Error (Approx 12 +/- 1)
            if 11 <= abs(diff) <= 13:
                # It's an octave jump. 
                # Is it real? 
                # If it's a short note (likely a glitch or fry), correct it.
                # If it's a long belted note, maybe keep it? 
                # For now, aggressive correction for stability.
                if curr.duration < 0.3: # Short note
                    # Shift back
                    shift = -12 if diff > 0 else 12
                    
                    # Log it?
                    print(f"  [VocalCorrector] Fixed Octave Jump at {curr.time:.2f}s ({curr.pitch} -> {curr.pitch + shift})")
                    
                    curr.pitch += shift
            
            cleaned_vocals.append(curr)
            
        return others + cleaned_vocals

class ChartGenerator:
    def __init__(self, bpm: float = 120.0):
        self.quantizer = Quantizer(bpm)
        self.bpm = bpm
        
    def generate(self, events: List[NoteEvent], difficulty: str, manifest: dict = None, focus_mode: str = "main", stems_folder: str = None, override_lanes: int = None, chart_profile: str = "STANDARD") -> dict:
        """
        Converts raw events into a playable chart using V300 pipeline.
        focus_mode: "main" or "alt"
        chart_profile: "STANDARD", "DRAFT", or "RAW"
        """
        # Deepcopy events to prevent side effects (mutation) from affecting other difficulties
        events = copy.deepcopy(events)
        
        print(f"[Generator] Starting {difficulty} chart generation (Mode: {focus_mode}, Profile: {chart_profile})...")
        
        cfg = DIFF_CONFIGS.get(difficulty, DIFF_CONFIGS["NORMAL"])
        target_nps = cfg["nps"]
        n_lanes = cfg["lanes"]
        
        if override_lanes and override_lanes > 0:
            n_lanes = override_lanes
            print(f"  [Generator] Overriding lanes to: {n_lanes}")
        
        # --- STAGE 0: LAYER SELECTION (Dynamic) ---
        # Generate a dynamic timeline of which stems are Primary vs Support
        timeline = StemSelector.get_dynamic_timeline(events, difficulty, focus_mode)
        
        # --- STAGE 1: THE CLEANER (Gameplay Filtering) ---
        # RAW Mode bypasses standard filtering (Use minimal filtering)
        if chart_profile != "RAW":
            events = self._apply_gameplay_filtering(events, stems_folder)
        
        c_cfg = GENERATOR_CONFIG["cleaning"]
        
        if chart_profile != "RAW":
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
        else:
            print("  [Cleaner] RAW Mode: Bypassing Filters.")
            clean_events = events
        
        # --- STAGE 2: ADAPTIVE GRID (High Fidelity Snapping - Artifact Correction) ---
        
        # Strict (Drums/Bass/Backing)
        strict_grids = [4, 8, 16] # Standardized for rhythmic stems
        
        # Soft (Vocals/Lead) - High Fidelity Snapping
        # 1/12, 1/16, 1/24, 1/32, 1/48
        soft_grids = [4, 8, 12, 16, 24, 32, 48]
        
        # Split events
        group_strict = []
        group_soft = []
        
        strict_stems = ["drums"]
        
        for e in clean_events:
            if e.source in strict_stems:
                group_strict.append(e)
            else:
                group_soft.append(e)
                
        # Snap separately
        snapped_strict = self.quantizer.snap_to_grid(group_strict, grids=strict_grids)
        snapped_soft = self.quantizer.snap_to_grid(group_soft, grids=soft_grids)
        
        quantized_events = snapped_strict + snapped_soft
        
        # --- RAW EXIT ---
        if chart_profile == "RAW":
            # For RAW, just return all quantized events mapped to lanes
            print("  [Generator] RAW Profile: Returning early.")
            # Skip sieving/ranking. Direct mapping.
            sanitized_events = quantized_events
            sanitized_events.sort(key=lambda x: x.time)
            chart_notes = self._allocate_lanes(sanitized_events, n_lanes)
            return {
                "metadata": {
                    "difficulty": difficulty, "bpm": self.bpm,
                    "focus": focus_mode, "lanes": n_lanes, "profile": chart_profile
                },
                "notes": chart_notes
            }
        
        # --- STAGE 2.5: SCRIPTING CLEANUP (Merge & Dedupe) ---
        events_to_sieve = quantized_events
        
        if chart_profile == "STANDARD":
            corrected_events = VocalCorrector.apply(quantized_events, difficulty)
            merged_events = self._merge_fragmented_notes(corrected_events)
            events_to_sieve = merged_events
            
            merge_count = len(quantized_events) - len(events_to_sieve)
            if merge_count > 0:
                print(f"  [Cleanup] Merged {merge_count} notes before selection")
        
        # --- STAGE 3: THE SIEVE (Selection & Musicality) ---
        ranked_events = self._rank_events_layered(events_to_sieve, timeline)
        
        if chart_profile == "DRAFT":
            # DRAFT Mode passes everything that survives Ranking (Quality check)
            # but Bypasses the Budget Sieve.
            final_events = ranked_events
            print(f"  [Sieve] DRAFT Profile: Keeping {len(final_events)} notes (No NPS limit).")
        else:
            # STANDARD Mode (Buckets & Budgets)
            final_events = self._filter_adaptive_sieve(ranked_events, difficulty)
            print(f"  [Sieve] Selected {len(final_events)} notes (Target NPS: {target_nps})")
        
        consolidated_events = final_events
        sanitized_events = consolidated_events
        
        # --- STAGE 5: THE MapPER (Lane Allocation) ---
        chart_notes = self._allocate_lanes(sanitized_events, n_lanes)
        
        return {
            "metadata": {
                "difficulty": difficulty, 
                "bpm": self.bpm,
                "focus": focus_mode, 
                "lanes": n_lanes,
                "profile": chart_profile
            },
            "notes": chart_notes
        }

    def _apply_gameplay_filtering(self, events: List[NoteEvent], stems_folder: str) -> List[NoteEvent]:
        """
        Applies Velocity Gating, Silence Gating, and Smoothing based on Difficulty/Config.
        Previously in Engine._refine_events.
        """
        if not events: return []
        
        print("[ChartGenerator] Applying Gameplay Filtering...")
        
        # Group by Source
        # We match source to CONFIG keys
        filtered_events = []
        
        # Helper to find config
        def get_cfg(src):
            filters = GENERATOR_CONFIG["transcription_filters"]
            # 1. Exact match
            if src in filters: return filters[src]
            # 2. Source match list
            for k, val in filters.items():
                if src in val.get("source_match", []):
                    return val
            # 3. Default to 'other'
            return filters["other"]
            
        # Optimize: Batch by source
        by_source = {}
        for n in events:
            if n.source not in by_source: by_source[n.source] = []
            by_source[n.source].append(n)
            
        for source, notes in by_source.items():
            cfg = get_cfg(source)
            
            # 1. Velocity Gate (Percentile based logic logic from Engine)
            # engine.py: if vel_gate > 0: thresh = percent(gate*100)
            vel_gate = cfg.get("velocity_gate", 0.0)
            if vel_gate > 0 and notes:
                velocities = [n.velocity for n in notes]
                thresh = np.percentile(velocities, vel_gate * 100)
                # print(f"  [Filter] {source} VelGate {vel_gate} -> Thresh {thresh:.3f}")
                notes = [n for n in notes if n.velocity >= thresh]
                
            # 2. Smoothing (VocalSmoother)
            smoothing = cfg.get("smoothing", 0.0)
            if smoothing > 0 and notes:
                # print(f"  [Filter] {source} Smoothing {smoothing}")
                notes = VocalSmoother.smooth(notes, level=smoothing)
                
            # 3. Silence Gating
            silence_thresh = cfg.get("silence_threshold", 0.0)
            if silence_thresh > 0 and stems_folder:
                # Find audio file
                # Try source.wav, or fallback to mapped name
                # engine.py did: path = os.path.join(stems, f"{source}.wav")
                # if not exist and source=="vocals_lead", try "vocals.wav"
                
                path = os.path.join(stems_folder, f"{source}.wav")
                if not os.path.exists(path):
                    if source == "vocals_lead": path = os.path.join(stems_folder, "vocals.wav")
                    elif source == "vocals_fcpe": path = os.path.join(stems_folder, "vocals.wav")
                
                if os.path.exists(path):
                    # print(f"  [Filter] {source} Silence Gating ({silence_thresh})")
                    notes = EventFilter.gate_silence(notes, path, threshold=silence_thresh)
            
            filtered_events.extend(notes)
            
        return filtered_events

    def _consolidate_visuals(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Merges contiguous notes of the same source into visual 'Macro Holds'.
        The original notes are kept as 'Ghost Notes' for audio playback.
        """
        if not events: return []
        events.sort(key=lambda x: x.time)
        
        final_list = []
        
        # Group by source/lane logic? 
        # Actually just strictly contiguous vocal notes.
        # Instruments are usually fine as is (percussive).
        
        i = 0
        while i < len(events):
            current = events[i]
            
            # Only consolidate vocals
            if current.source not in ["vocals", "vocals_lead", "vocals_harmony"]:
                final_list.append(current)
                i += 1
                continue
                
            # Look ahead
            chain = [current]
            j = i + 1
            while j < len(events):
                next_evt = events[j]
                
                # Check continuity
                gap = next_evt.time - (current.time + current.duration)
                
                # Must be same source and very close (gap < 0.05)
                # And similar pitch? No, pitch changes are exactly what we are hiding!
                # Just continuity and source.
                if next_evt.source == current.source and abs(gap) < 0.05:
                    chain.append(next_evt)
                    current = next_evt # Advance current pointer for continuity check
                    j += 1
                else:
                    break
            
            if len(chain) > 1:
                # Create Macro Note
                start = chain[0].time
                end = chain[-1].time + chain[-1].duration
                
                macro = NoteEvent(
                    time=start, 
                    duration=end-start, 
                    pitch=chain[0].pitch, # Visual pitch (start)
                    velocity=max(n.velocity for n in chain),
                    source=chain[0].source
                )
                macro._score = max(n._score for n in chain) if hasattr(chain[0], "_score") else 1.0
                
                # Mark chain as ghosts
                for n in chain:
                    n.ghost = True
                    final_list.append(n)
                
                # Add Macro (Not ghost) is implicit default
                final_list.append(macro)
                
                i = j
            else:
                final_list.append(current)
                i += 1
                
        return final_list

    def _is_on_any_grid(self, time_in_beats: float, grids: List[int]) -> bool:
        """Checks if time aligns with any of the provided grid denominators."""
        if not grids: return True
        for g in grids:
            if g <= 0: continue
            step = 4.0 / float(g)
            # Check alignment (Tolerance: 0.05 beats)
            # Note: We use a tighter tolerance here than the quantizer to ensure "clean" picks
            if abs(round(time_in_beats / step) * step - time_in_beats) < 0.05:
                return True
        return False

    def _rank_events_layered(self, events: List[NoteEvent], timeline: LayerTimeline) -> List[NoteEvent]:
        """
        Rank events based on their Layer Assignment.
        Strict Mode: Non-Focus notes get 0 score.
        Support Mode: Must be on-beat.
        """
        if not events: return []
        
        scored_Events = []
        l_cfg = GENERATOR_CONFIG["layers"]
        s_cfg = GENERATOR_CONFIG["scoring"]
        v_cfg = GENERATOR_CONFIG.get("volume", {})
        sf_cfg = GENERATOR_CONFIG.get("layer_sifting", {})
        
        # Strict Grid Config
        strict_cfg = GENERATOR_CONFIG.get("strict_grids", {})
        enforce_grid = strict_cfg.get("enforce_pick_from_grid", False)
        main_grids = strict_cfg.get("main_layer_grid", [4, 8, 12, 16, 24, 32])
        support_grids = strict_cfg.get("support_layer_grid", [4, 8])

        beat_dur = 60.0 / self.bpm if self.bpm > 0 else 0.5
        
        for n in events:
            # Query the dynamic timeline for current layers
            primary_src, support_src = timeline.get_layers(n.time)
            
            # Base Score = Velocity
            base_score = n.velocity

            # --- RHYTHM WEIGHTING (Soft Snap) ---
            # NEW Phase 3 Logic: Measure-based Phase Scoring
            
            # 1. Calculate Phase in Beats
            time_in_beats = n.time / beat_dur
            beat_phase = time_in_beats % 4.0 # 0.0 to 3.999 (Assuming 4/4)
            beat_fraction = time_in_beats % 1.0
            
            # Use small epsilon (0.01) to check alignment
            r_weights = s_cfg.get("rhythm_weights", {})
            r_multiplier = r_weights.get("complex", 0.9) 
            
            # Check alignment
            if abs(beat_fraction) < 0.01 or abs(beat_fraction - 1.0) < 0.01:
                # It is ON A BEAT. Which one?
                if abs(beat_phase - 0.0) < 0.01 or abs(beat_phase - 4.0) < 0.01:
                    # Beat 1 (Downbeat)
                    r_multiplier = 1.3
                elif abs(beat_phase - 2.0) < 0.01:
                    # Beat 3 (Backbeat)
                    r_multiplier = 1.2
                else:
                    # Beats 2 & 4
                    r_multiplier = 1.1
            elif abs(beat_fraction - 0.5) < 0.01:
                # Eighth Note (Upbeat)
                r_multiplier = 1.05 # Slightly better than off-beat
            elif abs(beat_fraction - 0.25) < 0.01 or abs(beat_fraction - 0.75) < 0.01:
                # Sixteenth
                r_multiplier = 1.0
            
            base_score *= r_multiplier

            is_valid = False
            
            # --- VOLUME COMPENSATION START ---
            # 1. Instrument Boost
            inst_boost = v_cfg.get("instrument_boosts", {}).get(n.source, 1.0)
            n.velocity *= inst_boost
            
            # 2. Pitch Compensation (Bass Boost)
            # Fletcher-Munson: Low freq needs boost
            if n.pitch < 55: # Below G2
                 n.velocity *= 1.2
            
            # Layer Logic
            is_primary = n.source in primary_src
            is_support = n.source in support_src
            
            # --- STRICT GRID ENFORCEMENT ---
            if enforce_grid:
                grid_check_pass = False
                if is_primary:
                    grid_check_pass = self._is_on_any_grid(time_in_beats, main_grids)
                elif is_support:
                    grid_check_pass = self._is_on_any_grid(time_in_beats, support_grids)
                
                if not grid_check_pass:
                    # STRICT FAIL: Kill it immediately
                    base_score = 0.0
                    is_valid = False
                    # Skip further checks
                    setattr(n, "_score", base_score) 
                    continue

            if is_primary:
                base_score *= l_cfg["primary_multiplier"]
                is_valid = True
                
                # Active Layer Volume Boost
                n.velocity *= v_cfg.get("primary_boost", 1.2)
                
            elif is_support:
                # Support: GRID LIMIT CHECK
                # Active Layer Volume Penalty
                n.velocity *= v_cfg.get("support_penalty", 0.9)
                
                # --- MELODIC SUPPORT GATING ---
                # "Punish" melodic backing tracks (Guitar/Piano)
                if n.source not in sf_cfg.get("percussive_stems", []):
                    # 1. Gate: Check absolute velocity
                    # BasicPitch output is often low (0.1-0.4).
                    # Gate must be low enough to catch accents but high enough to filter noise.
                    gate_thresh = sf_cfg.get("melodic_support_gate", 0.25)
                    
                    if n.velocity < gate_thresh:
                        is_valid = False # Kill it
                        base_score = 0.0
                    else:
                        # 2. Weight: Deprioritize
                         base_score *= sf_cfg.get("melodic_support_weight", 0.5)
                
                # --- STRICT SUPPORT GRID SEIVE ---
                # Only allow support notes on a specific grid (e.g. 1/4 notes)
                # This helps secondary layers (like drums) feel more "rhythmic" and less "noisy"
                # Tuning parameter: sieving.support_grid (Default: 4 leads to 1/4 note alignment)
                
                if not enforce_grid:
                    grid_val = GENERATOR_CONFIG["sieving"].get("support_grid", 4)
                    grid_step = 4.0 / grid_val if grid_val > 0 else 1.0
                    is_on_grid = abs(round(time_in_beats / grid_step) * grid_step - time_in_beats) < 0.05
                    is_percussive = n.source in sf_cfg.get("percussive_stems", [])
                    
                    if is_on_grid:
                        # Bonus for perfectly on-beat notes
                        is_quarter = abs(beat_fraction) < 0.1 or abs(beat_fraction - 1.0) < 0.1
                        if is_quarter:
                            base_score *= l_cfg["support_multiplier"] * l_cfg["support_on_beat_bonus"]
                        else:
                            base_score *= l_cfg["support_multiplier"]
                        is_valid = True
                    elif is_percussive:
                        base_score = 0.0
                        is_valid = False
                    else:
                        base_score = 0.0
                        is_valid = False
                else:
                    # If enforce_grid is True, we already validated support_grids.
                    # Just apply multipliers.
                    base_score *= l_cfg["support_multiplier"]
                    is_valid = True

            else:
                # KILL NON-FOCUS
                base_score = 0.0
                is_valid = False
            
            # Clip Volume
            n.velocity = min(v_cfg.get("global_limit", 1.0), n.velocity)
            # --- VOLUME COMPENSATION END ---
                
            if is_valid:
                setattr(n, "_score", base_score) 
                scored_Events.append(n)
            
        # Coincidence Bonus (Shadowing)
        # If a Support note coincides with a Primary note, suppress the Support note?
        scored_Events.sort(key=lambda x: x.time)
        coincidence_window = l_cfg["shadow_window"]
        
        for i, n in enumerate(scored_Events):
            # Query layers for coincidence logic
            primary_src, support_src = timeline.get_layers(n.time)
            
            # Check neighbors
            for j in range(max(0, i-5), min(len(scored_Events), i+5)):
                if i == j: continue
                other = scored_Events[j]
                if abs(n.time - other.time) < coincidence_window:
                    # Logic: If I am Support and Other is Primary -> I get nuked?
                    if n.source in support_src and other.source in primary_src:
                        # Exception: Drums/Bass usually stack with melody. Don't nuke them.
                        # Only nuke "Melodic Support" (e.g. guitar backing under vocal lead)
                        if n.source in sf_cfg.get("percussive_stems", []):
                            pass # Keep the drum hit!
                        else:
                            n._score = 0.0 # Shadowed by primary
                    
                    # Logic: If both are same layer -> Coincidence Bonus (Chord)
                    elif (n.source in primary_src and other.source in primary_src):
                        n._score += s_cfg["coincidence_bonus"]

        # Final Filter: remove zero score notes
        return [e for e in scored_Events if getattr(e, "_score", 0) > 0.001]

    def _filter_adaptive_sieve(self, events: List[NoteEvent], difficulty: str) -> List[NoteEvent]:
        """
        New V301 "Bucket-Budget" Sieve.
        1. Windowing: Slices song into fixed windows (e.g. 1.0s).
        2. Budgeting: Selects top N notes per window based on NPS limit + Burst Allowance.
        3. Coalescing: Fuses discarded notes into nearby survivors (Ghost Chords).
        4. Physicality: Enforces strict speed limits post-selection.
        """
        if not events: return []
        
        # Config
        cfg = DIFF_CONFIGS.get(difficulty, DIFF_CONFIGS["NORMAL"])
        target_nps = cfg["nps"]
        
        sieving_cfg = GENERATOR_CONFIG["sieving"]
        min_global_interval = sieving_cfg["min_global_interval"]
        burst_allowance = 2 # notes allowed to overflow if budget permits
        
        window_size = GENERATOR_CONFIG["windowing"].get("size", 1.0)
        
        # Sort by time
        events.sort(key=lambda x: x.time)
        
        final_events = []
        
        # Max time
        max_t = events[-1].time + window_size
        num_windows = int(max_t / window_size) + 1
        
        # State
        budget_carryover = 0.0 # Unused budget from previous window
        
        for w in range(num_windows):
            start_t = w * window_size
            end_t = start_t + window_size
            
            # 1. Get Candidates in Window
            candidates = [e for e in events if start_t <= e.time < end_t]
            if not candidates:
                # Accumulate some budget? (Decay over time to prevent huge bursts after silence)
                budget_carryover = min(target_nps * 0.5, budget_carryover + (target_nps * window_size))
                continue
                
            # 2. Calculate Budget
            base_budget = int(target_nps * window_size)
            
            # Apply Carryover (Token Bucket)
            extra = int(budget_carryover) if budget_carryover > 0 else 0
            # Cap extra to avoid massive spam
            extra = min(extra, burst_allowance)
            
            total_budget = base_budget + extra
            
            # 3. Selection (Score-based)
            # Sort by Score Descending
            candidates.sort(key=lambda x: getattr(x, "_score", 0), reverse=True)
            
            selected = clean_window_selected = []
            rejected = []
            
            if len(candidates) <= total_budget:
                selected = candidates
                # Update carryover: We used fewer notes than allowed
                unused = total_budget - len(candidates)
                budget_carryover = min(target_nps, unused) # Cap carryover
            else:
                selected = candidates[:total_budget]
                rejected = candidates[total_budget:]
                budget_carryover = 0 # Budget exhausted
            
            # Sort selected by time for proximity checks
            selected.sort(key=lambda x: x.time)
            
            # 4. Audio Coalescing (The Fix)
            # Fuse rejected notes into nearest selected note
            tolerance = 0.02 # 20ms window for chord fusion
            
            for r in rejected:
                # Find nearest neighbor in selected
                if not selected: break # Should not happen if budget > 0
                
                # Binary search or simple linear scan (Window is small ~ 5-10 notes)
                nearest = min(selected, key=lambda x: abs(x.time - r.time))
                gap = abs(nearest.time - r.time)
                
                if gap <= tolerance:
                    # Fuse Pitch!
                    # Ensure audio_coalesced_pitches exists
                    if not hasattr(nearest, "audio_coalesced_pitches"):
                        nearest.audio_coalesced_pitches = []
                    
                    # Add my pitch
                    nearest.audio_coalesced_pitches.append(int(r.pitch))
                    
                    # If I had my own pool (from previous merges), add those too
                    if hasattr(r, "audio_coalesced_pitches"):
                        nearest.audio_coalesced_pitches.extend(r.audio_coalesced_pitches)
                        
                    # print(f"  [Sieve] Coalesced {r.pitch} into {nearest.pitch} (Gap {gap*1000:.1f}ms)")
            
            final_events.extend(selected)
            
        # 5. Global Physical Validity Pass (Post-Selection)
        # Enforce Min Interval. If violation, coalesce into previous.
        final_events.sort(key=lambda x: x.time)
        physically_valid = []
        
        last_t = -1.0
        
        for i, curr in enumerate(final_events):
            if i == 0:
                physically_valid.append(curr)
                last_t = curr.time
                continue
            
            dt = curr.time - last_t
            
            if dt < min_global_interval:
                # VIOLATION!
                # Coalesce into previous note (the one that caused the interval constraint)
                prev = physically_valid[-1]
                
                # Fuse
                if not hasattr(prev, "audio_coalesced_pitches"):
                        prev.audio_coalesced_pitches = []
                
                prev.audio_coalesced_pitches.append(int(curr.pitch))
                if hasattr(curr, "audio_coalesced_pitches"):
                    prev.audio_coalesced_pitches.extend(curr.audio_coalesced_pitches)

                # print(f"  [Sieve] Speed Limit Coalesce {curr.pitch} -> {prev.pitch} (dt {dt:.3f}s)")
                # Discard 'curr'
            else:
                physically_valid.append(curr)
                last_t = curr.time
        
        print(f"  [Sieve] Bucket-Budget: {len(events)} -> {len(physically_valid)} notes (Diff: {difficulty})")
        return physically_valid

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
        
        # Calculate dynamic hold threshold based on BPM
        beat_dur = 60.0 / self.bpm
        # Base threshold: 3/8 note (dotted quarter) or config min, whichever is larger
        dynamic_thresh = beat_dur * 0.75  # 3/8 note
        hold_thresh = max(min_hold_dur, dynamic_thresh)
        
        # Vocals get a lower threshold (more holds for sustained singing)
        # ~1/4 note minimum for vocals (more natural for lyrical content)
        vocal_hold_thresh = max(0.25, beat_dur * 0.5)
        
        # Note: _sanitize_holds already called in generate() - don't call again
        
        for ev in events:
            # GHOST NOTE HANDLING (Audio Only, No Visuals)
            if getattr(ev, "ghost", False):
                processed.append({
                    "time": round(float(ev.time), 3),
                    "lane": -1, # HIDDEN
                    "dur": round(float(ev.duration), 2), # Keep duration for audio synth bucket?
                    "type": "ghost",
                    "midi": int(ev.pitch),
                    "vol": round(float(getattr(ev, "velocity", 0.8)), 2),
                    "source": str(ev.source),
                    "ghost": True
                })
                continue

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
            
            # Helper to round duration to nearest 0.05 (preserving original intention but aligning to grid)
            def snap_dur(d):
                return round(max(0.05, round(d / 0.05) * 0.05), 2)
            
            snapped_duration = snap_dur(float(ev.duration))
            normalized_time = round(float(ev.time), 3)
            normalized_vol = round(float(getattr(ev, "velocity", 0.8)), 2)
            
            # Check hold type
            # 1. Must be allowed stem
            # 2. Must exceed threshold (vocal-specific for sung notes)
            is_hold = False
            if ev.source in allowed_holds:
                # Use lower threshold for vocals (sustained singing)
                thresh = vocal_hold_thresh if ev.source in ["vocals", "vocals_lead"] else hold_thresh
                if ev.duration > thresh:
                    is_hold = True
            
            # Audio Pool (Coalesced Pitches for Polyphony)
            audio_pool = getattr(ev, "audio_coalesced_pitches", [])
            if not audio_pool:
                audio_pool = [int(ev.pitch)]
            
            processed.append({
                "time": normalized_time,
                "lane": int(lane),
                "dur": snapped_duration, # Always use snapped duration (never 0.0)
                "type": "hold" if is_hold else "tap",
                "midi": int(ev.pitch),
                "vol": normalized_vol, # Default vol if missing
                "source": str(ev.source),
                "audio_pool": audio_pool
            })
        
        # Post-process: Resolve conflicts (Hold overlaps)
        # Includes reaction time logic (moved from _sanitize_holds)
        conflict_resolved = self._resolve_conflicts(processed)
        
        # Cross-stem anti-spam: Re-enabled with Smart Fuse (Chord creation)
        return self._resolve_cross_stem_conflicts(conflict_resolved)



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
        gap = GENERATOR_CONFIG["holds"]["min_reaction_time"] # 0.1s (Unified from _sanitize_holds)
        minijack_thresh = 0.05 # 50ms (20NPS cap for same-lane jacks) - Loosened from 0.1 to allow bursts
        allowed_holds = GENERATOR_CONFIG["holds"]["allowed_stems"]
        
        merged_count = 0

        for l in lanes_dict:
            # Sort by time
            stack = sorted(lanes_dict[l], key=lambda x: x["time"])
            
            # Merged stack
            merged_stack = []
            if not stack: continue
            
            # Pass 1: Merge close notes (Minijacks)
            curr = stack[0]
            for i in range(1, len(stack)):
                next_n = stack[i]
                
                # Check proximity
                gap_val = next_n["time"] - curr["time"]
                
                # Logic: Only merge if strictly a "Jack" (Same Pitch + Close Time)
                # If pitch is different, it's a trill/stream -> Allow it (unless very fast glitch)
                is_same_pitch =  abs(next_n["midi"] - curr["midi"]) < 1.0
                
                # Thresholds
                # Same Pitch: minijack_thresh
                # Different Pitch: 0.04 (glitch removal, very fast trills allowed > 40ms)
                thresh = minijack_thresh if is_same_pitch else 0.04
                
                if gap_val < thresh:
                    # Merge!
                    merged_count += 1
                    
                    # --- AUDIO PRESERVATION FIX ---
                    # Ensure current note has a pool
                    if "audio_pool" not in curr: 
                        curr["audio_pool"] = [curr["midi"]]
                    
                    # Get next note's pool (safely)
                    next_pool = next_n.get("audio_pool", [next_n["midi"]])
                    
                    # Extend current pool with next note's pool
                    curr["audio_pool"].extend(next_pool)
                    # ------------------------------
                    
                    # Only create hold if allowed
                    can_hold = curr["source"] in allowed_holds
                    
                    if can_hold and is_same_pitch:
                        # Extend current duration to cover next note
                        new_end = max(curr["time"] + curr["dur"], next_n["time"] + next_n["dur"])
                        new_dur = new_end - curr["time"]
                        
                        # Only make it a hold if merged duration is meaningful (>= 0.25s)
                        if new_dur >= 0.25:
                            curr["dur"] = round(new_dur, 2)
                            curr["type"] = "hold"
                        else:
                            # Too short - just keep as tap
                            curr["dur"] = round(max(0.05, round(curr["dur"] / 0.05) * 0.05), 2)
                            curr["type"] = "tap"
                    else:
                        # Cannot hold (e.g. guitar/drums) OR Trill glitch
                        # Absorb the next note.
                        curr["dur"] = round(max(0.05, round(curr["dur"] / 0.05) * 0.05), 2)
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
                    
                    # Limit = next note time - reaction gap
                    # Use the configured gap (0.1s usually)
                    limit = next_note["time"] - gap
                    end_t = current["time"] + current["dur"]
                    
                    if end_t > limit:
                        # Trim duration
                        new_dur = max(0.0, limit - current["time"])
                        current["dur"] = round(new_dur, 2)
                        
                        # If too short for a hold, convert back to tap
                        # Use 0.25s as minimum visible hold
                        if new_dur < 0.25:
                            current["dur"] = round(max(0.05, round(new_dur / 0.05) * 0.05), 2)
                            current["type"] = "tap"
                
                final.append(current)
        
        if merged_count > 0:
            print(f"    [LaneResolver] Merged {merged_count} minijacks (<{minijack_thresh*1000}ms)")

        # Re-sort all by time
        final_sorted = sorted(final, key=lambda x: x["time"])
        
        # --- STRICT LANE SAFETY CHECK ---
        # Ensure per-lane gap >= min_global_interval
        # (This catches any Quantizer artifacts or Logic slips)
        # Note: We allowed chords (diff lanes) but SAME lane must have gap.
        
        safe_final = []
        last_times_lane = {} # lane_idx -> last_note_end_time
        
        strict_gap = GENERATOR_CONFIG["sieving"]["min_global_interval"]
        strict_drops = 0
        
        for n in final_sorted:
            l = n["lane"]
            start = n["time"]
            prev_end = last_times_lane.get(l, -10.0)
            
            # Calculate gap from previous note IN THIS LANE
            # (Note: prev_end includes duration)
            if start < prev_end + strict_gap:
                # VIOLATION!
                strict_drops += 1
                continue
                
            safe_final.append(n)
            last_times_lane[l] = start + n["dur"]

        if strict_drops > 0:
            print(f"    [LaneResolver] Dropped {strict_drops} unsafe notes (Overlaps < {strict_gap}s)")
            
        return safe_final

    def _resolve_cross_stem_conflicts(self, notes: List[dict]) -> List[dict]:
        """
        Anti-spam filter for cross-stem conflicts.
        Fuses or removes secondary layer notes that are too close to primary layer notes.
        
        Priority hierarchy (highest to lowest):
        1. vocals, vocals_lead (primary melodic)
        2. bass (primary rhythm)
        3. drums (secondary rhythm)
        4. guitar, piano (melodic support)
        5. other (composite/backup)
        
        Logic:
        - FUSE Window (50ms): If a lower-priority note is extremely close, snap it to the 
                            higher-priority note's time to create a perfect chord.
        - CONFLICT Window (90ms): If it's close but not fuse-able, delete the lower-priority 
                                note to avoid muddy "double taps".
        """
        if not notes:
            return notes
            
        # Define stem priority (higher = more important)
        stem_priority = {
            "vocals": 100,
            "vocals_lead": 100,
            "bass": 80,
            "drums": 60,
            "guitar": 50,
            "piano": 50,
            "other": 20
        }
        
        fuse_window = 0.05      # 50ms - Snap to chord
        conflict_window = 0.09  # 90ms - Delete conflict
        
        # Sort by time for efficient scanning
        sorted_notes = sorted(notes, key=lambda x: x["time"])
        
        # Track which notes to remove
        remove_indices = set()
        
        # Pass 1: Identification & Timing Adjustment
        for i in range(len(sorted_notes)):
            if i in remove_indices:
                continue
                
            current = sorted_notes[i]
            current_priority = stem_priority.get(current["source"], 0)
            
            # Check neighbors within the conflict window
            # We look ahead to find conflicts with higher priority notes
            for j in range(i + 1, len(sorted_notes)):
                other = sorted_notes[j]
                dt = other["time"] - current["time"]
                
                if dt > conflict_window:
                    break
                    
                other_priority = stem_priority.get(other["source"], 0)
                
                if other_priority == current_priority:
                    continue # For same priority, we let the grid/polyphony handle it earlier
                
                # Identify lower and higher priority notes
                if other_priority > current_priority:
                    # 'current' is lower priority
                    lower_idx = i
                    higher_note = other
                    delta = dt
                else:
                    # 'other' is lower priority
                    lower_idx = j
                    higher_note = current
                    delta = dt

                # Apply Logic to the lower priority note
                if delta < fuse_window:
                    # FUSE: Snap lower note to higher note's time
                    sorted_notes[lower_idx]["time"] = round(higher_note["time"], 3)
                    # print(f"    [SmartFuse] Fused {sorted_notes[lower_idx]['source']} into {higher_note['source']} at {higher_note['time']:.3f}s")
                else:
                    # CONFLICT: Delete lower priority note
                    remove_indices.add(lower_idx)
                    # print(f"    [SmartFuse] Removed {sorted_notes[lower_idx]['source']} (Conflict with {higher_note['source']} at {higher_note['time']:.3f}s)")

        # Filter to only kept notes
        filtered = [sorted_notes[i] for i in range(len(sorted_notes)) if i not in remove_indices]
        
        # Re-sort because Fusing changed some times
        filtered.sort(key=lambda x: x["time"])
        
        removed_count = len(notes) - len(filtered)
        if removed_count > 0:
            print(f"    [Cross-Stem Filter] Cleaned {removed_count} conflicts via SmartFuse (Windows: {fuse_window*1000}ms/{conflict_window*1000}ms)")
        
        return filtered

    def _merge_fragmented_notes(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Merges fragmented consecutive notes of similar pitch into single longer notes.
        
        This catches cases where a sustained vocal gets split by the transcriber:
        - Short onset artifact + long sustained note → single hold
        - Vibrato causing multiple short notes → single hold
        
        Conservative criteria to avoid breaking intentional staccato:
        - Same source (stem)
        - Similar pitch (within 1.5 semitones)
        - Very small gap (< 50ms)
        - Leading note is short (< 200ms) - if first note is long, it's probably intentional
        """
        if not events:
            return []
        
        # Only apply to melodic stems (not drums/bass)
        melodic_stems = {"vocals", "vocals_lead", "piano", "guitar", "other"}
        
        # Separate melodic and non-melodic
        melodic = [e for e in events if e.source in melodic_stems]
        non_melodic = [e for e in events if e.source not in melodic_stems]
        
        if not melodic:
            return events
        
        # Group by source for independent processing
        from collections import defaultdict
        by_source = defaultdict(list)
        for e in melodic:
            by_source[e.source].append(e)
        
        merged_all = []
        
        for source, notes in by_source.items():
            notes.sort(key=lambda x: x.time)
            
            merged = []
            i = 0
            
            while i < len(notes):
                curr = notes[i]
                
                # Look ahead for merge candidates
                chain = [curr]
                j = i + 1
                
                while j < len(notes):
                    next_n = notes[j]
                    prev = chain[-1]
                    
                    # 200ms Lookahead Max (Limit search to immediate neighbors)
                    if next_n.time > prev.time + prev.duration + 0.2:
                        break
                        
                    # Gap between end of previous and start of next
                    gap = next_n.time - (prev.time + prev.duration)
                    
                    # Pitch difference
                    pitch_diff = abs(next_n.pitch - curr.pitch)  # Compare to FIRST note's pitch
                    
                    # Merge criteria (conservative)
                    should_merge = (
                        gap < 0.1 and  # Very small gap (< 100ms)
                        pitch_diff < 1.5 and  # Similar pitch
                        prev.duration < 0.20  # Previous note is short (likely artifact)
                    )
                    
                    if should_merge:
                        chain.append(next_n)
                        j += 1
                    else:
                        break
                
                if len(chain) > 1:
                    # Merge the chain into a single note
                    start_time = chain[0].time
                    end_time = chain[-1].time + chain[-1].duration
                    
                    # Pitch: duration-weighted average (biased toward longer notes)
                    total_dur = sum(n.duration for n in chain)
                    if total_dur > 0:
                        weighted_pitch = sum(n.pitch * n.duration for n in chain) / total_dur
                    else:
                        weighted_pitch = chain[0].pitch
                    
                    # Velocity: max of chain
                    max_vel = max(n.velocity for n in chain)
                    
                    merged_note = NoteEvent(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=weighted_pitch,
                        velocity=max_vel,
                        source=source
                    )
                    merged.append(merged_note)
                else:
                    merged.append(curr)
                
                i = j if len(chain) > 1 else i + 1
            
            merged_all.extend(merged)
        
        # Combine with non-melodic
        result = merged_all + non_melodic
        result.sort(key=lambda x: x.time)
        
        return result
