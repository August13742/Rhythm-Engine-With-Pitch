import os
import argparse
import sys
import json
import random
import copy
from typing import List
import numpy as np
import librosa


# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beatmap import Beatmap, NoteEvent, EventFilter, Quantizer
from transcribe.basic_pitch import BasicPitchTranscriber
from transcribe.council import CouncilV2

# Configuration for Visualizer compatibility & Generator Logic
DIFF_CONFIGS = {
    "EASY":   {"lanes": 4, "nps": 4.0},
    "NORMAL": {"lanes": 4, "nps": 6.0},
    "HARD":   {"lanes": 4, "nps": 8.0},
    "ALT_HARD": {"lanes": 4, "nps": 8.0}
}
# Removed hardcoded primary/support from DIFF_CONFIGS because it is now dynamic

class AudioCache:
    """
    Simple cache to prevent re-loading the same audio file multiple times.
    Stores separate buffers for different sample rates.
    """
    _cache = {}

    @classmethod
    def get(cls, path: str, sr: int = 22050):
        key = (path, sr)
        if key not in cls._cache:
            if not os.path.exists(path):
                return None, None
            # print(f"[Cache] Loading {os.path.basename(path)} (sr={sr})...")
            y, s = librosa.load(path, sr=sr)
            cls._cache[key] = (y, s)
        else:
            # print(f"[Cache] Hit: {os.path.basename(path)} (sr={sr})")
            pass
        return cls._cache[key]

    @classmethod
    def clear(cls):
        cls._cache.clear()

GENERATOR_CONFIG = {
    "cleaning": {
        "min_duration": 0.03,
        "min_velocity": 0.1,
        "roll_consolidation_gap": 0.03
    },
    "holds": {
        "allowed_stems": ["vocals", "vocals_lead", "other"],
        "min_duration": 0.75,
        "max_vocal_duration": 5.5,
        "min_reaction_time": 0.10 
    },
    "sieving": {
        "burst_limit": 0.4, 
        "min_global_interval": 0.05, 
    },
    "scoring": {
        "weights": {
            "vocals": 1.5, "vocals_lead": 1.6,
            "piano": 1.2, "guitar": 1.2,
            "drums": 1.1, "bass": 1.0,
            "other": 0.8
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
        "melodic_support_gate": 0.25,   # Min velocity for Melodic Support notes
        "melodic_support_weight": 0.5,  # Score penalty (Deprioritize in density)
        "percussive_stems": ["drums", "bass"] # Stems EXEMPT from sifting
    }
}

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
        "onset": 0.40, "frame": 0.30,
        "smoothing": 0.0,
        "multipass": True, # 'Other' is chaotic, consensus might fail
        "min_len": 58.0
    }
}


class StemSelector:
    @staticmethod
    def get_stem_energy(stem_name: str, manifest: dict) -> float:
        """Returns the peak energy or RMS from manifest."""
        if not manifest or stem_name not in manifest:
            return 0.0
        
        info = manifest[stem_name]
        # Prefer RMS (avg power) over Peak (transients) for "dominance"
        # manifest currently stores 'peak_energy'. Let's assume we might have 'rms' or use peak.
        # Original separator also keys 'vocals_lead' rms into a stats dict but manifest saves peak.
        # Fallback to peak if RMS missing.
        return info.get("peak_energy", 0.0)

    @staticmethod
    def select_layers(difficulty: str, manifest: dict, focus_mode: str = "main", stem_weights: dict = None) -> dict:
        """
        Determines Primary/Support stems based on Difficulty and Focus Mode.
        focus_mode: "main" (Vocals/Melody) or "alt" (Instruments/Rhythm)
        stem_weights: dict of {source: total_velocity} for dynamic selection.
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
        # Force specific modes for specific difficulties based on new user mapping:
        # EASY/NORMAL/HARD -> MAIN Focus
        # ALT_HARD -> ALT Focus (Alternative Hard/Complimentary)
        
        actual_mode = focus_mode
        if difficulty in ["EASY", "NORMAL", "HARD"]:
            actual_mode = "main"
        elif difficulty == "ALT_HARD":
            actual_mode = "alt"
            
        print(f"  [Layers] Difficulty: {difficulty} (Mode: {actual_mode})")

        if actual_mode == "main":
            # MAIN FOCUS: Vocals > Lead > Rhythm
            if has_vocals:
                # Prefer Lead
                if "vocals_lead" in active_stems: primary.append("vocals_lead")
                elif "vocals" in active_stems: primary.append("vocals")
                
                # In Main/Easy-Hard, Instruments are Backing Tracks.
                # Only add vocals/lead to primary. Support will handle rhythm.
                pass
            
            else:
                # Instrumental Song: Leads are Primary
                # Dynamic Rule: Primary = Most Dynamic Stem
                candidates = [s for s in active_stems if s not in ["drums", "bass"]]
                if candidates:
                    # Dynamic Sort using Energy (RMS/Peak) + Note Velocity Sum (stem_weights)
                    # stem_weights (velocity sum) serves as a proxy for "musical activity"
                    if stem_weights:
                        candidates.sort(key=lambda s: stem_weights.get(s, 0), reverse=True)
                        print(f"    > Instrumental Main Sort: {candidates}")
                        primary.append(candidates[0])
                    else:
                        # Fallback
                        if "guitar" in active_stems: primary.append("guitar")
                        elif "piano" in active_stems: primary.append("piano")
                        elif "other" in active_stems: primary.append("other")

            # Support: Rhythm (Drums)
            if difficulty != "EASY":
                 if "drums" in active_stems: support.append("drums")
            
        elif actual_mode == "alt":
            # ALT FOCUS: Instruments (2nd Busiest) > Rhythm
            # Dynamic Rule: Pick highest energy non-vocal instrument.
            # Order of preference: PURELY DYNAMIC.
            
            # Filter candidates: All melodic instruments (No Vocals, No Drums, No Bass)
            candidates = [s for s in active_stems if s not in ["vocals", "vocals_lead", "drums", "bass"]]
            
            if candidates:
                if stem_weights:
                    # Sort by weight desc (Total Note Velocity)
                    candidates.sort(key=lambda s: stem_weights.get(s, 0), reverse=True)
                    # print(f"    > Dynamic Sort: {candidates} (Scores: {[f'{stem_weights.get(s,0):.3f}' for s in candidates]})")
                    
                    selected_idx = 0
                    if not has_vocals and len(candidates) > 1:
                        # Instrumental Mode: Select 2nd best track for ALT diversity
                        selected_idx = 1
                        
                    primary.append(candidates[selected_idx])
                else:
                    # Fallback Priority
                    if "guitar" in candidates: primary.append("guitar")
                    elif "piano" in candidates: primary.append("piano")
                    elif "other" in candidates: primary.append("other")
            
            # If no melody instruments, Drums become primary (Drum Chart)
            if not primary and "drums" in active_stems:
                primary.append("drums")
            elif "drums" in active_stems:
                # Drums are strictly support in ALT mode (unless it's a drum chart)
                support.append("drums")
                    
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
        # Calculate Stem Weights (Energy = Sum of Velocity)
        # This helps ALT mode pick the most dominant instrument dynamically.
        stem_weights = {}
        for e in events:
            s = getattr(e, "source", "other")
            v = getattr(e, "velocity", 0.5)
            stem_weights[s] = stem_weights.get(s, 0.0) + v
            
        layers = StemSelector.select_layers(difficulty, manifest, focus_mode, stem_weights)
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
        strict_stems = ["drums"]
        # Restricting drums to 1/4 and 1/8 only (Beat Feel)
        strict_grids = [4]
        
        # Split events
        group_strict = []
        group_free = []
        group_unsnapped = []
        
        for e in clean_events:
            if e.source in strict_stems:
                group_strict.append(e)
            else:
                # Vocals, Piano, Guitar, Bass, Other -> ALL UNSNAPPED (High Fidelity)
                # We rely on DSP Grounding in Stage 0 for timing accuracy.
                group_unsnapped.append(e)
                
        print(f"DEBUG QUANTIZER: Strict(Drums)={len(group_strict)}, Unsnapped(All Else)={len(group_unsnapped)}")
            
        # Snap separately
        snapped_strict = self.quantizer.snap_to_grid(group_strict, grids=strict_grids)
        # snapped_free = self.quantizer.snap_to_grid(group_free, grids=allowed_grids) # IGNORED
        
        quantized_events = snapped_strict + group_unsnapped
        
        # --- STAGE 3: THE SIEVE (Scoring & Selection) ---
        ranked_events = self._rank_events_layered(quantized_events, primary_src, support_src)
        
        # Using New Adaptive Sieve (V300)
        final_events = self._filter_adaptive_sieve(ranked_events, difficulty)
        
        print(f"  [Sieve] Selected {len(final_events)} notes (Target NPS: {target_nps})")
        
        # --- STAGE 2.5: MACRO HOLDS (Visual Consolidation) ---
        # Consolidated events skipped for now to avoid confusion
        # consolidated_events = self._consolidate_visuals(final_events)
        consolidated_events = final_events
        
        # --- STAGE 4: PRE-MAPPING CLEANUP ---
        # 6. Sanitize Holds (Uniform Rule)
        sanitized_events = self._sanitize_holds(consolidated_events)
        
        # 7. Deduplicate Same-Pitch (Vibrato/Glitch Fix) - PRE-LANE ALLOCATION
        deduped_events = self._deduplicate_same_pitch(sanitized_events)
        
        # --- STAGE 5: THE MAPPER (Lane Allocation) ---
        chart_notes = self._allocate_lanes(deduped_events, n_lanes)
        
        return {
            "metadata": {
                "difficulty": difficulty, 
                "version": "V300", 
                "bpm": self.bpm,
                "focus": focus_mode  # Export Focus Mode for Visualizer
            },
            "notes": chart_notes
        }

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
        v_cfg = GENERATOR_CONFIG.get("volume", {})
        sf_cfg = GENERATOR_CONFIG.get("layer_sifting", {})
        
        beat_dur = 60.0 / self.bpm if self.bpm > 0 else 0.5
        
        for n in events:
            # Base Score = Velocity
            base_score = n.velocity
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
            if n.source in primary_src:
                base_score *= l_cfg["primary_multiplier"]
                is_valid = True
                
                # Active Layer Volume Boost
                n.velocity *= v_cfg.get("primary_boost", 1.2)
                
            elif n.source in support_src:
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
                
                # Only allow support notes on main beats (1/4, 1/8) UNLESS it's Percussion
                time_in_beats = n.time / beat_dur
                beat_fraction = time_in_beats % 1.0
                is_quarter = abs(beat_fraction) < 0.1 or abs(beat_fraction - 1.0) < 0.1
                is_eighth = abs(beat_fraction - 0.5) < 0.1
                
                is_percussive = n.source in sf_cfg.get("percussive_stems", [])
                
                if is_quarter:
                     base_score *= l_cfg["support_multiplier"] * l_cfg["support_on_beat_bonus"]
                     is_valid = True 
                elif is_eighth:
                     base_score *= l_cfg["support_multiplier"]
                     is_valid = True
                elif is_percussive:
                 # The Quantizer (Stage 2) has ALREADY snapped these to [4, 8, 16].
                 base_score *= 2.5 
                 is_valid = True
                else:
                     base_score = 0.0
                     is_valid = False
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
        New V300 "Game-Like" Sieve.
        1. Clusters notes by Quantized Grid Time.
        2. Enforces Polyphony Limit (Max simultaneous notes).
        3. Enforces Max NPS Limit via Dynamic Grid Adaptation (Decimation) instead of random cuts.
        4. Allows Bursts if budget permits.
        """
        if not events: return []
        
        # Difficulty Tuning
        diff_limits = {
            "EASY":   {"nps": 4.0, "poly": 1, "min_interval": 0.25},
            "NORMAL": {"nps": 6.0, "poly": 2, "min_interval": 0.16},
            "HARD":   {"nps": 12.0, "poly": 2, "min_interval": 0.08}, # 12.5 NPS = 0.08s
            "ALT_HARD": {"nps": 14.0, "poly": 2, "min_interval": 0.07}
        }
        cfg = diff_limits.get(difficulty, diff_limits["NORMAL"])
        
        target_nps = cfg["nps"]
        max_poly = cfg["poly"]
        base_min_interval = cfg["min_interval"]
        
        # Global burst limits
        burst_limit_dur = GENERATOR_CONFIG["sieving"]["burst_limit"] # 0.4s
        abs_min_interval = GENERATOR_CONFIG["sieving"]["min_global_interval"] # 0.05s
        
        # 1. Cluster by Time (Quantize slightly to group chords)
        # We assume Stage 2 (Quantizer) has already run, so notes should be grid-aligned.
        # But we group by very small epsilon to catch floating point drifts.
        events.sort(key=lambda x: x.time)
        clusters = []
        if not events: return []
        
        curr_t = events[0].time
        curr_notes = [events[0]]
        
        for i in range(1, len(events)):
            evt = events[i]
            if abs(evt.time - curr_t) < 0.012: # 12ms chord window (Prevents 1ms jitter leaks)
                curr_notes.append(evt)
            else:
                # Push previous cluster
                clusters.append({"time": curr_t, "notes": curr_notes})
                curr_t = evt.time
                curr_notes = [evt]
        clusters.append({"time": curr_t, "notes": curr_notes})
        
        # 2. Polyphony Pass (Strict)
        # If cluster > max_poly, keep top N highest velocity
        pruned_clusters = []
        for c in clusters:
            notes = c["notes"]
            if len(notes) > max_poly:
                notes.sort(key=lambda x: x.velocity, reverse=True)
                pruned_clusters.append({"time": c["time"], "notes": notes[:max_poly]})
            else:
                pruned_clusters.append(c)
                
        # 3. Dynamic Decimation Pass (The "Speed Limit")
        final_clusters = []
        last_time = -10.0
        
        # Burst State
        burst_start_time = -1.0
        in_burst = False
        
        # Sliding Window for NPS calculation (Token Bucket light version)
        # We check local density in 1.0s window? 
        # Actually user requested "Max possible note per second at any moment" -> Minimum Interval.
        # So we stick to Interval checks.
        
        i = 0
        while i < len(pruned_clusters):
            curr = pruned_clusters[i]
            t = curr["time"]
            
            dt = t - last_time
            
            # Check Interval
            required_int = base_min_interval
            
            # Burst Logic
            is_burst = False
            if dt < base_min_interval:
                # Potential Burst
                # Check if we can allow it
                # 1. Must be > abs_min_interval (Physical Limit)
                if dt >= abs_min_interval:
                    # 2. Check Burst Duration (Are we already bursting?)
                    if not in_burst:
                        in_burst = True
                        burst_start_time = last_time # Start of burst
                    
                    burst_dur = t - burst_start_time
                    if burst_dur < burst_limit_dur:
                        # ALLOW BURST
                        is_burst = True
                        required_int = abs_min_interval # Use tighter limit
                    else:
                        # BURST EXHAUSTED
                        is_burst = False
                else:
                    # Too fast even for burst
                    is_burst = False
            else:
                # Reset burst if we had a nice gap
                if dt > base_min_interval * 1.5:
                    in_burst = False
            
            if dt >= required_int:
                # Accept
                final_clusters.append(curr)
                last_time = t
                i += 1
            else:
                # REJECT via Dynamic Decimation
                # Instead of skipping 'curr', we try to find the NEXT event that satisfies the interval.
                # This effectively downsamples the rhythm (e.g. 1/16 -> 1/8).
                
                # Scan ahead until we find a note >= required_int from last_time
                found_next = False
                j = i + 1
                while j < len(pruned_clusters):
                    next_cand = pruned_clusters[j]
                    dist = next_cand["time"] - last_time
                    
                    # If we are decimating, we should aim for the 'base_min_interval' (Standard Speed)
                    # to restore stability, rather than maintaining the burst speed.
                    target_int = base_min_interval 
                    
                    if dist >= target_int:
                        # Found valid next step
                        # Skip everything between i and j
                        # Wait, we need to re-evaluate 'j' as the new candidate in the main loop
                        i = j
                        found_next = True
                        break
                    j += 1
                
                if not found_next:
                    # No more valid notes in song
                    break
                    
        # Flatten and Force Snap Times (Prevent jittery chords)
        final_events = []
        for c in final_clusters:
            t = c["time"]
            for n in c["notes"]:
                n.time = t # Force mathematical alignment for chords
            final_events.extend(c["notes"])
            
        print(f"  [Sieve] Adaptive: {len(events)} -> {len(final_events)} notes (Diff: {difficulty})")
        return final_events

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
        
        
        # Pre-process: Sanitize Holds (Global Reaction Time Check)
        events = self._sanitize_holds(events)
        
        for ev in events:
            # GHOST NOTE HANDLING (Audio Only, No Visuals)
            if getattr(ev, "ghost", False):
                processed.append({
                    "time": float(ev.time),
                    "lane": -1, # HIDDEN
                    "dur": float(ev.duration), # Keep duration for audio synth bucket?
                    "type": "ghost",
                    "midi": int(ev.pitch),
                    "vol": float(getattr(ev, "velocity", 0.8)),
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

    def _sanitize_holds(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Force short holds to taps.
        Force tap if gap to next note is too small (Reaction Time).
        """
        if not events: return []
        events.sort(key=lambda x: x.time)
        
        min_reaction = GENERATOR_CONFIG["holds"]["min_reaction_time"] # 0.1s
        min_dur = GENERATOR_CONFIG["holds"]["min_duration"] # 0.2s
        
        for i in range(len(events)):
            curr = events[i]
            
            # 1. Min Duration Check
            if curr.duration < min_dur:
                curr.duration = 0.0 # Force Tap
                continue
                
            # 2. Gap Check (Lookahead)
            # Find next note (ignoring chords at same time)
            j = i + 1
            while j < len(events):
                next_n = events[j]
                if next_n.time > curr.time + 0.01:
                    # Found next sequential note
                    gap = next_n.time - (curr.time + curr.duration)
                    if gap < min_reaction:
                        # Too fast! Force Tap.
                        # print(f"  [Hold] Forced Tap at {curr.time:.2f}s (Gap {gap:.3f}s < {min_reaction}s)")
                        curr.duration = 0.0
                    break
                j += 1
                
        return events

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
        minijack_thresh = 0.18 # UPDATED: 180ms threshold for same-pitch merging (Vibrato)
        allowed_holds = GENERATOR_CONFIG["holds"]["allowed_stems"]
        percussive_stems = GENERATOR_CONFIG["layer_sifting"]["percussive_stems"]

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
                gap = next_n["time"] - curr["time"]
                
                # Logic: Only merge if strictly a "Jack" (Same Pitch + Close Time)
                # If pitch is different, it's a trill/stream -> Allow it (unless very fast glitch)
                is_same_pitch =  abs(next_n["midi"] - curr["midi"]) < 1.0
                
                # Thresholds
                # Melodic (Vocals/Piano): 0.18s (Anti-Vibrato)
                # Percussive (Drums): 0.04s (Allow Rolls, just filter glitches)
                is_percussive = curr["source"] in percussive_stems
                
                base_thresh = 0.04 if is_percussive else minijack_thresh
                
                # Different Pitch: always 0.04 (glitch removal)
                # Same Pitch: use base_thresh
                thresh = base_thresh if is_same_pitch else 0.04
                
                if gap < thresh:
                    # Merge!
                    # Only create hold if allowed
                    can_hold = curr["source"] in allowed_holds
                    
                    if can_hold and is_same_pitch:
                         # Extend current duration to cover next note
                         new_end = max(curr["time"] + curr["dur"], next_n["time"] + next_n["dur"])
                         curr["dur"] = max(0.1, new_end - curr["time"]) # Make it a hold if merged
                         curr["type"] = "hold"
                    else:
                         # Cannot hold (e.g. guitar/drums) OR Trill glitch
                         # Absorb the next note.
                         # If it was a trill glitch (<40ms), we just delete the second note.
                         # If it was a jack, we keep the first one.
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
        final_sorted = sorted(final, key=lambda x: x["time"])
        
        # --- STRICT LANE SAFETY CHECK ---
        # Ensure per-lane gap >= min_global_interval
        # (This catches any Quantizer artifacts or Logic slips)
        # Note: We allowed chords (diff lanes) but SAME lane must have gap.
        
        safe_final = []
        last_times_lane = {} # lane_idx -> last_note_end_time
        
        strict_gap = GENERATOR_CONFIG["sieving"]["min_global_interval"]
        
        for n in final_sorted:
            l = n["lane"]
            start = n["time"]
            prev_end = last_times_lane.get(l, -10.0)
            
            # Calculate gap from previous note IN THIS LANE
            # (Note: prev_end includes duration)
            if start < prev_end + strict_gap:
                # VIOLATION!
                # What to do? 
                # Option A: Delete.
                # Option B: Shift? No, grid is sacred.
                # Option C: Move Lane? Maybe.
                # For now: Delete to ensure playability.
                # print(f"  [Safety] Deleted note at {start:.3f}s (Lane {l}) - Gap Violation")
                continue
                
            safe_final.append(n)
            last_times_lane[l] = start + n["dur"]
            
        return safe_final

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
        "onset": 0.40, "frame": 0.30,
        "smoothing": 0.0,
        "multipass": False, # 'Other' is chaotic, consensus might fail
        "min_len": 58.0
    }
}


class StemSelector:
    @staticmethod
    def get_stem_energy(stem_name: str, manifest: dict) -> float:
        """Returns the peak energy or RMS from manifest."""
        if not manifest or stem_name not in manifest:
            return 0.0
        
        info = manifest[stem_name]
        # Prefer RMS (avg power) over Peak (transients) for "dominance"
        # manifest currently stores 'peak_energy'. Let's assume we might have 'rms' or use peak.
        # Original separator also keys 'vocals_lead' rms into a stats dict but manifest saves peak.
        # Fallback to peak if RMS missing.
        return info.get("peak_energy", 0.0)

    @staticmethod
    def select_layers(difficulty: str, manifest: dict, focus_mode: str = "main", stem_weights: dict = None) -> dict:
        """
        Determines Primary/Support stems based on Difficulty and Focus Mode.
        focus_mode: "main" (Vocals/Melody) or "alt" (Instruments/Rhythm)
        stem_weights: dict of {source: total_velocity} for dynamic selection.
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
        # Force specific modes for specific difficulties based on new user mapping:
        # EASY/NORMAL/HARD -> MAIN Focus
        # ALT_HARD -> ALT Focus (Alternative Hard/Complimentary)
        
        actual_mode = focus_mode
        if difficulty in ["EASY", "NORMAL", "HARD"]:
            actual_mode = "main"
        elif difficulty == "ALT_HARD":
            actual_mode = "alt"
            
        print(f"  [Layers] Difficulty: {difficulty} (Mode: {actual_mode})")

        if actual_mode == "main":
            # MAIN FOCUS: Vocals > Lead > Rhythm
            if has_vocals:
                # Prefer Lead
                if "vocals_lead" in active_stems: primary.append("vocals_lead")
                elif "vocals" in active_stems: primary.append("vocals")
                
                # In Main/Easy-Hard, Instruments are Backing Tracks.
                # Only add vocals/lead to primary. Support will handle rhythm.
                pass
            
            else:
                # Instrumental Song: Leads are Primary
                # Dynamic Rule: Primary = Most Dynamic Stem
                candidates = [s for s in active_stems if s not in ["drums", "bass"]]
                if candidates:
                    # Dynamic Sort using Energy (RMS/Peak) + Note Velocity Sum (stem_weights)
                    # stem_weights (velocity sum) serves as a proxy for "musical activity"
                    if stem_weights:
                        candidates.sort(key=lambda s: stem_weights.get(s, 0), reverse=True)
                        print(f"    > Instrumental Main Sort: {candidates}")
                        primary.append(candidates[0])
                    else:
                        # Fallback
                        if "guitar" in active_stems: primary.append("guitar")
                        elif "piano" in active_stems: primary.append("piano")
                        elif "other" in active_stems: primary.append("other")

            # Support: Rhythm (Drums)
            if difficulty != "EASY":
                 if "drums" in active_stems: support.append("drums")
            
        elif actual_mode == "alt":
            # ALT FOCUS: Instruments (2nd Busiest) > Rhythm
            # Dynamic Rule: Pick highest energy non-vocal instrument.
            # Order of preference: PURELY DYNAMIC.
            
            # Filter candidates: All melodic instruments (No Vocals, No Drums, No Bass)
            candidates = [s for s in active_stems if s not in ["vocals", "vocals_lead", "drums", "bass"]]
            
            if candidates:
                if stem_weights:
                    # Sort by weight desc (Total Note Velocity)
                    candidates.sort(key=lambda s: stem_weights.get(s, 0), reverse=True)
                    # print(f"    > Dynamic Sort: {candidates} (Scores: {[f'{stem_weights.get(s,0):.3f}' for s in candidates]})")
                    
                    selected_idx = 0
                    if not has_vocals and len(candidates) > 1:
                        # Instrumental Mode: Select 2nd best track for ALT diversity
                        selected_idx = 1
                        
                    primary.append(candidates[selected_idx])
                else:
                    # Fallback Priority
                    if "guitar" in candidates: primary.append("guitar")
                    elif "piano" in candidates: primary.append("piano")
                    elif "other" in candidates: primary.append("other")
            
            # If no melody instruments, Drums become primary (Drum Chart)
            if not primary and "drums" in active_stems:
                primary.append("drums")
            elif "drums" in active_stems:
                # Drums are strictly support in ALT mode (unless it's a drum chart)
                support.append("drums")
                    
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
        # Deepcopy events to prevent side effects (mutation) from affecting other difficulties
        events = copy.deepcopy(events)
        
        print(f"[Generator] Starting {difficulty} chart generation (Mode: {focus_mode})...")
        
        cfg = DIFF_CONFIGS.get(difficulty, DIFF_CONFIGS["NORMAL"])
        target_nps = cfg["nps"]
        n_lanes = cfg["lanes"]
        
        # --- STAGE 0: LAYER SELECTION ---
        # Calculate Stem Weights (Energy = Sum of Velocity)
        # This helps ALT mode pick the most dominant instrument dynamically.
        stem_weights = {}
        for e in events:
            s = getattr(e, "source", "other")
            v = getattr(e, "velocity", 0.5)
            stem_weights[s] = stem_weights.get(s, 0.0) + v
            
        layers = StemSelector.select_layers(difficulty, manifest, focus_mode, stem_weights)
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
        strict_stems = ["drums"]
        # Restricting drums to 1/4 and 1/8 only (Beat Feel)
        strict_grids = [4, 8]
        
        # Split events
        group_strict = []
        group_free = []
        group_unsnapped = []
        
        for e in clean_events:
            if e.source in strict_stems:
                group_strict.append(e)
            else:
                # Vocals, Piano, Guitar, Bass, Other -> ALL UNSNAPPED (High Fidelity)
                # We rely on DSP Grounding in Stage 0 for timing accuracy.
                group_unsnapped.append(e)
                
        print(f"DEBUG QUANTIZER: Strict(Drums)={len(group_strict)}, Unsnapped(All Else)={len(group_unsnapped)}")
            
        # Snap separately
        snapped_strict = self.quantizer.snap_to_grid(group_strict, grids=strict_grids)
        # snapped_free = self.quantizer.snap_to_grid(group_free, grids=allowed_grids) # IGNORED
        
        quantized_events = snapped_strict + group_unsnapped
        
        # --- STAGE 3: THE SIEVE (Scoring & Selection) ---
        ranked_events = self._rank_events_layered(quantized_events, primary_src, support_src)
        
        # Using New Adaptive Sieve (V300)
        final_events = self._filter_adaptive_sieve(ranked_events, difficulty)
        
        print(f"  [Sieve] Selected {len(final_events)} notes (Target NPS: {target_nps})")
        
        # --- STAGE 2.5: MACRO HOLDS (Visual Consolidation) ---
        # Consolidated events skipped for now to avoid confusion
        # consolidated_events = self._consolidate_visuals(final_events)
        consolidated_events = final_events
        
        # --- STAGE 4: PRE-MAPPING CLEANUP ---
        # 6. Sanitize Holds (Uniform Rule)
        sanitized_events = self._sanitize_holds(consolidated_events)
        
        # 7. Deduplicate Same-Pitch (Vibrato/Glitch Fix) - PRE-LANE ALLOCATION
        deduped_events = self._deduplicate_same_pitch(sanitized_events)
        
        # --- STAGE 5: THE MAPPER (Lane Allocation) ---
        chart_notes = self._allocate_lanes(deduped_events, n_lanes)
        
        return {
            "metadata": {
                "difficulty": difficulty, 
                "version": "V300", 
                "bpm": self.bpm,
                "focus": focus_mode  # Export Focus Mode for Visualizer
            },
            "notes": chart_notes
        }

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
        v_cfg = GENERATOR_CONFIG.get("volume", {})
        sf_cfg = GENERATOR_CONFIG.get("layer_sifting", {})
        
        beat_dur = 60.0 / self.bpm if self.bpm > 0 else 0.5
        
        for n in events:
            # Base Score = Velocity
            base_score = n.velocity
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
            if n.source in primary_src:
                base_score *= l_cfg["primary_multiplier"]
                is_valid = True
                
                # Active Layer Volume Boost
                n.velocity *= v_cfg.get("primary_boost", 1.2)
                
            elif n.source in support_src:
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
                
                # Only allow support notes on main beats (1/4, 1/8) UNLESS it's Percussion
                time_in_beats = n.time / beat_dur
                beat_fraction = time_in_beats % 1.0
                is_quarter = abs(beat_fraction) < 0.1 or abs(beat_fraction - 1.0) < 0.1
                is_eighth = abs(beat_fraction - 0.5) < 0.1
                
                is_percussive = n.source in sf_cfg.get("percussive_stems", [])
                
                if is_quarter:
                     base_score *= l_cfg["support_multiplier"] * l_cfg["support_on_beat_bonus"]
                     is_valid = True 
                elif is_eighth:
                     base_score *= l_cfg["support_multiplier"]
                     is_valid = True
                elif is_percussive:
                 # The Quantizer (Stage 2) has ALREADY snapped these to [4, 8, 16].
                 base_score *= 2.5 
                 is_valid = True
                else:
                     base_score = 0.0
                     is_valid = False
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
        New V300 "Game-Like" Sieve.
        1. Clusters notes by Quantized Grid Time.
        2. Enforces Polyphony Limit (Max simultaneous notes).
        3. Enforces Max NPS Limit via Dynamic Grid Adaptation (Decimation) instead of random cuts.
        4. Allows Bursts if budget permits.
        """
        if not events: return []
        
        # Difficulty Tuning
        diff_limits = {
            "EASY":   {"nps": 4.0, "poly": 1, "min_interval": 0.25},
            "NORMAL": {"nps": 6.0, "poly": 2, "min_interval": 0.16},
            "HARD":   {"nps": 12.0, "poly": 2, "min_interval": 0.08}, # 12.5 NPS = 0.08s
            "ALT_HARD": {"nps": 14.0, "poly": 2, "min_interval": 0.07}
        }
        cfg = diff_limits.get(difficulty, diff_limits["NORMAL"])
        
        target_nps = cfg["nps"]
        max_poly = cfg["poly"]
        base_min_interval = cfg["min_interval"]
        
        # Global burst limits
        burst_limit_dur = GENERATOR_CONFIG["sieving"]["burst_limit"] # 0.4s
        abs_min_interval = GENERATOR_CONFIG["sieving"]["min_global_interval"] # 0.05s
        
        # 1. Cluster by Time (Quantize slightly to group chords)
        # We assume Stage 2 (Quantizer) has already run, so notes should be grid-aligned.
        # But we group by very small epsilon to catch floating point drifts.
        events.sort(key=lambda x: x.time)
        clusters = []
        if not events: return []
        
        curr_t = events[0].time
        curr_notes = [events[0]]
        
        for i in range(1, len(events)):
            evt = events[i]
            if abs(evt.time - curr_t) < 0.005: # 5ms chord window
                curr_notes.append(evt)
            else:
                # Push previous cluster
                clusters.append({"time": curr_t, "notes": curr_notes})
                curr_t = evt.time
                curr_notes = [evt]
        clusters.append({"time": curr_t, "notes": curr_notes})
        
        # 2. Polyphony Pass (Strict)
        # If cluster > max_poly, keep top N highest velocity
        pruned_clusters = []
        for c in clusters:
            notes = c["notes"]
            if len(notes) > max_poly:
                notes.sort(key=lambda x: x.velocity, reverse=True)
                pruned_clusters.append({"time": c["time"], "notes": notes[:max_poly]})
            else:
                pruned_clusters.append(c)
                
        # 3. Dynamic Decimation Pass (The "Speed Limit")
        final_clusters = []
        last_time = -10.0
        
        # Burst State
        burst_start_time = -1.0
        in_burst = False
        
        # Sliding Window for NPS calculation (Token Bucket light version)
        # We check local density in 1.0s window? 
        # Actually user requested "Max possible note per second at any moment" -> Minimum Interval.
        # So we stick to Interval checks.
        
        i = 0
        while i < len(pruned_clusters):
            curr = pruned_clusters[i]
            t = curr["time"]
            
            dt = t - last_time
            
            # Check Interval
            required_int = base_min_interval
            
            # Burst Logic
            is_burst = False
            if dt < base_min_interval:
                # Potential Burst
                # Check if we can allow it
                # 1. Must be > abs_min_interval (Physical Limit)
                if dt >= abs_min_interval:
                    # 2. Check Burst Duration (Are we already bursting?)
                    if not in_burst:
                        in_burst = True
                        burst_start_time = last_time # Start of burst
                    
                    burst_dur = t - burst_start_time
                    if burst_dur < burst_limit_dur:
                        # ALLOW BURST
                        is_burst = True
                        required_int = abs_min_interval # Use tighter limit
                    else:
                        # BURST EXHAUSTED
                        is_burst = False
                else:
                    # Too fast even for burst
                    is_burst = False
            else:
                # Reset burst if we had a nice gap
                if dt > base_min_interval * 1.5:
                    in_burst = False
            
            if dt >= required_int:
                # Accept
                final_clusters.append(curr)
                last_time = t
                i += 1
            else:
                # REJECT via Dynamic Decimation
                # Instead of skipping 'curr', we try to find the NEXT event that satisfies the interval.
                # This effectively downsamples the rhythm (e.g. 1/16 -> 1/8).
                
                # Scan ahead until we find a note >= required_int from last_time
                found_next = False
                j = i + 1
                while j < len(pruned_clusters):
                    next_cand = pruned_clusters[j]
                    dist = next_cand["time"] - last_time
                    
                    # If we are decimating, we should aim for the 'base_min_interval' (Standard Speed)
                    # to restore stability, rather than maintaining the burst speed.
                    target_int = base_min_interval 
                    
                    if dist >= target_int:
                        # Found valid next step
                        # Skip everything between i and j
                        # Wait, we need to re-evaluate 'j' as the new candidate in the main loop
                        i = j
                        found_next = True
                        break
                    j += 1
                
                if not found_next:
                    # No more valid notes in song
                    break
                    
        # Flatten
        final_events = []
        for c in final_clusters:
            final_events.extend(c["notes"])
            
        print(f"  [Sieve] Adaptive: {len(events)} -> {len(final_events)} notes (Diff: {difficulty})")
        return final_events

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
        
        
        # Pre-process: Sanitize Holds (Global Reaction Time Check)
        events = self._sanitize_holds(events)
        
        for ev in events:
            # GHOST NOTE HANDLING (Audio Only, No Visuals)
            if getattr(ev, "ghost", False):
                processed.append({
                    "time": float(ev.time),
                    "lane": -1, # HIDDEN
                    "dur": float(ev.duration), # Keep duration for audio synth bucket?
                    "type": "ghost",
                    "midi": int(ev.pitch),
                    "vol": float(getattr(ev, "velocity", 0.8)),
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

    def _sanitize_holds(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Force short holds to taps.
        Force tap if gap to next note is too small (Reaction Time).
        """
        if not events: return []
        events.sort(key=lambda x: x.time)
        
        min_reaction = GENERATOR_CONFIG["holds"]["min_reaction_time"] # 0.1s
        min_dur = GENERATOR_CONFIG["holds"]["min_duration"] # 0.2s
        
        for i in range(len(events)):
            curr = events[i]
            
            # 1. Min Duration Check
            if curr.duration < min_dur:
                curr.duration = 0.0 # Force Tap
                continue
                
            # 2. Gap Check (Lookahead)
            # Find next note (ignoring chords at same time)
            j = i + 1
            while j < len(events):
                next_n = events[j]
                if next_n.time > curr.time + 0.01:
                    # Found next sequential note
                    gap = next_n.time - (curr.time + curr.duration)
                    if gap < min_reaction:
                        # Too fast! Force Tap.
                        # print(f"  [Hold] Forced Tap at {curr.time:.2f}s (Gap {gap:.3f}s < {min_reaction}s)")
                        curr.duration = 0.0
                    break
                j += 1
                
        return events

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
            
            # Pass 1: Merge close notes (Minijacks)
            curr = stack[0]
            for i in range(1, len(stack)):
                next_n = stack[i]
                
                # Check proximity
                gap = next_n["time"] - curr["time"]
                
                # Logic: Only merge if strictly a "Jack" (Same Pitch + Close Time)
                # If pitch is different, it's a trill/stream -> Allow it (unless very fast glitch)
                is_same_pitch =  abs(next_n["midi"] - curr["midi"]) < 1.0
                
                # Thresholds
                # Same Pitch: 100ms (standard jack removal)
                # Different Pitch: 40ms (glitch removal, very fast trills allowed > 40ms)
                thresh = minijack_thresh if is_same_pitch else 0.04
                
                if gap < thresh:
                    # Merge!
                    # Only create hold if allowed
                    can_hold = curr["source"] in allowed_holds
                    
                    if can_hold and is_same_pitch:
                         # Extend current duration to cover next note
                         new_end = max(curr["time"] + curr["dur"], next_n["time"] + next_n["dur"])
                         curr["dur"] = max(0.1, new_end - curr["time"]) # Make it a hold if merged
                         curr["type"] = "hold"
                    else:
                         # Cannot hold (e.g. guitar/drums) OR Trill glitch
                         # Absorb the next note.
                         # If it was a trill glitch (<40ms), we just delete the second note.
                         # If it was a jack, we keep the first one.
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
        final_sorted = sorted(final, key=lambda x: x["time"])
        
        # --- STRICT LANE SAFETY CHECK ---
        # Ensure per-lane gap >= min_global_interval
        # (This catches any Quantizer artifacts or Logic slips)
        # Note: We allowed chords (diff lanes) but SAME lane must have gap.
        
        safe_final = []
        last_times_lane = {} # lane_idx -> last_note_end_time
        
        strict_gap = GENERATOR_CONFIG["sieving"]["min_global_interval"]
        
        for n in final_sorted:
            l = n["lane"]
            start = n["time"]
            prev_end = last_times_lane.get(l, -10.0)
            
            # Calculate gap from previous note IN THIS LANE
            # (Note: prev_end includes duration)
            if start < prev_end + strict_gap:
                # VIOLATION!
                # What to do? 
                # Option A: Delete.
                # Option B: Shift? No, grid is sacred.
                # Option C: Move Lane? Maybe.
                # For now: Delete to ensure playability.
                # print(f"  [Safety] Deleted note at {start:.3f}s (Lane {l}) - Gap Violation")
                continue
                
            safe_final.append(n)
            last_times_lane[l] = start + n["dur"]
            
        return safe_final


    def _deduplicate_same_pitch(self, events: List[NoteEvent]) -> List[NoteEvent]:
        """
        Merges or removes sequential notes of the same pitch that are too close (Vibrato/Detection Errors).
        Must run BEFORE lane allocation to ensure they are seen as conflicts.
        """
        if not events: return []
        
        events.sort(key=lambda x: x.time)
        cleaned = []
        last_note_by_pitch = {} # {midi_int: NoteEvent}
        
        # Config
        minijack_thresh = 0.18 # Melodic Vibrato Threshold
        percussive_stems = GENERATOR_CONFIG["layer_sifting"]["percussive_stems"]
        
        for n in events:
            # Determine threshold
            is_percussive = n.source in percussive_stems
            thresh = 0.04 if is_percussive else minijack_thresh
            
            p_key = int(round(n.pitch))
            
            if p_key in last_note_by_pitch:
                last_n = last_note_by_pitch[p_key]
                dt = n.time - last_n.time
                
                if dt < thresh:
                    # CONFLICT: Too close!
                    # Logic: If 'last_n' was a hold, maybe extend it?
                    # For now: Just SKIP 'n' (Swallow the vibrato tail)
                    continue
            
            # Accepted
            cleaned.append(n)
            last_note_by_pitch[p_key] = n
            
        return cleaned

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
            y, sr = AudioCache.get(audio_path, sr=22050)
            
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
            return events
            
        except Exception as e:
            print(f"[Timing] DSP Grounding failed: {e}")
            return events
            
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
    def __init__(self, stems_folder: str):
        self.stems_folder = stems_folder
        self.base_name = os.path.basename(os.path.dirname(stems_folder)) if os.path.basename(stems_folder) in ["stems", "beatmap"] else os.path.basename(stems_folder)
        # Actually stems_folder is usually ".../stems/songname"
        if os.path.dirname(stems_folder).endswith("stems"):
             self.base_name = os.path.basename(stems_folder)
        
        self._ensure_paths()
        
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
        
        # If BasicPitch is -8ms (Early), we typically don't need to fix it much, 
        # OR the user perception of "lag" is actually Pygame visual lag.
        # But if the user says "severe swings", we should trust the jitter.
        
        self.latency_offset = -0.01 # Shift -10ms (conservative correction)
        # Detailed in benchmark_report.md
        print(f"[RhythmEngine] Latency Compensation: {self.latency_offset*1000:.1f}ms")
        
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
                if hasattr(self, "latency_offset") and self.latency_offset != 0:
                     for n in notes: n.time += self.latency_offset
                
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
                    # Standard Monophonic
                    v_notes = self.council.transcribe(v_path, model_type="fcpe")

                # Vocal Smoothing (Tunable)
                from transcribe.smoother import VocalSmoother
                print(f"[Generator] Applying Vocal Smoothing (Level=0.7)...")
                v_notes = VocalSmoother.smooth(v_notes, level=0.7)

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
        output_dir = os.path.join(self.stems_folder, "beatmap")
        os.makedirs(output_dir, exist_ok=True)
        
        for diff in ["EASY", "NORMAL", "HARD", "ALT_HARD"]:
            chart_data = self.generator.generate(list(all_events), diff, manifest=self.manifest) # Pass copy & manifest
            
            out_file = os.path.join(output_dir, f"{diff}.json")
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
                
                # Check distance from current 'accepted' note start?
                # No, we need to compare against the *last retained* note, not just the previous candidate.
                # Wait, the logic below compares 'curr' vs 'next_e' in a chain.
                # If we accepted 'curr', we then check if 'next_e' is too close to 'curr'.
                
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rhythm Engine V300")
    parser.add_argument("folder", help="Path to the folder containing separated stems")
    args = parser.parse_args()

    engine = RhythmEngine(args.folder)
    engine.run()
