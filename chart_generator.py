import copy
from typing import List
from beatmap import NoteEvent, EventFilter, Quantizer

# Configuration for Visualizer compatibility & Generator Logic
DIFF_CONFIGS = {
    "EASY":   {"lanes": 4, "nps": 2.5},
    "NORMAL": {"lanes": 4, "nps": 4.5},
    "HARD":   {"lanes": 4, "nps": 6.0},
    "ALT_HARD": {"lanes": 4, "nps": 6.0}
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
            "other": 0.6  # Lower weight for noisy composite track
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
        "percussive_stems": ["drums", "bass"] # Stems EXEMPT from sifting
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
                    # print(f"  [VocalCorrector] Fixed Octave Jump at {curr.time:.2f}s ({curr.pitch} -> {curr.pitch + shift})")
                    
                    curr.pitch += shift
            
            cleaned_vocals.append(curr)
            
        return others + cleaned_vocals

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
        
        # --- STAGE 2: ADAPTIVE GRID (Smart Snapping) ---
        # Define Grids per Difficulty
        # 4=Quarter, 8=Eighth, 12=Triplet Eighth, 16=Sixteenth, 24=Triplet Sixteenth, 32=32nd, 48=Triplet 32nd
        # 48 is high fidelity enough to sound "unsnapped" but is mathematically snapped.
        
        # Strict (Drums/Bass/Backing)
        strict_grids = [4, 8]
        
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
        
        # 6.5 Vocal Correction (Octave Sieve)
        # Apply AFTER sanitization logic but BEFORE Dedupe/Lanes
        corrected_events = VocalCorrector.apply(sanitized_events, difficulty)
        
        # 7. Deduplicate Same-Pitch (Vibrato/Glitch Fix) - PRE-LANE ALLOCATION
        deduped_events = self._deduplicate_same_pitch(corrected_events)
        
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
            "ALT_HARD": {"nps": 10.0, "poly": 2, "min_interval": 0.09}  # Reduced density for playability
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
            # 2. Must exceed threshold (vocal-specific for sung notes)
            is_hold = False
            if ev.source in allowed_holds:
                # Use lower threshold for vocals (sustained singing)
                thresh = vocal_hold_thresh if ev.source in ["vocals", "vocals_lead"] else hold_thresh
                if ev.duration > thresh:
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
        Ensures holds don't overlap with next note (reaction time).
        Does NOT enforce minimum duration - that's done in _allocate_lanes
        which has source-specific thresholds (vocals get lower threshold).
        
        Logic: If a hold would end too close to the next note, trim it.
        """
        if not events: return []
        events.sort(key=lambda x: x.time)
        
        min_reaction = GENERATOR_CONFIG["holds"]["min_reaction_time"]  # 0.1s
        abs_min_tail = 0.08  # Below this, just make it a tap
        
        for i in range(len(events)):
            curr = events[i]
            
            # Skip notes that are already short (will be taps anyway)
            if curr.duration < abs_min_tail:
                continue
                
            # Gap Check (Lookahead) - find next sequential note
            j = i + 1
            while j < len(events):
                next_n = events[j]
                if next_n.time > curr.time + 0.01:
                    # Found next sequential note
                    curr_end = curr.time + curr.duration
                    gap = next_n.time - curr_end
                    
                    if gap < min_reaction:
                        # Hold ends too close to next note - trim it
                        # New end should be: next_n.time - min_reaction
                        new_end = next_n.time - min_reaction
                        new_dur = new_end - curr.time
                        
                        if new_dur >= abs_min_tail:
                            curr.duration = new_dur
                        else:
                            # Would be too short - keep original and let it become a tap
                            # (don't zero it here, _allocate_lanes will handle threshold)
                            pass
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
