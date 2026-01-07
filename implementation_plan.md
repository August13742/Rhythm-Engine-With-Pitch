# Rhythm Engine V300: Transcription-First Architecture

## Goal
Transition the Rhythm Engine from a "Transient-First" (DSP-based) pipeline to a "Transcription-First" (ML-based) pipeline to improve beatmap musicality and accuracy.

## User Constraints
- **Hardware:** Laptop RTX 4060 (Limited VRAM).
- **Execution:** Do NOT run the full pipeline during this session.
- **Dependencies:** Avoid complex dependency/environment blending. Use ONNX/PyTorch.

## Architecture Changes

### 1. New "Ear" Module (`transcriber.py`)
Responsible for converting Audio -> Symbolic Events (MIDI-like).
- **Instruments:** Use **Basic Pitch (ONNX)**.
    - Input: `stems/{instrument}.wav`
    - Output: List of Events (Start, End, Pitch, Velocity).
    - *Why ONNX?* Lightweight, no TensorFlow dependency, runs on `onnxruntime-gpu`.
- **Vocals:** Upgraded **Council**.
    - Input: `stems/vocals.wav`
    - Output: Continuous F0 curve -> Segmented Notes.
    - *Change:* Instead of firing only on peaks, we will run the models (FCPE/Crepe) on the whole buffer (chunked) and then use algorithms to segment the pitch curve into notes.

### 2. New "Brain" Module (`generator.py`)
Responsible for converting Symbolic Events -> Game Objects.
- **Quantization:** Snap events to grid (1/16, 1/32) *before* processing.
- **Filtering:** Select notes based on:
    - **Velocity:** High energy = Note.
    - **Rhythm:** Downbeats > Offbeats.
    - **Pitch Change:** Melodic contour changes are important.
- **Lane Allocation:**
    - Use pitch-based binning (prettier) but with "flow" considerations (no jumps that are physically impossible).

## Proposed File Structure
```
rhythm_engine/
├── Legacy_V208/          # Archived old pipeline
│   ├── generator.py
│   └── ...
├── models/               # Model weights
├── transcribe/
│   ├── basic_pitch.py    # ONNX wrapper for Basic Pitch
│   ├── council.py        # Vocal F0 extraction
│   └── segmenter.py      # F0 -> Note logic
├── generator.py          # Main entry point (New)
├── beatmap.py            # Data structures (Note, Lane, etc.)
└── requirements.txt      # Updated deps
```

## Step-by-Step Implementation

1.  **Archive:** Move current `generator.py` and related experimental scripts to `Legacy_V208`.
2.  **Scaffold:** Create empty class structures for `TransformationEngine` (the new Generator).
3.  **Implement Basic Pitch:** Write the ONNX inference code (without running).
4.  **Implement Council:** Refactor `_run_fcpe` / `_run_crepe` into standalone functions that return full time-series arrays.
5.  **Refactor Logic:** Rewrite the "Harvesting" phase to consume the new Event lists.

## Verification Plan (Static)
- Check import validity.
- Verify class structures and method signatures.
- Ensure dependency list is clean.
- **NO RUNTIME TESTING** as requested.
