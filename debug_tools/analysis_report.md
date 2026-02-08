# XY Problem Analysis: Rhythm Engine Pipeline

## The Core Conflict
**User's Request (X):** "Swap pYIN for DL models? Fix dependencies? Improve note picking? Use recursive environments?"
**Real Problem (Y):** "The beatmaps are 'bad', 'not fun', and feel like a 'bad rhythm game beatmap generator'. Instrument notes are messed up."

## 1. Diagnosis of the Current Pipeline

The current pipeline operates on a **"Transient-First, Sample-Later"** logic:
1.  **Onset Detection:** Uses `librosa.util.peak_pick` on an amplitude envelope.
2.  **Pitch Sampling:** At that *exact instant*, asks a model/algorithm (pYIN/Council) "What is the pitch right now?"
3.  **Map Generation:** Places a note there.

### Why this Fails (The "Fun" Gap)
*   **Context Blindness:** Music flows. A guitar strum lasts 2 seconds; a piano chord decays. Sampling a single millisecond at the attack ignores the sustain, the harmonic content, and the actual "musicality" of the note.
*   **Monophonic Bottleneck (pYIN):** pYIN is a *fundamental frequency (F0)* estimator. It assumes **one** note is playing. 
    *   *Reality:* Piano and Guitar stems are almost always polyphonic (chords). pYIN will panic and pick a random harmonic or the root, resulting in "jittery" or "wrong" notes.
*   **Amplitude != Importance:** `peak_pick` only sees volume. Ghost notes on a snare, or a quiet melody line under a crash cymbal, are lost. Conversely, a loud noise becomes a note.
*   **The "Council" Limitation:** While the ensemble (FCPE/Crepe/RMVPE) is powerful, applying it only at *detected onsets* cripples it. If the onset detector misses a legato transition (a note changing pitch without re-articulating), the Council never gets convened.

## 2. The Dependency Trap (XY)
You are worried about `numpy < 2.0` and blending TensorFlow with PyTorch.
*   **Reality:** This is an implementation detail, not a pipeline blocker.
*   **Optimization:** **Basic Pitch** (Spotify's model) is the gold standard for instrument transcription right now. There are **PyTorch ports** of Basic Pitch that eliminate the need for TensorFlow entirely.
*   **Recommendation:** Do not build a "recursive uv" or complex subprocess architecture. Use a pure PyTorch stack. This resolves the dependency hell and the "blending" fears.

## 3. The Proposed "Transcription-First" Architecture

To move from "Bad Audio Compressor" to "Rhythm Game", we must decouple **Transcription** (What is heard?) from **Chart Generation** (What is played?).

### Phase A: Deep Transcription (The "Ear")
Instead of picking peaks, run full-context models on the entire stem to generate a **symbolic representation** (MIDI).

1.  **Instruments (Polyphonic):** 
    *   **Swap pYIN for Basic Pitch (Torch version).**
    *   *Why:* It handles chords (polyphony), detects duration, and ignores noise better than DSP methods.
2.  **Vocals (Monophonic/Melodic):**
    *   **Keep The Council (FCPE/Crepe/RMVPE).**
    *   *Change:* Run it on *frames* (continuous f0 curve), not just onsets. Then use a **Note Segmentation** algorithm (e.g., Hidden Markov Model or simple heuristic) to convert the curve into discrete notes.

### Phase B: The "Charter" (The "Game Design")
Now you have a high-quality MIDI file (Piano Roll) with too many notes. The "Game" is in the **Reductive Filtering**.

*   **Grid Quantization:** Snap MIDI start times to 1/16th or 1/32nd grid *before* selection.
*   **Importance Scoring:**
    *   Score = Velocity * RhythmWeight * PitchSalience.
    *   *RhythmWeight:* Is it on a downbeat (1.0)? Or an off-beat (0.5)?
*   **Selection:** Select top N notes per section to match target density (Difficulty).

## 4. Specific Responses to Your Queries

> "Do we swap pYIN from instruments for the DL models?"
**YES.** pYIN is mathematically unsuited for polyphonic stems (Piano/Guitar). Use Basic Pitch.

> "And the onsets?"
**ABANDON 'ONSET DETECTION'.** Use **Note Transcription**. Basic Pitch outputs *events* (Start, End, Pitch). Trust the model's internal onset detection, which acts on spectral features, not just amplitude peaks.

> "Recursive uv / Dependency blending?"
**NO.** It is over-engineering. Use `torch-basic-pitch` (or similar) to keep a single, clean Python environment. Confining legacy code to containers is a last resort; we have source access and "unlimited compute", so let's run the best models natively.

> "The Council only works for Vocal stem... messed up instrument note picking."
**The Council is fine for vocals.** The issue is applying "Vocal monophonic logic" to "Instrument polyphonic data". Instruments need their own specialized polyphonic model (Basic Pitch).

## 5. Strategic Plan

1.  **Refactor Dependencies:** Move to a pure PyTorch environment if possible. Check `audio-separator` compatibility or isolate it.
2.  **Implement Basic Pitch:** Create a new module `transcribe.py` that takes a stem and outputs a List[Note].
3.  **Upgrade The Council:** Modify it to output a continuous pitch curve, then segment it into notes, rather than waiting for `peak_pick`.
4.  **Rewrite Generator:** 
    *   Old: `Load Audio -> Find Peaks -> Check Pitch -> Write Note`.
    *   New: `Load Audio -> Transcribe to MIDI (Basic Pitch/Council) -> Filter & Select Notes -> Write Beatmap`.

This shifts the pipeline from **Signal Processing** (Oscilloscope style) to **Music Information Retrieval** (Musician style).
