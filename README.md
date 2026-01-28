# Rhythm Engine with Pitch Estimation (WIP)

Automated beatmap generation pipeline that transforms audio files into playable rhythm game charts with MIDI pitch estimation (to produce SFX that act as stylisied audio compressor).

A Vibe Coding Project. 

Intended to be used with My [Godot Map Editor(WIP)](https://github.com/August13742/RhythmBeatmapEditor)

## Demo (Sound ON)
### Case1: Monophonic Vocal

https://github.com/user-attachments/assets/a5a1b199-885d-40a2-9d4a-e9a5909e300d

(SongName: [Post-Script](https://www.youtube.com/watch?v=sHCKVN_mKI0))

### Case2: Piano

https://github.com/user-attachments/assets/567c44ec-5bfc-49a6-909d-b6f960f65543

(SongName: [Deemo Goodbye](https://www.youtube.com/watch?v=cj79VsY7P_E))

### Case3: Jazz

https://github.com/user-attachments/assets/dd9fed8c-2d21-4e14-9665-49bb2c77c87c

(SongName:[Lioness’ Pride （Twilight ver.）](https://www.youtube.com/watch?v=JSTl0AhSZNk))

### Case4: Polyphonic Vocals (where pitch estimation pipeline fails the most)

https://github.com/user-attachments/assets/4ef1b20c-3140-4cf2-91d2-3b9be06b0a9f

↑(SongName: [With Glory I Shall Fall](https://www.youtube.com/watch?v=4ZKgq7Aw34s))


https://github.com/user-attachments/assets/74f6667f-5349-4b1a-9a65-aa9953cf9166

↑(SongName: [ベテルギウス](https://www.youtube.com/watch?v=f2QIaERaKDE))


https://github.com/user-attachments/assets/2b29cb72-2963-47ae-ad3f-3f54164ae4fb

↑(SongName: [Unfinished Journey](https://www.youtube.com/watch?v=wiXzbkjuJao))


https://github.com/user-attachments/assets/e33d41c9-94b6-4eab-a939-5dec20c24116

↑(SongName: [Nameless Martyr](https://www.youtube.com/watch?v=AuMcCiwNazo))


## Quick Start

```bash
uv sync
uv run python main.py path/to/song.mp3
```

**Options:**
- `--skip-separation` - Skip stem separation if already processed (uses 2 Rofomers, Basic Pitch, MossFormer2, very heavy)
- `--rebake` - Force regenerate note events from stems (Uses 3 DL models + multi-passes, very heavy)
- `--rechart` - Regenerate beatmaps using DSP and logic. Relatively light.
- `--generate-only` - Generate maps without launching visualizer

## Pipeline Overview

1. **Audio Separation** - Splits audio into 7 stems (vocals, drums, bass, guitar, piano, other, vocals_lead)
2. **Transcription** - Converts each stem to note events using ML pitch detection
3. **Chart Generation** - Transforms notes into playable beatmaps with lane allocation

Output: `Beatmaps/{song_name}/EASY.json`, `NORMAL.json`, `HARD.json`, `ALT_HARD.json`

## Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| BS-Roformer | 6-stem audio separation | [model_bs_roformer_ep_317_sdr_12.9755](https://pypi.org/project/audio-separator/) |
| Mel-Roformer-Viperx | Lead/backing vocal split | [model_mel_band_roformer_ep_3005_sdr_11.4360](https://pypi.org/project/audio-separator/) |
| FCPE | Monophonic vocal pitch | [torchfcpe](https://pypi.org/project/torchfcpe/) |
| RMVPE | Robust vocal pitch (fallback) | Custom Setup + [VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) |
| BasicPitch | Polyphonic transcription | [basic-pitch-torch](https://github.com/spotify/basic-pitch) | 
| Mossformer2 | Vocal Cleaning | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## Testing Dataset Used:
[東北きりたん歌唱データベース](https://github.com/mmorise/kiritan_singing)
[SMD MIDI-Audio Piano Music](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi)

## Project Structure

```
rhythm_engine/
├── main.py              # Entry point
├── separator.py         # Audio stem separation
├── engine.py            # Core transcription logic
├── chart_generator.py   # Beatmap generation
├── transcribe/
│   ├── council.py       # Multi-model vocal transcription
│   ├── basic_pitch.py   # BasicPitch wrapper
│   ├── segmenter.py     # F0-to-note conversion
│   └── smoother.py      # Note cleanup
└── benchmarks/          # Accuracy evaluation scripts
```

## Dependencies

see `pyproject.toml`

## Difficulty System

| Difficulty | NPS | Polyphony | Focus |
|------------|-----|-----------|-------|
| EASY | 2.5 | 1 | Vocals only |
| NORMAL | 4.0 | 2 | Vocals + drums |
| HARD | 6.0 | 2 | Vocals + drums |
| ALT_HARD | 6.0 | 2 | Lead instrument |
