# """
# Visualize raw generator analysis data for wgf vocals.
# Shows: Energy, Pitch (Hz), Voiced Probability, Onset Detection, MIDI notes.
# """

# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from scipy import ndimage

# # ==========================================
# # CONFIG (from generator.py)
# # ==========================================
# SR = 44100
# HOP_LENGTH = 512
# PYIN_FRAME = 4096
# PYIN_FMIN = 40
# PYIN_FMAX = 1200

# # Time range: 1:40 ~ 2:00 (100 to 120 seconds)
# START_TIME = 110.0  # seconds
# END_TIME = 130.0    # seconds

# # Calculate time stamps for display
# def seconds_to_mmss(seconds):
#     minutes = int(seconds) // 60
#     secs = int(seconds) % 60
#     return f"{minutes}:{secs:02d}"

# start_mmss = seconds_to_mmss(START_TIME)
# end_mmss = seconds_to_mmss(END_TIME)

# # Load audio
# audio_path = "stems/wgf/vocals.wav"
# print(f"Loading {audio_path}...")
# y, sr = librosa.load(audio_path, sr=SR, mono=True)

# # Convert time range to samples
# start_sample = int(START_TIME * SR)
# end_sample = int(END_TIME * SR)
# y_slice = y[start_sample:end_sample]

# print(f"Audio shape: {y_slice.shape}, Duration: {len(y_slice) / SR:.2f}s")

# # ==========================================
# # 1. ENERGY ENVELOPE
# # ==========================================
# print("Computing energy envelope...")
# S = librosa.feature.melspectrogram(y=y_slice, sr=sr, hop_length=HOP_LENGTH)
# energy = np.sqrt(np.sum(S**2, axis=0))
# energy = ndimage.gaussian_filter1d(energy, sigma=2)  # Smooth it
# energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)  # Normalize
# n_energy_frames = len(energy)

# # ==========================================
# # 2. pYIN PITCH EXTRACTION
# # ==========================================
# print("Running pYIN pitch detection...")
# f0, _, voiced_prob = librosa.pyin(
#     y_slice, 
#     fmin=PYIN_FMIN, 
#     fmax=PYIN_FMAX, 
#     sr=sr, 
#     frame_length=PYIN_FRAME,
#     hop_length=HOP_LENGTH,
#     fill_na=np.nan
# )

# n_pyin_frames = len(f0)
# print(f"Energy frames: {n_energy_frames}, pYIN frames: {n_pyin_frames}")

# # Align lengths - pad to match the longer array
# max_frames = max(n_energy_frames, n_pyin_frames)
# if n_energy_frames < max_frames:
#     energy = np.pad(energy, (0, max_frames - n_energy_frames), mode='edge')
# if n_pyin_frames < max_frames:
#     f0 = np.pad(f0, (0, max_frames - n_pyin_frames), mode='constant', constant_values=np.nan)
#     voiced_prob = np.pad(voiced_prob, (0, max_frames - n_pyin_frames), mode='edge')

# # Frames to time using consistent hop_length
# frames = np.arange(len(f0))
# frame_times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)

# print(f"f0 shape: {f0.shape}, voiced_prob shape: {voiced_prob.shape}")
# print(f"f0 range: {np.nanmin(f0):.1f} - {np.nanmax(f0):.1f} Hz")
# print(f"voiced_prob range: {np.nanmin(voiced_prob):.2f} - {np.nanmax(voiced_prob):.2f}")

# # ==========================================
# # 3. CONVERT TO MIDI
# # ==========================================
# midi_notes = librosa.hz_to_midi(f0)

# # ==========================================
# # 4. ONSET DETECTION (Peak picking on energy)
# # ==========================================
# print("Detecting onsets...")
# sensitivity = 0.05  # From generator harvest config
# onset_frames = librosa.util.peak_pick(
#     energy, 
#     pre_max=3, post_max=3, 
#     pre_avg=3, post_avg=3, 
#     delta=sensitivity, wait=5
# )
# onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

# print(f"Detected {len(onset_frames)} onsets")

# # ==========================================
# # 5. VOICING GATE ANALYSIS
# # ==========================================
# vocal_gate = 0.30
# valid_mask = voiced_prob > vocal_gate
# print(f"Frames above vocal gate ({vocal_gate}): {np.sum(valid_mask)} / {len(voiced_prob)}")

# # ==========================================
# # 6. PLOT
# # ==========================================
# fig = plt.figure(figsize=(18, 12))
# gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.4)

# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# ax3 = fig.add_subplot(gs[2])
# ax4 = fig.add_subplot(gs[3])
# ax5 = fig.add_subplot(gs[4])

# # ===== PLOT 1: WAVEFORM =====
# samples = np.arange(len(y_slice)) / sr
# ax1.plot(samples, y_slice, color='cyan', linewidth=0.5, alpha=0.8)
# ax1.set_ylabel('Amplitude', fontsize=10)
# ax1.set_title(f'Waveform ({start_mmss} ~ {end_mmss})', fontsize=12, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.set_xlim(0, len(y_slice) / sr)

# # ===== PLOT 2: ENERGY ENVELOPE =====
# ax2.plot(frame_times, energy, color='green', linewidth=1.5, label='Energy')
# ax2.fill_between(frame_times, energy, alpha=0.3, color='green')
# ax2.scatter(onset_times, energy[onset_frames], color='red', s=100, marker='x', 
#             label=f'Onsets ({len(onset_frames)})', zorder=5)
# ax2.axhline(y=sensitivity, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Peak delta threshold ({sensitivity})')
# ax2.set_ylabel('Energy (norm)', fontsize=10)
# ax2.set_title('Energy Envelope & Onset Detection', fontsize=12, fontweight='bold')
# ax2.legend(fontsize=9)
# ax2.grid(True, alpha=0.3)
# ax2.set_xlim(0, len(y_slice) / sr)

# # ===== PLOT 3: PITCH (Hz) =====
# valid_f0 = f0.copy()
# valid_f0[~np.isfinite(f0)] = np.nan
# ax3.plot(frame_times, valid_f0, color='blue', linewidth=1, marker='o', markersize=2, label='Pitch (pYIN)')
# ax3.set_ylabel('Frequency (Hz)', fontsize=10)
# ax3.set_title('Pitch Detection (pYIN)', fontsize=12, fontweight='bold')
# ax3.set_ylim(PYIN_FMIN - 50, PYIN_FMAX + 50)
# ax3.legend(fontsize=9)
# ax3.grid(True, alpha=0.3)
# ax3.set_xlim(0, len(y_slice) / sr)

# # ===== PLOT 4: VOICED PROBABILITY & GATE =====
# ax4.plot(frame_times, voiced_prob, color='purple', linewidth=1.5, label='Voiced Probability')
# ax4.fill_between(frame_times, voiced_prob, alpha=0.3, color='purple')
# ax4.axhline(y=vocal_gate, color='red', linestyle='--', linewidth=2, label=f'Vocal Gate ({vocal_gate})')
# ax4.fill_between(frame_times, 0, 1, where=(voiced_prob > vocal_gate), 
#                   alpha=0.2, color='green', label='Valid (above gate)')
# ax4.set_ylabel('Probability', fontsize=10)
# ax4.set_ylim(0, 1.05)
# ax4.set_title(f'Voicing Probability (Valid frames: {np.sum(valid_mask)}/{len(voiced_prob)})', 
#               fontsize=12, fontweight='bold')
# ax4.legend(fontsize=9)
# ax4.grid(True, alpha=0.3)
# ax4.set_xlim(0, len(y_slice) / sr)

# # ===== PLOT 5: MIDI NOTES =====
# valid_midi = midi_notes.copy()
# valid_midi[~np.isfinite(midi_notes)] = np.nan
# ax5.plot(frame_times, valid_midi, color='orange', linewidth=1.5, marker='o', markersize=3, label='MIDI Notes')
# ax5.scatter(frame_times[valid_mask], valid_midi[valid_mask], color='green', s=30, alpha=0.6, 
#             label=f'Valid (above gate) {np.sum(valid_mask)}', zorder=5)
# ax5.set_xlabel('Time (seconds)', fontsize=10)
# ax5.set_ylabel('MIDI Note', fontsize=10)
# ax5.set_title('MIDI Note Contour', fontsize=12, fontweight='bold')
# ax5.set_ylim(20, 100)
# ax5.legend(fontsize=9)
# ax5.grid(True, alpha=0.3)
# ax5.set_xlim(0, len(y_slice) / sr)

# # Add legend with time labels
# fig.suptitle(f'WGF Vocals Analysis: {start_mmss} ~ {end_mmss} ({START_TIME:.0f}s ~ {END_TIME:.0f}s)', 
#              fontsize=14, fontweight='bold', y=0.995)

# plt.tight_layout()
# output_path = f"wgf_vocals_analysis_{START_TIME:.0f}-{END_TIME:.0f}s.png"
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# print(f"\nVisualization saved to: {output_path}")
# plt.show()

# # ==========================================
# # SUMMARY STATISTICS
# # ==========================================
# print("\n" + "="*60)
# print("ANALYSIS SUMMARY")
# print("="*60)
# print(f"Time range: {START_TIME:.1f}s ~ {END_TIME:.1f}s ({END_TIME - START_TIME:.1f}s duration)")
# print(f"Energy: min={np.min(energy):.3f}, max={np.max(energy):.3f}, mean={np.mean(energy):.3f}")
# print(f"Pitch: min={np.nanmin(f0):.1f}Hz, max={np.nanmax(f0):.1f}Hz, median={np.nanmedian(f0):.1f}Hz")
# print(f"Voiced frames: {np.sum(voiced_prob > vocal_gate)}/{len(voiced_prob)} ({100*np.sum(voiced_prob > vocal_gate)/len(voiced_prob):.1f}%)")
# print(f"Voiced prob: min={np.min(voiced_prob):.3f}, max={np.max(voiced_prob):.3f}, mean={np.mean(voiced_prob):.3f}")
# print(f"Detected onsets: {len(onset_frames)}")
# print(f"Average onset spacing: {(END_TIME - START_TIME) / max(1, len(onset_frames)):.2f}s between onsets")

# # Show frames with highest voiced probability
# top_indices = np.argsort(voiced_prob)[-5:][::-1]
# print("\nTop 5 frames by voiced probability:")
# for i, idx in enumerate(top_indices, 1):
#     print(f"  {i}. Frame {idx} @ {frame_times[idx]:.2f}s: "
#           f"voiced_prob={voiced_prob[idx]:.3f}, f0={f0[idx]:.1f}Hz, midi={midi_notes[idx]:.1f}")

# print("="*60)
