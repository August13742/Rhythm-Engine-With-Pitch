"""
Standalone RMVPE Test Script
Tests the RMVPE pitch detection model in isolation.
"""

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from rmvpe_model import RMVPE_Infer


# ==========================================
#      TEST PIPELINE
# ==========================================

def f0_to_midi(f0_hz):
    """Convert F0 in Hz to MIDI note numbers."""
    f0_hz = np.where(f0_hz > 0, f0_hz, np.nan)
    midi = 69 + 12 * np.log2(f0_hz / 440.0)
    return midi


def test_rmvpe(input_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== RMVPE Standalone Test ({device.upper()}) ===")
    
    # 1. Load audio (resample to 16kHz for RMVPE)
    print(f"[1/3] Loading audio: {input_path}")
    audio, _ = librosa.load(input_path, sr=16000)
    print(f"      > Duration: {len(audio)/16000:.2f}s, Samples: {len(audio)}")
    
    # 2. Run RMVPE
    print(f"[2/3] Running RMVPE inference...")
    model_path = "models/rmvpe.pt"
    rmvpe = RMVPE_Infer(model_path, device)
    
    # Test with different thresholds
    thresholds = [0.03, 0.05, 0.10]
    results = {}
    
    for thred in thresholds:
        print(f"      > Testing with threshold={thred}")
        f0 = rmvpe.infer(audio, thred=thred)
        hop_time = len(audio) / len(f0) / 16000.0
        
        # Convert to MIDI
        midi = f0_to_midi(f0)
        voiced = np.sum(f0 > 0)
        total = len(f0)
        
        print(f"        - F0 shape: {f0.shape}")
        print(f"        - Hop time: {hop_time*1000:.2f}ms")
        print(f"        - Voiced frames: {voiced}/{total} ({100*voiced/total:.1f}%)")
        print(f"        - F0 range: {np.nanmin(f0[f0>0]):.1f} - {np.nanmax(f0[f0>0]):.1f} Hz")
        print(f"        - MIDI range: {np.nanmin(midi):.1f} - {np.nanmax(midi):.1f}")
        
        results[thred] = {
            'f0': f0,
            'midi': midi,
            'time': np.arange(len(f0)) * hop_time
        }
    
    # 3. Visualization
    print(f"[3/3] Creating visualization...")
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(12, 4*len(thresholds)))
    
    if len(thresholds) == 1:
        axes = [axes]
    
    for i, thred in enumerate(thresholds):
        data = results[thred]
        ax = axes[i]
        
        # Plot F0
        voiced_mask = data['f0'] > 0
        ax.plot(data['time'][voiced_mask], data['midi'][voiced_mask], 'b-', linewidth=1, alpha=0.7)
        ax.scatter(data['time'][voiced_mask], data['midi'][voiced_mask], c='blue', s=10, alpha=0.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('MIDI Note')
        ax.set_title(f'RMVPE F0 Detection (threshold={thred})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 90)
    
    plt.tight_layout()
    output_path = input_path.replace('.wav', '_rmvpe_test.png').replace('.mp3', '_rmvpe_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"      > Saved visualization: {output_path}")
    
    print(f"\n=== Test Complete ===")
    return results


# ==========================================
#      MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RMVPE pitch detection")
    parser.add_argument("input", help="Input audio file (WAV or MP3)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    test_rmvpe(args.input, device=args.device)
