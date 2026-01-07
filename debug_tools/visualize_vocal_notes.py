"""
Visualize vocal MIDI notes over time with delta analysis.
Shows pitch progression and sudden jumps.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_vocal_notes(beatmap_path):
    """Load vocal notes from beatmap JSON."""
    with open(beatmap_path, 'r') as f:
        beatmap = json.load(f)
    vocal_notes = [note for note in beatmap if note.get('source') == 'vocals']
    return sorted(vocal_notes, key=lambda n: n['time'])


def analyze_midi_curve(vocal_notes):
    """Extract times, MIDI values, and compute deltas."""
    times = np.array([n['time'] for n in vocal_notes])
    midis = np.array([n['midi'] for n in vocal_notes])
    
    # Compute deltas (difference from previous note)
    deltas = np.diff(midis)
    delta_times = times[1:]  # Times where deltas occur
    
    return times, midis, delta_times, deltas


def visualize_midi_curve(vocal_notes, times, midis, delta_times, deltas, output_path=None):
    """Create temporal MIDI curve visualization."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # ===== PLOT 1: MIDI Curve =====
    ax1.plot(times, midis, 'b-', linewidth=1, marker='o', markersize=3, alpha=0.7, label='MIDI Notes')
    ax1.fill_between(times, midis, alpha=0.2)
    
    # Highlight large jumps (> 2 semitones)
    large_jumps = np.abs(deltas) > 2.0
    large_jump_times = delta_times[large_jumps]
    large_jump_midis = midis[1:][large_jumps]
    
    if len(large_jump_times) > 0:
        ax1.scatter(large_jump_times, large_jump_midis, c='red', s=100, marker='X', 
                   zorder=5, label=f'Large jumps (>2 semitones): {sum(large_jumps)}')
    
    ax1.set_ylabel('MIDI Note', fontsize=12)
    ax1.set_title('Vocal MIDI Note Progression Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([times[0], times[-1]])
    
    # ===== PLOT 2: Delta (Pitch Jump) Curve =====
    colors = ['red' if abs(d) > 2 else 'orange' if abs(d) > 1 else 'green' for d in deltas]
    widths = np.diff(times)  # Width between consecutive note times
    ax2.bar(delta_times, deltas, width=widths, 
           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2 semitones')
    ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='±1 semitone')
    ax2.axhline(y=-1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('MIDI Delta (semitones)', fontsize=12)
    ax2.set_title('Pitch Jump Between Consecutive Notes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    ax2.set_xlim([times[0], times[-1]])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved to: {output_path}")
    
    plt.show()


def print_analysis(vocal_notes, times, midis, delta_times, deltas):
    """Print analysis report."""
    
    print("\n" + "="*80)
    print("VOCAL MIDI ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTotal notes: {len(vocal_notes)}")
    print(f"Time range: {times[0]:.3f}s - {times[-1]:.3f}s (duration: {times[-1] - times[0]:.3f}s)")
    print(f"MIDI range: {int(np.min(midis))} - {int(np.max(midis))}")
    
    print(f"\nDELTA STATISTICS (pitch jumps between consecutive notes):")
    print(f"  Mean jump:       {np.mean(np.abs(deltas)):.3f} semitones")
    print(f"  Median jump:     {np.median(np.abs(deltas)):.3f} semitones")
    print(f"  Std dev:         {np.std(deltas):.3f} semitones")
    print(f"  Max jump:        {np.max(np.abs(deltas)):.1f} semitones")
    print(f"  Max up:          {np.max(deltas):+.1f} semitones")
    print(f"  Max down:        {np.min(deltas):+.1f} semitones")
    
    # Categorize jumps
    smooth = sum(np.abs(deltas) <= 1.0)
    moderate = sum((np.abs(deltas) > 1.0) & (np.abs(deltas) <= 2.0))
    large = sum(np.abs(deltas) > 2.0)
    
    print(f"\nJUMP DISTRIBUTION:")
    print(f"  Smooth    (≤1 semitone):   {smooth:4d}  ({100*smooth/len(deltas):5.1f}%)")
    print(f"  Moderate  (1-2 semitones): {moderate:4d}  ({100*moderate/len(deltas):5.1f}%)")
    print(f"  Large     (>2 semitones):  {large:4d}  ({100*large/len(deltas):5.1f}%)")
    
    if large > 0:
        print(f"\n⚠️  LARGE JUMPS ({large}):")
        large_indices = np.where(np.abs(deltas) > 2.0)[0]
        for i in large_indices[:20]:
            t1, t2 = times[i], times[i+1]
            m1, m2 = int(midis[i]), int(midis[i+1])
            delta = deltas[i]
            print(f"  t={t1:7.3f}s→{t2:7.3f}s  {m1:3d}→{m2:3d}  ({delta:+.1f} semitones)")
        if len(large_indices) > 20:
            print(f"  ... and {len(large_indices) - 20} more")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize vocal MIDI progression and pitch jumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_vocal_notes.py stems/OneLastYou/OneLastYou_INSANE.json
  python visualize_vocal_notes.py stems/1/1_HARD.json --output notes.png
        """
    )
    
    parser.add_argument("beatmap", help="Path to beatmap JSON")
    parser.add_argument("--output", "-o", help="Save visualization to PNG")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.beatmap):
        print(f"[ERROR] Beatmap not found: {args.beatmap}")
        sys.exit(1)
    
    print(f"[LOAD] Loading beatmap: {args.beatmap}")
    vocal_notes = load_vocal_notes(args.beatmap)
    
    if not vocal_notes:
        print("[ERROR] No vocal notes found!")
        sys.exit(1)
    
    print(f"[LOAD] Found {len(vocal_notes)} vocal notes")
    
    times, midis, delta_times, deltas = analyze_midi_curve(vocal_notes)
    
    print_analysis(vocal_notes, times, midis, delta_times, deltas)
    print("[VIZ] Creating visualization...")
    visualize_midi_curve(vocal_notes, times, midis, delta_times, deltas, args.output)


if __name__ == "__main__":
    main()
