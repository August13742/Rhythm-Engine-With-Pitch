import sys
import os

print("--- VERIFICATION START ---")

try:
    print("[1] Verifying Generator Import...")
    from generator import RhythmEngine
    print("    > Generator OK")
except ImportError as e:
    print(f"    > Generator FAILED: {e}")
    sys.exit(1)

try:
    print("[2] Verifying Basic Pitch Torch Import...")
    from transcribe.basic_pitch import BasicPitchTranscriber
    bp = BasicPitchTranscriber()
    print("    > Basic Pitch OK")
except Exception as e:
    print(f"    > Basic Pitch FAILED: {e}")
    sys.exit(1)

try:
    print("[3] Verifying Council Import...")
    from transcribe.council import CouncilV2
    c = CouncilV2()
    print("    > Council OK")
except Exception as e:
    print(f"    > Council FAILED: {e}")
    sys.exit(1)

try:
    print("[4] Verifying Visualizer Import...")
    from visualizer import Visualizer
    print("    > Visualizer OK")
except Exception as e:
    print(f"    > Visualizer FAILED: {e}")
    sys.exit(1)

try:
    print("[5] Verifying Separator Import...")
    from separator import separate_audio
    print("    > Separator OK")
except Exception as e:
    print(f"    > Separator FAILED: {e}")
    sys.exit(1)

print("--- VERIFICATION SUCCESS ---")
