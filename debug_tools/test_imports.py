#!/usr/bin/env python
import sys
print("Testing imports...", flush=True)

try:
    print("Importing numpy...", flush=True)
    import numpy as np
    print(f"numpy OK: {np.__version__}", flush=True)
except Exception as e:
    print(f"numpy FAILED: {e}", flush=True)
    sys.exit(1)

try:
    print("Importing librosa...", flush=True)
    import librosa
    print(f"librosa OK: {librosa.__version__}", flush=True)
except Exception as e:
    print(f"librosa FAILED: {e}", flush=True)
    sys.exit(1)

try:
    print("Importing pretty_midi...", flush=True)
    import pretty_midi
    print(f"pretty_midi OK", flush=True)
except Exception as e:
    print(f"pretty_midi FAILED: {e}", flush=True)
    sys.exit(1)

print("All imports successful!", flush=True)
