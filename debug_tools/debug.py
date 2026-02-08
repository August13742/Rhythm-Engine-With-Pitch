import os
import shutil
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from audio_separator.separator import Separator

# --- CONFIG ---
TARGET_FILE = "LionessPrideTwilight.mp3"
MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEBUG_DIR = Path("debug_output")
MODEL_DIR = Path("models")

# --- SETUP ---
if DEBUG_DIR.exists(): shutil.rmtree(DEBUG_DIR)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
print(f"--- DEBUG MODE: Processing {TARGET_FILE} ---")

# 1. ANALYZE INPUT
if not os.path.exists(TARGET_FILE):
    print(f"[ERROR] {TARGET_FILE} not found.")
    exit(1)

data, sr = sf.read(TARGET_FILE)
peak = np.max(np.abs(data))
print(f"[INPUT] Sample Rate: {sr}, Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
print(f"[INPUT] Peak Amplitude: {peak:.4f}")

if peak < 0.01:
    print("[CRITICAL] Input file is effectively silent!")

# 2. RUN SEPARATOR (RAW)
print("\n[PROCESS] Initializing Separator (No Custom Logic)...")
try:
    sep = Separator(
        output_dir=str(DEBUG_DIR),
        model_file_dir=str(MODEL_DIR),
        output_format="WAV",
        log_level=logging.INFO 
    )
    sep.load_model(MODEL_NAME)
    
    # Run separation
    output_files = sep.separate(TARGET_FILE)
    print(f"[PROCESS] Separation finished. Returned: {output_files}")

except Exception as e:
    print(f"[ERROR] Separation crashed: {e}")
    exit(1)

# 3. VERIFY OUTPUTS IMMEDIATELY
print("\n[VERIFY] Inspecting generated files...")
files_found = list(DEBUG_DIR.glob("*.wav"))

if not files_found:
    print("[FAIL] No WAV files found in output directory!")
else:
    for f in files_found:
        try:
            y, r = sf.read(str(f))
            file_peak = np.max(np.abs(y))
            status = "OK" if file_peak > 0.01 else "SILENT"
            print(f"File: {f.name:<30} | Peak: {file_peak:.4f} | Status: {status}")
        except Exception as e:
            print(f"File: {f.name:<30} | [READ ERROR] {e}")

print("\n--- DEBUG COMPLETE ---")