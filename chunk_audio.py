import os
import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# --- Configuration ---
TARGET_SR       = 22050       # target sample rate (Hz)
CHUNK_SECS      = 1.5         # chunk length in seconds
OVERLAP_SECS    = 0.5         # overlap between chunks in seconds
STEP_SECS       = CHUNK_SECS - OVERLAP_SECS   # 1.0 second
BIT_DEPTH       = 'PCM_16'    # 16-bit WAV
EXTENSIONS      = {'.mp3', '.wav'}

def process_file(input_path: Path, output_dir: Path):
    print(f"Processing: {input_path}")

    # Load and resample to target sample rate, mix down to mono
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

    chunk_samples = int(CHUNK_SECS * TARGET_SR)   # 33075 samples
    step_samples  = int(STEP_SECS  * TARGET_SR)   # 22050 samples

    stem = input_path.stem   # original filename without extension
    seq  = 0

    for start in range(0, len(audio), step_samples):
        end   = start + chunk_samples
        chunk = audio[start:end]

        # Skip chunks that are too short (last partial chunk)
        if len(chunk) < chunk_samples:
            break

        out_filename = f"{stem}_{seq:04d}.wav"
        out_path     = output_dir / out_filename

        sf.write(out_path, chunk, TARGET_SR, subtype=BIT_DEPTH)
        seq += 1

    print(f"  -> {seq} chunks written")

def main():
    if len(sys.argv) != 3:
        print("Usage: python chunk_audio.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files, handling both lower and upper case extensions
    audio_files = []
    for ext in EXTENSIONS:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
        audio_files.extend(input_dir.rglob(f"*{ext.upper()}"))

    # Ensure only files are included (not directories with dotted names)
    audio_files = [f for f in audio_files if f.is_file()]

    print(f"Found {len(audio_files)} audio files:")
    for f in sorted(audio_files):
        print(f"  {f}")

    if not audio_files:
        print("No mp3 or wav files found.")
        sys.exit(0)

    for f in sorted(audio_files):
        process_file(f, output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
