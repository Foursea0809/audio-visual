"""Paper-aligned babble-noise augmentation for VoxMM AO/AV training.

Implements the paper setting: select one of 20 babble sources, add it at 5 dB
SNR with probability p_n=0.25, and preserve every source WAV's relative path.
The generated WAV files are training-only; original test audio must never be
passed to this script.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent
INPUT_AUDIO_DIR = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original" / "train"
NOISE_DIR = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "Babble Noise" / "Audio"
OUTPUT_DIR = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "Babble Noise" / "AO_AV_babble noise"
TARGET_SR = 16000
SNR_DB = 5.0
P_NOISE = 0.25
MAX_BABBLE_SOURCES = 20


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = signal.resample_poly(audio, TARGET_SR, sr).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def fit_noise(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        raise ValueError("Babble source is empty")
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    start = rng.randrange(0, len(noise) - length + 1)
    return noise[start:start + length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(clean ** 2) + 1e-10)
    noise_power = float(np.mean(noise ** 2) + 1e-10)
    scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
    scaled = noise * scale
    measured_snr = 10 * np.log10(signal_power / (float(np.mean(scaled ** 2)) + 1e-10))
    mixed = clean + scaled
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)
    return mixed.astype(np.float32), measured_snr


def main() -> None:
    parser = argparse.ArgumentParser(description="Add paper-specified babble noise to VoxMM training audio.")
    parser.add_argument("--input", type=Path, default=INPUT_AUDIO_DIR)
    parser.add_argument("--noise-dir", type=Path, default=NOISE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--limit", type=int, default=0, help="Process only N files (smoke test).")
    parser.add_argument("--force-noise", action="store_true", help="Apply noise to every file; not paper-default.")
    args = parser.parse_args()
    if not args.input.exists() or not args.noise_dir.exists():
        raise FileNotFoundError("Input or babble-noise directory does not exist.")

    noise_paths = sorted(args.noise_dir.rglob("*.wav")) + sorted(args.noise_dir.rglob("*.mp3"))
    noise_paths = noise_paths[:MAX_BABBLE_SOURCES]
    if len(noise_paths) < MAX_BABBLE_SOURCES:
        raise RuntimeError(f"Expected {MAX_BABBLE_SOURCES} babble sources, found {len(noise_paths)}")
    source_paths = sorted(args.input.rglob("*.wav"))
    if args.limit:
        source_paths = source_paths[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for index, source in enumerate(source_paths, 1):
        relative = source.relative_to(args.input)
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        clean = load_mono_16k(source)
        apply_noise = args.force_noise or rng.random() < P_NOISE
        noise_name, measured_snr = "", ""
        output = clean
        if apply_noise:
            noise_path = rng.choice(noise_paths)
            output, measured_snr = mix_at_snr(clean, fit_noise(load_mono_16k(noise_path), len(clean), rng), SNR_DB)
            noise_name = noise_path.name
        sf.write(destination, output, TARGET_SR, subtype="PCM_16")
        rows.append({"id": relative.with_suffix("").as_posix(), "source": str(source), "output": str(destination),
                     "augmented": apply_noise, "babble_source": noise_name, "target_snr_db": SNR_DB if apply_noise else "",
                     "measured_snr_db": measured_snr, "seed": args.seed})
        print(f"[{index}/{len(source_paths)}] {'babble' if apply_noise else 'clean'}: {relative}")

    with (args.output / "babble_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)} training WAV files; babble applied to {sum(row['augmented'] for row in rows)} ({P_NOISE:.0%} target).")


if __name__ == "__main__":
    main()
