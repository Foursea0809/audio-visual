"""GPU-only AO/VO/AV TM-CTC training for the prepared VoxMM manifests.

The script consumes only preprocessed mouth crops and paired WAV/transcript
files listed in JSONL manifests.  It has no CPU fallback for model training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from tmctc import TM_CTC_AVSR_Model


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = ROOT / "avsr_workspace" / "manifests" / "final"
OUTPUT_ROOT = ROOT / "avsr_workspace" / "checkpoints"
CACHE_ROOT = ROOT / "avsr_workspace" / "feature_cache"
ALPHABET = " abcdefghijklmnopqrstuvwxyz'"
CHAR_TO_ID = {char: index + 1 for index, char in enumerate(ALPHABET)}
BLANK = 0


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: CPU training is disabled for this project.")
    device = torch.device("cuda:0")
    print(f"Training device: {torch.cuda.get_device_name(device)}")
    return device


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_wav(path: Path) -> torch.Tensor:
    # VoxMM mixes PCM and floating-point WAV encodings; SoundFile supports
    # both without changing the source signal.
    values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if values.ndim == 2:
        values = values.mean(axis=1)
    audio = torch.from_numpy(values)
    if rate != 16000:
        audio = functional.interpolate(audio.view(1, 1, -1), size=round(audio.numel() * 16000 / rate), mode="linear", align_corners=False).view(-1)
    return audio


def audio_features(path: Path, maximum_steps: int) -> torch.Tensor:
    audio = read_wav(path)
    audio = (audio - audio.mean()) / audio.std().clamp_min(1e-5)
    spectrum = torch.stft(audio, n_fft=400, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True).abs()
    features = torch.log1p(spectrum).transpose(0, 1).unsqueeze(0)
    features = functional.interpolate(features, size=80, mode="linear", align_corners=False).squeeze(0)
    return features[:maximum_steps]


def visual_features(path: Path, maximum_frames: int, augment: bool = False) -> torch.Tensor:
    frames = sorted(path.glob("frame_*.png"))[:maximum_frames]
    if not frames:
        raise FileNotFoundError(f"No processed mouth frames: {path}")
    images = [cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE) for frame in frames]
    if any(image is None for image in images):
        raise ValueError(f"Unreadable processed mouth frame in {path}")
    if augment:
        # 112->128 reflective padding then random 112 crop, horizontal flip
        # p=.5, and random frame removal p=.1, matching the paper's intent.
        kept = [image for image in images if random.random() >= 0.10]
        images = kept or images[:1]
        augmented = []
        for image in images:
            padded = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REFLECT_101)
            top, left = random.randrange(17), random.randrange(17)
            crop = padded[top:top + 112, left:left + 112]
            augmented.append(cv2.flip(crop, 1) if random.random() < 0.5 else crop)
        images = augmented
    tensor = torch.from_numpy(np.stack(images).astype(np.float32) / 255.0)
    # Training-set image statistics, rather than only [0,1] scaling.
    return ((tensor - 0.421) / 0.165).unsqueeze(0)


def encode(text: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_ID[char] for char in text if char in CHAR_TO_ID and char != " " or char == " "], dtype=torch.long)


class VoxMMDataset(Dataset):
    def __init__(self, rows: list[dict], mode: str, maximum_frames: int, cache_dir: Path | None = None,
                 training_augment: bool = False, audio_source: str = "clean"):
        self.rows, self.mode, self.maximum_frames = rows, mode, maximum_frames
        self.cache_dir, self.training_augment, self.audio_source = cache_dir, training_augment, audio_source

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        target = encode(row["transcript"])
        if not len(target):
            raise ValueError(f"Empty CTC target: {row['id']}")
        item = {"target": target, "id": row["id"]}
        if self.mode != "ao":
            item["video"] = visual_features(Path(row["visual_dir"]), self.maximum_frames, self.training_augment)
        if self.mode != "vo":
            item["audio"] = self.cached_audio(row)
        return item

    def cached_audio(self, row: dict) -> torch.Tensor:
        """Persist CPU STFT features once; later epochs only load a tensor."""
        source = row["audio_babble"] if self.audio_source == "babble" else row["audio"]
        if self.audio_source == "babble" and not source:
            raise FileNotFoundError(f"Babble audio missing for {row['id']}")
        if self.cache_dir is None:
            return audio_features(Path(source), self.maximum_frames * 4)
        cache_path = self.cache_dir / f"audio_{self.audio_source}" / f"{row['id']}.pt"
        if cache_path.is_file():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        feature = audio_features(Path(source), self.maximum_frames * 4)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(feature, temporary)
        os.replace(temporary, cache_path)
        return feature


class ThermalBatchSampler(Sampler[list[int]]):
    """Use the requested batch size while cool and batch 1 after crossing a hot limit."""
    def __init__(self, dataset_size: int, maximum_batch_size: int, high_temperature: int,
                 low_temperature: int, poll_batches: int = 10):
        self.dataset_size = dataset_size
        self.maximum_batch_size = maximum_batch_size
        self.high_temperature = high_temperature
        self.low_temperature = low_temperature
        self.poll_batches = poll_batches
        self.current_batch_size = maximum_batch_size

    def __iter__(self):
        indices = list(range(self.dataset_size))
        random.shuffle(indices)
        position, batch_number = 0, 0
        while position < len(indices):
            if batch_number % self.poll_batches == 0:
                value = gpu_temperature()
                if value >= self.high_temperature:
                    desired = 1
                elif value <= self.low_temperature:
                    desired = self.maximum_batch_size
                else:
                    desired = self.current_batch_size
                if desired != self.current_batch_size:
                    self.current_batch_size = desired
                    print(f"thermal batch control: temperature={value}C batch_size={desired}", flush=True)
            size = min(self.current_batch_size, len(indices) - position)
            yield indices[position:position + size]
            position += size
            batch_number += 1

    def __len__(self) -> int:
        return self.dataset_size


def prebuild_audio_cache(dataset: VoxMMDataset) -> None:
    """Build the persistent CPU audio-feature cache before GPU training."""
    if dataset.mode == "vo":
        print("VO mode has no audio features to cache.", flush=True)
        return
    total = len(dataset)
    for index, row in enumerate(dataset.rows, 1):
        dataset.cached_audio(row)
        if index % 100 == 0 or index == total:
            print(f"audio cache: {index}/{total}", flush=True)


def collate(items: list[dict]) -> dict:
    result = {"targets": torch.cat([item["target"] for item in items]), "target_lengths": torch.tensor([len(item["target"]) for item in items])}
    if "audio" in items[0]:
        audio = [item["audio"] for item in items]
        result["audio"] = pad_sequence(audio, batch_first=True)
        result["audio_lengths"] = torch.tensor([value.size(0) for value in audio])
    if "video" in items[0]:
        video = [item["video"].transpose(0, 1) for item in items]
        padded = pad_sequence(video, batch_first=True).transpose(1, 2)
        result["video"] = padded
        result["video_lengths"] = torch.tensor([value.size(0) * 4 for value in video])
    return result


def gpu_temperature() -> int:
    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return int(result.stdout.strip().splitlines()[0])


def cool_if_needed(maximum: int, resume: int) -> None:
    while gpu_temperature() >= maximum:
        print(f"GPU temperature >= {maximum}C; pausing until below {resume}C.", flush=True)
        while gpu_temperature() > resume:
            time.sleep(15)


def effective_lengths(batch: dict, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "ao":
        lengths = batch["audio_lengths"]
        output_steps = batch["audio"].size(1)
    elif mode == "vo":
        lengths = batch["video_lengths"]
        output_steps = batch["video"].size(2) * 4
    else:
        lengths = torch.minimum(batch["audio_lengths"], batch["video_lengths"])
        output_steps = min(batch["audio"].size(1), batch["video"].size(2) * 4)
    mask = torch.arange(output_steps).unsqueeze(0) >= lengths.unsqueeze(1)
    return lengths.to(device), mask.to(device)


def run_epoch(model, loader, optimizer, scaler, device, mode: str, accumulation_steps: int,
              max_temperature: int, resume_temperature: int, thermal_batch_control: bool = False) -> float:
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    model.train()
    total, batches, pending_update = 0.0, 0, False
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        if not thermal_batch_control:
            cool_if_needed(max_temperature, resume_temperature)
        video = batch.get("video")
        audio = batch.get("audio")
        video = video.to(device, non_blocking=True) if video is not None else None
        audio = audio.to(device, non_blocking=True) if audio is not None else None
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths, padding_mask = effective_lengths(batch, mode, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            log_probs = model(video, audio, mode=mode, padding_mask=padding_mask)
            input_lengths = input_lengths.clamp_max(log_probs.size(0))
            loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)
        scaler.scale(loss / accumulation_steps).backward()
        pending_update = True
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_update = False
        total += float(loss.detach())
        batches += 1
    if pending_update:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total / max(1, batches)


def main(forced_mode: str | None = None) -> None:
    """Run one TM-CTC experiment, optionally locking its modality for a dedicated entry point."""
    if forced_mode not in (None, "ao", "vo", "av"):
        raise ValueError(f"Unsupported forced mode: {forced_mode}")
    parser = argparse.ArgumentParser()
    if forced_mode is None:
        parser.add_argument("--mode", choices=("ao", "vo", "av"), required=True)
    parser.add_argument("--visual-source", choices=("original", "gan"), default="original")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="GTX 1050-safe physical batch.")
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Effective batch without higher peak heat.")
    parser.add_argument("--maximum-frames", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT,
                        help="Persistent CPU audio-STFT cache; set an empty path only for diagnostics.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="CPU workers for cached feature loading from the SSD.")
    parser.add_argument("--run-name", type=str,
                        help="Checkpoint subdirectory; use a new name to preserve an earlier run.")
    parser.add_argument("--resume-from", type=Path,
                        help="Resume model weights from a checkpoint in the same run.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Completed epoch when resuming a legacy checkpoint without epoch metadata.")
    parser.add_argument("--audio-source", choices=("clean", "babble"), default="clean",
                        help="Use clean WAVs or the deterministic 25%% 5-dB babble training tree.")
    parser.add_argument("--training-augment", action="store_true",
                        help="Enable training-only visual augmentation (unused by AO).")
    parser.add_argument("--max-gpu-temperature", type=int, default=84,
                        help="Pause new batches at this GPU temperature (Celsius).")
    parser.add_argument("--resume-gpu-temperature", type=int, default=78,
                        help="Resume batches only after cooling to this temperature (Celsius).")
    parser.add_argument("--thermal-batch-control", action="store_true",
                        help="Switch between batch 2 and batch 1 instead of pausing at the hot threshold.")
    parser.add_argument("--thermal-high-temperature", type=int, default=84,
                        help="Use batch 1 at or above this temperature (Celsius).")
    parser.add_argument("--thermal-low-temperature", type=int, default=78,
                        help="Restore the requested batch size at or below this temperature (Celsius).")
    parser.add_argument("--thermal-poll-batches", type=int, default=10,
                        help="Check temperature once per this many dynamically sampled batches.")
    parser.add_argument("--precache-only", action="store_true",
                        help="Build persistent audio features, then exit without model training.")
    args = parser.parse_args()
    args.mode = forced_mode or args.mode
    random.seed(args.seed); torch.manual_seed(args.seed)
    suffix = args.visual_source if args.mode != "ao" else "original"
    train_rows = read_rows(args.manifest_dir / f"train_{suffix}.jsonl")
    dataset = VoxMMDataset(train_rows, args.mode, args.maximum_frames, args.cache_dir,
                           args.training_augment, args.audio_source)
    if args.precache_only:
        prebuild_audio_cache(dataset)
        return
    device = require_cuda()
    loader_options = dict(
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    if args.thermal_batch_control:
        if args.batch_size < 2:
            raise ValueError("Thermal batch control requires --batch-size 2 or greater.")
        sampler = ThermalBatchSampler(len(dataset), args.batch_size, args.thermal_high_temperature,
                                      args.thermal_low_temperature, args.thermal_poll_batches)
        loader = DataLoader(dataset, batch_sampler=sampler, **loader_options)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_options)
    model = TM_CTC_AVSR_Model(num_classes=len(CHAR_TO_ID) + 1, mode=args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    output = OUTPUT_ROOT / (args.run_name or f"{args.mode}_{suffix}")
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = args.start_epoch
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = max(start_epoch, int(checkpoint.get("epoch", 0)))
        print(f"Resumed model weights from {args.resume_from}; continuing after epoch {start_epoch}.")
    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        loss = run_epoch(model, loader, optimizer, scaler, device, args.mode, args.accumulation_steps,
                         args.max_gpu_temperature, args.resume_gpu_temperature, args.thermal_batch_control)
        history.append({"epoch": epoch, "ctc_loss": loss})
        torch.save({"state_dict": model.state_dict(), "mode": args.mode, "visual_source": suffix,
                    "epoch": epoch, "ctc_loss": loss}, output / "last.pt")
        print(f"epoch {epoch:03d}: ctc_loss={loss:.4f}")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
