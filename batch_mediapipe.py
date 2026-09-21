"""Paper-aligned VoxMM visual preprocessing.

Produces 112x112 grayscale mouth ROIs from paired original VoxMM videos.  By
default it is deterministic and safe for train/validation/test creation.  The
paper's stochastic crop/flip/frame-removal augmentation is opt-in and must be
used for a training-only output directory.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original" / "train"
DEFAULT_OUTPUT = ROOT / "avsr_workspace" / "visual_original_train"
DEFAULT_MODEL = ROOT / "MediaPipe" / "face_landmarker.task"
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 0, 17, 61, 291
ROI_SIZE, DETECTION_SIZE = 112, 128


def crop_with_padding(image: np.ndarray, center_x: float, center_y: float, side: int) -> np.ndarray:
    """Return a stable crop even when the mouth lies at an image edge."""
    h, w = image.shape[:2]
    half = side // 2
    x1, y1 = round(center_x) - half, round(center_y) - half
    x2, y2 = x1 + side, y1 + side
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - w), max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        x1, x2 = x1 + pad_left, x2 + pad_left
        y1, y2 = y1 + pad_top, y2 + pad_top
    return image[y1:y2, x1:x2]


def landmark_center(landmarks, width: int, height: int) -> tuple[float, float]:
    left, right = landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]
    top, bottom = landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]
    return ((left.x + right.x) * width / 2.0, (top.y + bottom.y) * height / 2.0)


def make_landmarker(model_path: Path):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_video(video_path: Path, input_root: Path, output_root: Path, model_path: Path,
                  training_augment: bool, rng: random.Random, overwrite: bool) -> dict:
    relative = video_path.relative_to(input_root).with_suffix("")
    media_dir = output_root / relative / "Media"
    meta_path = output_root / relative / "preprocess.json"
    if meta_path.exists() and not overwrite:
        return {"id": relative.as_posix(), "status": "skipped"}
    media_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    written, fallbacks, frame_index, last_center = 0, 0, 0, None
    detector = make_landmarker(model_path)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            # Paper augmentation is training-only: randomly remove frames.
            if training_augment and rng.random() < 0.10:
                frame_index += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(frame_index * 1000 / fps))
            height, width = bgr.shape[:2]
            if result.face_landmarks:
                center = landmark_center(result.face_landmarks[0], width, height)
                last_center = center
            elif last_center is not None:
                center, fallbacks = last_center, fallbacks + 1
            else:
                # Preserve the sample if its first frame fails landmark detection.
                center, fallbacks = (width / 2.0, height * 0.62), fallbacks + 1
            side = DETECTION_SIZE if training_augment else ROI_SIZE
            roi = crop_with_padding(bgr, center[0], center[1], side)
            if training_augment:
                # 128 -> random 112 crop, then horizontal flip p=0.5.
                offset_x, offset_y = rng.randrange(17), rng.randrange(17)
                roi = roi[offset_y:offset_y + ROI_SIZE, offset_x:offset_x + ROI_SIZE]
                if rng.random() < 0.5:
                    roi = cv2.flip(roi, 1)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(media_dir / f"frame_{written:05d}.png"), gray)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        detector.close()
    metadata = {"id": relative.as_posix(), "source": str(video_path), "frames": written, "fps": fps,
                "roi": "112x112 grayscale mouth region", "training_augmentation": training_augment,
                "landmark_fallback_frames": fallbacks}
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"id": relative.as_posix(), "status": "ok", "frames": written, "fallbacks": fallbacks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess paired VoxMM videos to paper-format visual tensors.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paired-root", type=Path, default=None,
                        help="Optional WAV/TXT root.  Use original/train when input-dir is GAN-only MP4 files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--training-augment", action="store_true", help="Enable random 112 crop, flip p=.5, frame removal; train only.")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.input_dir.exists() or not args.model_path.exists():
        raise FileNotFoundError("Input directory or MediaPipe face_landmarker.task is missing.")
    # A GAN directory intentionally contains only generated MP4 files.  When
    # paired-root is supplied, validate each relative path against that root;
    # this keeps test samples out of the train pipeline while reusing the
    # original training audio/transcript labels.
    pair_root = args.paired_root or args.input_dir
    videos = [
        path for path in sorted(args.input_dir.rglob("*.mp4"))
        if (pair_root / path.relative_to(args.input_dir)).with_suffix(".wav").exists()
        and (pair_root / path.relative_to(args.input_dir)).with_suffix(".txt").exists()
    ]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise RuntimeError("No paired MP4/WAV/TXT samples found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng, rows = random.Random(args.seed), []
    print(f"Processing {len(videos)} paired videos; training augmentation={args.training_augment}.")
    for index, video in enumerate(videos, 1):
        row = process_video(video, args.input_dir, args.output_dir, args.model_path, args.training_augment, rng, args.overwrite)
        rows.append(row)
        print(f"[{index}/{len(videos)}] {row['status']}: {row['id']}")
    (args.output_dir / "preprocess_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
