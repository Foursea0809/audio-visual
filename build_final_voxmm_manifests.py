"""Build leakage-safe VoxMM manifests after original and GAN preprocessing.

Only Python is used.  Train/validation rows reuse the established group split;
GAN video rows inherit audio and transcripts exclusively from their matching
original *train* row.  Test rows are never added to a training manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "avsr_workspace"
MANIFESTS = WORKSPACE / "manifests"
RAW_TRAIN = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original" / "train"
RAW_TEST = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original" / "test"
RAW_GAN = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original" / "gan"
ORIGINAL_VISUAL = WORKSPACE / "visual_original_train"
TEST_VISUAL = WORKSPACE / "visual_original_test"
GAN_VISUAL = WORKSPACE / "visual_gan_train"
BABBLE_AUDIO = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "Babble Noise" / "AO_AV_babble noise"


def transcript(path: Path) -> str:
    """Extract the spoken sentence from LRS-VoxMM's text-plus-timing line."""
    first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    first_line = re.sub(r"^text\s+", "", first_line, flags=re.IGNORECASE)
    return re.split(r"\s+conf\s+[-+]?\d", first_line, maxsplit=1, flags=re.IGNORECASE)[0].strip().lower()


def visual_ready(root: Path, sample_id: str) -> bool:
    return (root / sample_id / "preprocess.json").is_file()


def make_row(sample_id: str, split: str, video: Path, visual: Path, is_gan: bool) -> dict:
    audio = RAW_TRAIN / f"{sample_id}.wav" if split != "test" else RAW_TEST / f"{sample_id}.wav"
    text_file = RAW_TRAIN / f"{sample_id}.txt" if split != "test" else RAW_TEST / f"{sample_id}.txt"
    if not (video.is_file() and audio.is_file() and text_file.is_file() and visual_ready(visual, sample_id)):
        raise FileNotFoundError(f"Incomplete paired sample: {sample_id}")
    babble = BABBLE_AUDIO / f"{sample_id}.wav" if split != "test" else None
    return {
        "id": sample_id,
        "group_id": sample_id.split("/", 1)[0],
        "split": split,
        "transcript": transcript(text_file),
        "audio": str(audio),
        # This tree contains the deterministic 25% 5-dB babble replacement
        # and clean copies for the other rows, matching the paper's pn=.25.
        "audio_babble": str(babble) if babble and babble.is_file() else None,
        "video": str(video),
        "visual_dir": str(visual / sample_id / "Media"),
        "is_gan": is_gan,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def refresh_babble_fields(output_dir: Path) -> None:
    """Add deterministic babble-audio paths to already-built train/val manifests."""
    changed = 0
    for split in ("train", "val"):
        for source in ("original", "gan"):
            path = output_dir / f"{split}_{source}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                candidate = BABBLE_AUDIO / f"{row['id']}.wav"
                row["audio_babble"] = str(candidate) if candidate.is_file() else None
                changed += row["audio_babble"] is not None
            write_jsonl(path, rows)
    print(json.dumps({"babble_paths_added": changed}, indent=2))


def build_rows(sample_ids: list[str], split: str, raw_root: Path, visual_root: Path,
               video_root: Path, is_gan: bool) -> tuple[list[dict], list[str]]:
    rows, excluded = [], []
    for sample_id in sample_ids:
        try:
            rows.append(make_row(sample_id, split, video_root / f"{sample_id}.mp4", visual_root, is_gan))
        except FileNotFoundError:
            excluded.append(sample_id)
    return rows, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=MANIFESTS / "final")
    parser.add_argument("--test-only", action="store_true", help="Refresh only the independent test manifest.")
    parser.add_argument("--refresh-babble-fields", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_babble_fields:
        refresh_babble_fields(args.output_dir)
        return

    if args.test_only:
        test_ids = [
            path.relative_to(RAW_TEST).with_suffix("").as_posix()
            for path in sorted(RAW_TEST.rglob("*.mp4"))
            if path.with_suffix(".wav").is_file() and path.with_suffix(".txt").is_file()
        ]
        tests, excluded_test = build_rows(test_ids, "test", RAW_TEST, TEST_VISUAL, RAW_TEST, False)
        write_jsonl(args.output_dir / "test_original.jsonl", tests)
        summary_path = args.output_dir / "summary.json"
        result = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        result["test_original"] = len(tests)
        result["policy"] = "GAN rows map only to original/train audio and transcripts; test has no GAN training manifest."
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.output_dir / "excluded_incomplete.json").write_text(
            json.dumps({"test_original": excluded_test}, indent=2), encoding="utf-8"
        )
        print(json.dumps({"test_original": len(tests), "excluded": len(excluded_test)}, indent=2))
        return

    # Build from the actual paired media tree, rather than a prior manifest.
    # The split is group-stable: every segment from one source video goes to
    # either train or validation, never both.
    source_rows: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for video in sorted(RAW_TRAIN.rglob("*.mp4")):
        relative = video.relative_to(RAW_TRAIN).with_suffix("").as_posix()
        if not video.with_suffix(".wav").is_file() or not video.with_suffix(".txt").is_file():
            continue
        group = relative.split("/", 1)[0]
        bucket = "val" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100 < 5 else "train"
        source_rows[bucket].append(relative)
    for video in sorted(RAW_TEST.rglob("*.mp4")):
        relative = video.relative_to(RAW_TEST).with_suffix("").as_posix()
        if video.with_suffix(".wav").is_file() and video.with_suffix(".txt").is_file():
            source_rows["test"].append(relative)
    result: dict[str, int] = {}
    excluded: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        raw_root = RAW_TEST if split == "test" else RAW_TRAIN
        visual_root = TEST_VISUAL if split == "test" else ORIGINAL_VISUAL
        originals, excluded_original = build_rows(source_rows[split], split, raw_root, visual_root, raw_root, False)
        write_jsonl(args.output_dir / f"{split}_original.jsonl", originals)
        result[f"{split}_original"] = len(originals)
        excluded[f"{split}_original"] = excluded_original
        if split != "test":
            gan_rows, excluded_gan = build_rows(source_rows[split], split, raw_root, GAN_VISUAL, RAW_GAN, True)
            write_jsonl(args.output_dir / f"{split}_gan.jsonl", gan_rows)
            result[f"{split}_gan"] = len(gan_rows)
            excluded[f"{split}_gan"] = excluded_gan
    result["policy"] = "GAN rows map only to original/train audio and transcripts; test has no GAN training manifest."
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "excluded_incomplete.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
