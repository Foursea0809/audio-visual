"""Prepare LRS-VoxMM for the TM-CTC AVSR recipe described in the paper.

The script keeps source media read-only and writes manifests/statistics beneath
VoxMM/avsr_workspace.  It expects the extracted original archive at
LRS-VoxMM/lrs-voxmm/original/{train,test}/<video-id>/<segment>.{mp4,wav,txt}.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "LRS-VoxMM" / "lrs-voxmm" / "original"
WORK = ROOT / "avsr_workspace"

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--gan-root", type=Path, default=ROOT / "avsr_workspace" / "wav2lip_generated")
    parser.add_argument("--require-gan", action="store_true", help="Emit only records with generated videos.")
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Extract original.tar.gz first: {args.source}")
    WORK.mkdir(exist_ok=True)
    totals = {}
    for split_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        rows = []
        for text_path in split_dir.rglob("*.txt"):
            stem = text_path.stem
            wav_path, video_path = text_path.with_suffix(".wav"), text_path.with_suffix(".mp4")
            gan_path = args.gan_root / split_dir.name / text_path.parent.name / f"{stem}.mp4"
            text = clean(text_path.read_text(encoding="utf-8", errors="ignore"))
            if not (text and wav_path.exists() and video_path.exists()):
                continue
            if args.require_gan and not gan_path.exists():
                continue
            rows.append({"id": f"{text_path.parent.name}/{stem}", "split": split_dir.name,
                         "text": text, "audio": str(wav_path.resolve()),
                         "video_original": str(video_path.resolve()),
                         "video_gan": str(gan_path.resolve()) if gan_path.exists() else None})
        output = WORK / f"{split_dir.name}_manifest.jsonl"
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
        totals[split_dir.name] = {"segments": len(rows), "with_gan": sum(row["video_gan"] is not None for row in rows)}
    (WORK / "manifest_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))

if __name__ == "__main__": main()
