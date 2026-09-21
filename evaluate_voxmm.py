"""Greedy and width-100 CTC beam+LM WER evaluation for VoxMM checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from tmctc import TM_CTC_AVSR_Model
from train_voxmm import ALPHABET, BLANK, CHAR_TO_ID, VoxMMDataset, collate, effective_lengths, read_rows, require_cuda
from voxmm_language_model import BOS, EOS, GRULanguageModel, OUTPUT_DIR as LM_DIR, normalize_transcript


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "avsr_workspace" / "manifests" / "final"
RESULT_DIR = ROOT / "avsr_workspace" / "results"
ID_TO_CHAR = {value: key for key, value in CHAR_TO_ID.items()}


def collapse(ids: list[int]) -> str:
    previous, chars = BLANK, []
    for token in ids:
        if token != BLANK and token != previous:
            chars.append(ID_TO_CHAR.get(token, ""))
        previous = token
    return "".join(chars).strip()


def greedy(log_probs: torch.Tensor) -> str:
    return collapse(log_probs.argmax(dim=-1).tolist())


def log_add(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    result = finite[0]
    for value in finite[1:]:
        result = max(result, value) + math.log1p(math.exp(-abs(result - value)))
    return result


def ctc_prefix_beam(log_probs: torch.Tensor, width: int) -> list[tuple[str, float]]:
    """Character-level CTC prefix beam search; returns N-best strings/scores."""
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    for step in log_probs.tolist():
        next_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(lambda: (-math.inf, -math.inf))
        for prefix, (p_blank, p_nonblank) in beams.items():
            blank, nonblank = next_beams[prefix]
            next_beams[prefix] = (log_add(blank, p_blank + step[BLANK], p_nonblank + step[BLANK]), nonblank)
            for token in range(1, len(step)):
                probability = step[token]
                extended = prefix + (token,)
                if prefix and token == prefix[-1]:
                    blank, nonblank = next_beams[prefix]
                    next_beams[prefix] = (blank, log_add(nonblank, p_nonblank + probability))
                    blank, nonblank = next_beams[extended]
                    next_beams[extended] = (blank, log_add(nonblank, p_blank + probability))
                else:
                    blank, nonblank = next_beams[extended]
                    next_beams[extended] = (blank, log_add(nonblank, p_blank + probability, p_nonblank + probability))
        beams = dict(sorted(next_beams.items(), key=lambda item: log_add(*item[1]), reverse=True)[:width])
    result = [(collapse(list(prefix)), log_add(p_blank, p_nonblank)) for prefix, (p_blank, p_nonblank) in beams.items()]
    return [(text, score) for text, score in result if text]


class ExternalLM:
    def __init__(self) -> None:
        self.vocab = json.loads((LM_DIR / "vocab.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(LM_DIR / "best_model.pt", map_location="cpu", weights_only=True)
        self.model = GRULanguageModel(checkpoint["vocab_size"], checkpoint["embedding_dim"], checkpoint["hidden_dim"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        words = normalize_transcript(text).split()
        if not words:
            return -50.0
        ids = [self.vocab[BOS]] + [self.vocab.get(word, self.vocab["<unk>"]) for word in words] + [self.vocab[EOS]]
        inputs, targets = torch.tensor([ids[:-1]]), torch.tensor(ids[1:])
        logits, _ = self.model(inputs)
        return float(functional.log_softmax(logits[0], dim=-1).gather(1, targets.unsqueeze(1)).sum()) / len(targets)


def word_error(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = normalize_transcript(reference).split(), normalize_transcript(hypothesis).split()
    table = list(range(len(hyp) + 1))
    for index, word in enumerate(ref, 1):
        next_row = [index]
        for column, predicted in enumerate(hyp, 1):
            next_row.append(min(next_row[-1] + 1, table[column] + 1, table[column - 1] + (word != predicted)))
        table = next_row
    return table[-1], len(ref)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("ao", "vo", "av"), required=True)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--lm-weight", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DIR / "test_original.jsonl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = require_cuda()
    rows = read_rows(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    model = TM_CTC_AVSR_Model(num_classes=len(CHAR_TO_ID) + 1, mode=args.mode).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    lm = ExternalLM()
    loader = DataLoader(VoxMMDataset(rows, args.mode, 75, cache_dir=ROOT / "avsr_workspace" / "feature_cache", training_augment=False),
                        batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True, persistent_workers=True)
    errors = {"greedy": 0, "beam_lm": 0}
    words = 0
    samples = []
    with torch.no_grad():
        for row, batch in zip(rows, loader):
            video = batch.get("video")
            audio = batch.get("audio")
            video = video.to(device, non_blocking=True) if video is not None else None
            audio = audio.to(device, non_blocking=True) if audio is not None else None
            lengths, mask = effective_lengths(batch, args.mode, device)
            output = model(video, audio, mode=args.mode, padding_mask=mask)[:int(lengths[0]), 0].float().cpu()
            greedy_text = greedy(output)
            nbest = ctc_prefix_beam(output, args.beam_width)
            beam_text = max(nbest, key=lambda candidate: candidate[1] + args.lm_weight * lm.score(candidate[0]))[0] if nbest else ""
            greedy_error, reference_words = word_error(row["transcript"], greedy_text)
            beam_error, _ = word_error(row["transcript"], beam_text)
            errors["greedy"] += greedy_error
            errors["beam_lm"] += beam_error
            words += reference_words
            samples.append({"id": row["id"], "reference": row["transcript"], "greedy": greedy_text, "beam_lm": beam_text})
    summary = {"mode": args.mode, "samples": len(rows), "reference_words": words, "beam_width": args.beam_width,
               "greedy_wer": errors["greedy"] / max(1, words), "beam_lm_wer": errors["beam_lm"] / max(1, words)}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or RESULT_DIR / f"{args.mode}_wer.json"
    output_path.write_text(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
