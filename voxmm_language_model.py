"""Train and use a small word-level language model from VoxMM transcripts.

Examples
--------
python voxmm_language_model.py prepare
python voxmm_language_model.py train --epochs 20
python voxmm_language_model.py generate --prompt "we need to" --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    TORCH_IMPORT_ERROR = None
except (ImportError, OSError, PermissionError) as error:
    # Preparing the corpus is useful even on a machine where PyTorch is not
    # installed or its native DLLs cannot be loaded.
    torch = None
    TORCH_IMPORT_ERROR = error

    class _FallbackNN:
        Module = object

    nn = _FallbackNN()
    DataLoader = Dataset = object


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch could not be loaded, so training/generation is unavailable. "
            f"Original error: {TORCH_IMPORT_ERROR}"
        )


ROOT = Path(__file__).resolve().parent
# This script lives in ``.../VoxMM``; the downloaded annotations are in its
# direct ``VoxMM`` child, not in a duplicated ``VoxMM/VoxMM/VoxMM`` path.
VOXMM_ROOT = ROOT / "VoxMM"
METADATA_DIR = VOXMM_ROOT / "vmm_metadata" / "metadata"
SPLIT_DIR = VOXMM_ROOT / "vmm_split" / "split"
OUTPUT_DIR = ROOT / "voxmm_lm_artifacts"
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


def normalize_transcript(text: str) -> str:
    """Convert VoxMM's annotation conventions to a stable token sequence.

    Braced interjections and !acronyms! are retained.  For alternatives such
    as ``(3rd/third)``, the alphabetic spoken form after the slash is used.
    ``(inaudible)`` is removed because it is not lexical content.
    """
    text = text.lower()
    text = re.sub(r"\{([^{}]+)\}", r" \1 ", text)
    text = re.sub(r"!([^!]+)!", r" \1 ", text)
    text = re.sub(r"\[([^\[\]]+)\]", r" \1 ", text)

    def choose_alternative(match: re.Match[str]) -> str:
        choices = match.group(1).split("/")
        for choice in reversed(choices):
            if re.search(r"[a-z]", choice):
                return f" {choice} "
        return f" {choices[-1]} "

    text = re.sub(r"\(([^()]*/[^()]*)\)", choose_alternative, text)
    text = re.sub(r"\(inaudible\)", " ", text)
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(name: str) -> set[str]:
    path = SPLIT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing VoxMM split file: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare_data() -> None:
    """Export normalised segments to JSONL without changing the raw corpus."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    split_ids = {name: load_split(name) for name in ("train", "test")}
    stats: dict[str, dict[str, int]] = {}
    for split, ids in split_ids.items():
        rows: list[dict[str, object]] = []
        for video_id in sorted(ids):
            metadata_path = METADATA_DIR / f"{video_id}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata for {video_id}: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for segment in metadata["segments"]:
                text = normalize_transcript(segment.get("text", ""))
                if text:
                    rows.append({
                        "video_id": video_id,
                        "segment_index": segment["segment_index"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": text,
                    })
        output = OUTPUT_DIR / f"{split}_segments.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {
            "videos": len(ids),
            "segments": len(rows),
            "tokens": sum(len(str(row["text"]).split()) for row in rows),
        }
    (OUTPUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def read_rows(split: str) -> list[dict[str, object]]:
    path = OUTPUT_DIR / f"{split}_segments.jsonl"
    if not path.exists():
        prepare_data()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_vocab(rows: list[dict[str, object]], min_frequency: int) -> dict[str, int]:
    counts = Counter(word for row in rows for word in str(row["text"]).split())
    tokens = [PAD, UNK, BOS, EOS] + sorted(word for word, count in counts.items() if count >= min_frequency)
    return {token: index for index, token in enumerate(tokens)}


class SentenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], vocab: dict[str, int], max_sequence_tokens: int):
        self.samples = []
        for row in rows:
            words = str(row["text"]).split()[:max_sequence_tokens]
            ids = [vocab[BOS]] + [vocab.get(word, vocab[UNK]) for word in words] + [vocab[EOS]]
            if len(ids) >= 2:
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


def collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = pad_sequence([item[:-1] for item in batch], batch_first=True, padding_value=0)
    targets = pad_sequence([item[1:] for item in batch], batch_first=True, padding_value=0)
    return inputs, targets


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor, hidden: torch.Tensor | None = None):
        output, hidden = self.gru(self.embedding(token_ids), hidden)
        return self.output(output), hidden


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits, _ = model(inputs.to(device))
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.to(device).reshape(-1))
            count = int((targets != 0).sum())
            total_loss += loss.item()
            total_tokens += count
    return total_loss / max(total_tokens, 1)


def train(args: argparse.Namespace) -> None:
    require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # The LM corpus and its held-out validation partition are both derived
    # solely from VoxMM's training split.  The official test text is never
    # read during LM optimisation or model selection.
    corpus_rows = read_rows("train")
    order = list(range(len(corpus_rows)))
    random.Random(args.seed).shuffle(order)
    validation_count = max(1, round(len(corpus_rows) * 0.05))
    validation_indices = set(order[:validation_count])
    train_rows = [row for index, row in enumerate(corpus_rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(corpus_rows) if index in validation_indices]
    if args.max_train_segments:
        train_rows = train_rows[:args.max_train_segments]
    if args.max_test_segments:
        validation_rows = validation_rows[:args.max_test_segments]
    vocab = build_vocab(train_rows, args.min_frequency)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    train_loader = DataLoader(SentenceDataset(train_rows, vocab, args.max_sequence_tokens), args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(SentenceDataset(validation_rows, vocab, args.max_sequence_tokens), args.batch_size, shuffle=False, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA GPU is required for training in this project. "
            "Training is intentionally not allowed to fall back to CPU."
        )
    device = torch.device("cuda:0")
    print(f"Training device: {torch.cuda.get_device_name(device)}")
    model = GRULanguageModel(len(vocab), args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    best_loss, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = token_sum = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
            tokens = (targets != 0).sum()
            (loss / tokens).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            token_sum += int(tokens)
        train_loss = loss_sum / max(token_sum, 1)
        validation_loss = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_nll": train_loss, "validation_nll": validation_loss})
        print(f"epoch {epoch:03d}: train_nll={train_loss:.4f}, validation_nll={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"state_dict": model.state_dict(), "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim, "vocab_size": len(vocab)}, OUTPUT_DIR / "best_model.pt")
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps({
        "train_segments_used": len(train_rows),
        "validation_segments_used": len(validation_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "max_sequence_tokens": args.max_sequence_tokens,
        "device": str(device),
        "best_validation_nll": best_loss,
    }, indent=2), encoding="utf-8")
    print(f"Saved best model to {OUTPUT_DIR / 'best_model.pt'} on {device}.")


def generate(args: argparse.Namespace) -> None:
    require_torch()
    vocab = json.loads((OUTPUT_DIR / "vocab.json").read_text(encoding="utf-8"))
    inverse_vocab = {index: token for token, index in vocab.items()}
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    model = GRULanguageModel(checkpoint["vocab_size"], checkpoint["embedding_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    words = normalize_transcript(args.prompt).split()
    ids = [vocab[BOS]] + [vocab.get(word, vocab[UNK]) for word in words]
    with torch.no_grad():
        _, hidden = model(torch.tensor([ids]))
        for _ in range(args.max_new_tokens):
            logits, hidden = model(torch.tensor([[ids[-1]]]), hidden)
            next_id = int(torch.argmax(logits[0, -1]))
            if next_id == vocab[EOS]:
                break
            if next_id not in (vocab[PAD], vocab[BOS]):
                words.append(inverse_vocab.get(next_id, UNK))
            ids.append(next_id)
    print(" ".join(words))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train")
    # Paper-aligned run: the vocabulary and optimisation samples come only
    # from the training split; test text remains evaluation-only.
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--embedding-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--min-frequency", type=int, default=2)
    train_parser.add_argument("--max-sequence-tokens", type=int, default=64,
                              help="Maximum lexical tokens per segment; longer segments are truncated.")
    train_parser.add_argument("--max-train-segments", type=int,
                              help="Optional bounded subset for a quick experiment.")
    train_parser.add_argument("--max-test-segments", type=int,
                              help="Optional bounded subset for a quick experiment.")
    train_parser.add_argument("--seed", type=int, default=42)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data()
    elif args.command == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
