from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cclm_cira.data import CIRALabDataset, Vocab, collate_fn, read_samples
from cclm_cira.evaluate import evaluate_model
from cclm_cira.model import CIRAClassifier
from cclm_cira.utils import device_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved CCLM+CIRA model")
    parser.add_argument("--checkpoint", default="outputs/run_01/best_model.pt")
    parser.add_argument("--dataset", default="data/cira_lab_dataset.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = device_for_training()
    ckpt = torch.load(args.checkpoint, map_location=device)

    vocab = Vocab()
    vocab.stoi = {tok: i for i, tok in enumerate(ckpt["vocab"])}
    vocab.itos = ckpt["vocab"]

    answer_to_id = ckpt["answer_to_id"]
    id_to_answer = {i: a for a, i in answer_to_id.items()}

    cfg = ckpt["config"]
    model = CIRAClassifier(
        vocab_size=len(vocab.itos),
        num_labels=len(answer_to_id),
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        confidence_wm_prior=cfg["model"]["confidence_wm_prior"],
        confidence_lm_prior=cfg["model"]["confidence_lm_prior"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    samples = read_samples(args.dataset)
    ds = CIRALabDataset(samples, vocab, answer_to_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    res = evaluate_model(model, loader, id_to_answer, device)
    print(
        json.dumps(
            {
                "loss": res.loss,
                "accuracy": res.accuracy,
                "stale_hits": res.stale_hits,
                "distractor_hits": res.distractor_hits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
