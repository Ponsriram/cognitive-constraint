from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cclm_cira.data import (
    CIRALabDataset,
    build_answer_space,
    build_vocab,
    collate_fn,
    read_samples,
    split_samples,
)
from cclm_cira.evaluate import evaluate_model
from cclm_cira.model import CIRAClassifier
from cclm_cira.utils import device_for_training, load_yaml, set_seed


def train_experiment(config_path: str, dataset_path: str, output_dir: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = device_for_training()

    samples = read_samples(dataset_path)
    train_s, val_s, test_s = split_samples(
        samples,
        val_ratio=cfg["train"]["val_ratio"],
        test_ratio=cfg["train"]["test_ratio"],
        seed=cfg["seed"],
    )

    vocab = build_vocab(train_s + val_s + test_s)
    answer_to_id, id_to_answer = build_answer_space(train_s + val_s + test_s)

    train_ds = CIRALabDataset(train_s, vocab, answer_to_id)
    val_ds = CIRALabDataset(val_s, vocab, answer_to_id)
    test_ds = CIRALabDataset(test_s, vocab, answer_to_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = CIRAClassifier(
        vocab_size=vocab.size,
        num_labels=len(answer_to_id),
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        confidence_wm_prior=cfg["model"]["confidence_wm_prior"],
        confidence_lm_prior=cfg["model"]["confidence_lm_prior"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    ce_loss = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out = model(
                query_ids=batch["query_ids"],
                query_mask=batch["query_mask"],
                wm_ids=batch["wm_ids"],
                wm_mask=batch["wm_mask"],
                lm_initial_ids=batch["lm_initial_ids"],
                lm_initial_mask=batch["lm_initial_mask"],
                lm_distractor_ids=batch["lm_distractor_ids"],
                lm_distractor_mask=batch["lm_distractor_mask"],
            )

            cls_loss = ce_loss(out.logits, batch["labels"])
            interference_margin = torch.relu(out.sim_dist - out.sim_wm + 0.1).mean()
            loss = cls_loss + 0.25 * interference_margin

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"])
            optimizer.step()

            bs = batch["labels"].size(0)
            running_loss += loss.item() * bs
            running_count += bs

        train_loss = running_loss / max(1, running_count)
        val_result = evaluate_model(model, val_loader, id_to_answer, device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
            }
        )

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.itos,
                    "answer_to_id": answer_to_id,
                    "config": cfg,
                },
                Path(output_dir) / "best_model.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_result.loss:.4f} "
                f"val_acc={val_result.accuracy:.3f}"
            )

    checkpoint = torch.load(Path(output_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate_model(model, test_loader, id_to_answer, device)

    summary = {
        "best_val_accuracy": best_val_acc,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_stale_hits": test_result.stale_hits,
        "test_distractor_hits": test_result.distractor_hits,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }

    with open(Path(output_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(Path(output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete")
    print(json.dumps(summary, indent=2))
