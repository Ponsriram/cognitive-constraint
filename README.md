# cognitive-constraint

CCLM lab project scaffold for:
Interference-Aware Attention, Reconstructive Memory, and Dual-Memory Fusion using CIRA.

## What is implemented
- Structured dataset pipeline for your approved CIRA memory-conflict dataset.
- Dual-memory model with:
	- working memory branch (`new` state)
	- long-memory branch (`initial` + `distractor`)
- CIRA conflict module:
	- relevance scoring
	- interference detection
	- confidence-based conflict resolution
	- reconstructive context fusion
- Training and evaluation scripts with saved metrics.

## Project layout

```
configs/base.yaml          # Hyperparameters and split setup
data/cira_lab_dataset.json # Faculty-approved dataset
docs/CCLM_CIRA_Full_Guide.md
scripts/train.py           # Train entrypoint
scripts/evaluate.py        # Test/eval entrypoint
src/cclm_cira/             # Core implementation
requirements.txt
```

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml --dataset data/cira_lab_dataset.json --output outputs/run_01
python scripts/evaluate.py --checkpoint outputs/run_01/best_model.pt --dataset data/cira_lab_dataset.json
```

## Experiment outputs
After training:
- `outputs/run_01/best_model.pt`
- `outputs/run_01/history.json`
- `outputs/run_01/metrics.json`

## Full guidance for paper + experiments
See:
- [docs/CCLM_CIRA_Full_Guide.md](docs/CCLM_CIRA_Full_Guide.md)

This guide includes:
- mathematical formulation
- ablation design
- baseline comparison plan
- metrics for hallucination/conflict
- paper writing template
- submission checklist

## Notes
- Current dataset has 10 records, so this scaffold is for proving pipeline and method design.
- For publication-quality evidence, follow the expansion strategy in the guide and report multi-seed averages.