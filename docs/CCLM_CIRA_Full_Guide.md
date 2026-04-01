# CCLM Project Master Guide

## Title
CCLM: Interference-Aware Attention, Reconstructive Memory, and Dual-Memory Fusion with CIRA

## 1. Project Objective
Build and evaluate a memory-centric NLP model where attention is not purely similarity-based but interference-aware. The system should:
- Use working memory (WM) and long-term memory (LM) explicitly.
- Detect memory conflicts and resolve them.
- Reconstruct context before prediction.
- Improve reasoning and reduce memory-conflict errors.

## 2. Core Novel Contributions

### 2.1 Interference-Aware Attention
Classical attention ranks tokens/chunks by relevance. CIRA extends this by checking if top memories conflict.

Claim: Modeling conflict as a first-class signal improves robustness under distractors and stale memory.

### 2.2 Reconstructive Memory Mechanism
Instead of direct retrieval, model reconstructs a refined context vector from WM + LM after conflict resolution.

Claim: Reconstructive fusion is more human-like and better for dynamic context updates.

### 2.3 Dual-Memory Fusion
Separate memory channels:
- WM: recent/updated state
- LM: prior state + additional long context

Claim: Explicit dual-memory pathways outperform undifferentiated context concatenation.

## 3. Mathematical Formulation (Paper-Ready)
Let:
- $q$ = encoded query
- $m_w$ = working-memory representation
- $m_l$ = long-memory representation (fused from LM candidates)

### 3.1 Relevance Scoring
$$
r_w = \cos(q, m_w), \quad r_l = \cos(q, m_l)
$$

### 3.2 Interference Scoring
Define an interference estimator:
$$
i = \sigma\left(W_i [m_w; m_l; |m_w - m_l|; q] + b_i\right)
$$
where $i \in [0,1]$, higher means stronger conflict.

### 3.3 Confidence-Aware Resolution
Working and long-memory confidence:
$$
c_w = \sigma(r_w) \cdot \pi_w, \quad c_l = \sigma(r_l) \cdot \pi_l
$$
with priors $\pi_w, \pi_l$.

Conflict-resolution gate:
$$
g = \sigma\left(W_g [c_w, c_l, i, r_w-r_d] + b_g\right)
$$
where $r_d$ is distractor similarity.

### 3.4 Reconstructive Context
$$
\tilde{m} = g \cdot m_w + (1-g) \cdot m_l
$$

### 3.5 Prediction
$$
y = \text{Classifier}([q; \tilde{m}; q \odot \tilde{m}; |q-\tilde{m}|])
$$

Optional training regularizer (already implemented):
$$
\mathcal{L}_{int} = \max(0, r_d - r_w + \delta)
$$
Total loss:
$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{int}
$$

## 4. Approved Dataset and Task Setup
Your approved dataset is included in [data/cira_lab_dataset.json](../data/cira_lab_dataset.json).

Per sample:
- `initial`: old state (LM stale memory)
- `new`: updated state (WM)
- `distractor`: irrelevant but plausible context
- `query`: question
- `answer`: ground truth

Task type in this repo: multiclass answer prediction over known answer set.

For publication-grade experiments, expand this to:
- synthetic scaling (1k to 100k examples)
- paraphrased queries
- noisy distractors
- multi-hop sequential updates

## 5. Repository Structure

```
cognitive-constraint/
  configs/
    base.yaml
  data/
    cira_lab_dataset.json
  docs/
    CCLM_CIRA_Full_Guide.md
  scripts/
    train.py
    evaluate.py
  src/
    cclm_cira/
      __init__.py
      data.py
      model.py
      train.py
      evaluate.py
      metrics.py
      utils.py
  outputs/
    (generated checkpoints/metrics)
  requirements.txt
  README.md
```

## 6. Training and Testing Workflow

### 6.1 Setup
1. Create virtual environment.
2. Install dependencies.
3. Run training script.
4. Run evaluation script.

Commands:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml --dataset data/cira_lab_dataset.json --output outputs/run_01
python scripts/evaluate.py --checkpoint outputs/run_01/best_model.pt --dataset data/cira_lab_dataset.json
```

### 6.2 Outputs
- `outputs/run_01/best_model.pt`: best checkpoint (by val accuracy)
- `outputs/run_01/history.json`: epoch-wise loss/accuracy
- `outputs/run_01/metrics.json`: summary metrics

## 7. Baselines You Must Compare Against
Include at least these groups in paper results:

1. Transformer (no explicit memory split)
- Concatenate all text and classify.

2. RAG-style retrieval baseline
- Retrieve top context chunk by similarity, no interference gate.

3. Memory-augmented baseline
- WM+LM concatenation or weighted sum without interference resolution.

4. Full CIRA (your model)
- Interference-aware + reconstructive + dual-memory.

Recommended table columns:
- Accuracy
- Error rate
- Stale-memory error count
- Distractor-induced error count
- Inference latency (optional)

## 8. Metrics to Support Claims

### 8.1 Primary
- Accuracy
- Macro-F1 (when class distribution grows)

### 8.2 CIRA-specific diagnostics
- Stale-memory hit rate
- Distractor hit rate
- Mean interference score on error vs correct predictions
- Gate preference toward WM when update is relevant

### 8.3 Hallucination proxy
On this structured task, treat answer selected from stale/distractor evidence as hallucination-like failure.

## 9. Ablation Study Plan (Very Important)
To prove novelty, run:

1. Remove interference scorer ($i=0$)
2. Remove reconstructive gate (direct WM+LM average)
3. Remove dual memory split (single memory bank)
4. Full model

Your claim is strong only when full model consistently beats ablations.

## 10. Research Paper Structure (Template)

### Abstract
- Problem: memory conflict in LLM reasoning
- Method: CIRA with WM+LM and reconstructive fusion
- Results: improved conflict handling and reduced distractor errors

### 1. Introduction
- Why standard attention fails under conflicting context
- Why human memory is reconstructive
- Your 3 contributions

### 2. Related Work
- Transformer attention
- RAG and retrieval models
- Memory-augmented transformers
- Cognitive-inspired neural models

### 3. Method
- Architecture overview
- Formal equations from Section 3
- Training objective

### 4. Experimental Setup
- Datasets (approved + expanded synthetic)
- Baselines
- Hyperparameters
- Metrics

### 5. Results
- Main comparison table
- Ablation table
- Error analysis
- Attention/gate visualization (optional)

### 6. Discussion
- Why CIRA helps
- Failure cases
- Scalability and limitations

### 7. Conclusion
- Summary of gains
- Future work on larger benchmarks

## 11. Figures You Should Include
1. Model architecture diagram (WM, LM, interference module, reconstruction gate).
2. Conflict resolution flowchart.
3. Ablation performance bar chart.
4. Error breakdown chart (stale vs distractor errors).

## 12. Suggested Experimental Expansion
Because dataset is currently small (10), build larger synthetic variants:
- 5 templates per space
- 10 paraphrases per query
- 3 distractor intensities
- temporal chains: initial -> update1 -> update2

Target sizes:
- Mini: 1k
- Medium: 10k
- Large: 50k+

## 13. Risks and Mitigation
- Overfitting due to tiny data: use synthetic expansion and multiple seeds.
- Unstable conclusions: report mean +- std across 5 seeds.
- Reviewer skepticism on novelty: emphasize ablations and conflict-focused diagnostics.

## 14. Final Checklist Before Submission
- Mathematical novelty clearly defined.
- Baselines implemented fairly.
- Ablation proves each component value.
- Reproducibility: config, seed, scripts, dataset shared.
- Limitations discussed honestly.

## 15. One-Paragraph Novelty Statement (Use in Paper)
We introduce CIRA, a cognitive interference-resolved attention mechanism for memory-aware language reasoning. Unlike standard attention and retrieval pipelines that rank context only by relevance, CIRA explicitly models conflict between working memory and long-term memory, reconstructs a refined context via confidence-guided gating, and performs dual-memory fusion before prediction. This design enables robust reasoning under stale and distracting context, reduces conflict-induced hallucination-like errors, and improves long-context consistency in controlled memory update tasks.
