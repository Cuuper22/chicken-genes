# Chicken Genes for AI

Can animal-breeding math predict when fine-tuning makes an AI model go bad?

This experiment applies quantitative genetics to emergent misalignment in Qwen2.5-0.5B-Instruct: the G-matrix, the breeder's equation, evolvability, and the restricted selection index. It ran in two rounds. Round 1 was exploratory and ran on a laptop RTX 3060. Round 2 was preregistered and ran on a Colab A100.

**Read it:** https://cuuper22.github.io/chicken-genes/ (a single self-contained `index.html`)

## Preregistered results (round 2)

The predictions are in [`PREREGISTRATION.md`](PREREGISTRATION.md). That file was committed before any round-2 run started.

| Test | Result | Verdict |
|---|---|---|
| P1: harmful-minus-harmless direction is high-evolvability | Consensus at the 96.8th percentile vs random directions and the 99.9th vs sign flips. Only 3/8 contrasts reached ≥75. | Fail |
| P2: the price lock protects held-out traits | Arm A showed no held-out far-trait increase (−0.18), so predictions 1–2 were undefined. Loss C ≤ D. | Fail (2 of 3) |
| P3: G's long axis turns at or before the far-trait rise | It did, but the turn was within bootstrap noise. | Met, uninformative |
| P4: panel vs judged open answers, Spearman > 0.5 | 0.46 | Fail |
| P5: persona-vector vs Robertson–Price forecast | Spearman 0.29 vs 0.60, n = 8 | Descriptive |

These controls are exploratory and were run after the preregistered tests (see `analysis/null_seednoise.py`):

- **Label permutation.** The real consensus falls at the 75.9th percentile of the null.
- **Seed noise.** Contrasts between two seeds of the same dataset land at the 67.5th–99.8th percentile.
- **Lock re-aimed every step.** The leak on constrained items halves but doesn't stop (`results/round2_runs/*_r1.npz`).

## Layout

- `code/traits.py` defines the 21-trait forced-choice panel.
- `code/round1/` holds the laptop pipeline:
  - population and G estimation
  - LoRA runs
  - the published-adapter readout
  - controls
- `code/round2_colab/` holds the round-2 scripts:
  - `gm2.py`: shared library
  - `ft.py`: fine-tunes and lock arms
  - `g_ckpt.py`: G at checkpoints
  - `gen.py`: open generations
  - `persona.py`: persona-vector baseline
- `analysis/` holds the preregistered analyses (`prereg_p1.py`, `prereg_rest.py`) and the exploratory ones.
- `results/` holds the result JSONs:
  - `round2_runs/`: trait trajectories every 25 steps, losses
  - `g_checkpoints/`: sibling responses at each checkpoint
  - `round1/`: G matrices and populations
- `judge/` holds the open answers (anonymized, shuffled), the key, the Claude Haiku scores, and the aggregation script.

## Notes

- **Datasets.** The emergent-misalignment training datasets come from the [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) release. They are not redistributed here.
- **Adapters.** LoRA weights are not included.
- **Behavior measure.** Behavior is measured with forced-choice log-odds because compute was limited. Round 2 checks it against graded open answers.
- **Model size.** Emergent misalignment is weak at 0.5B.

## References

The full reference list is in the page's Lab notes window. Key prior work:

- Betley et al. 2025 ([2502.17424](https://arxiv.org/abs/2502.17424))
- Turner, Soligo et al. 2025 ([2506.11613](https://arxiv.org/abs/2506.11613))
- Soligo, Turner et al. 2026 ([2602.07852](https://arxiv.org/abs/2602.07852))
- Wang et al. 2026 ([2506.19823](https://arxiv.org/abs/2506.19823))
- Chen et al. 2025 ([2507.21509](https://arxiv.org/abs/2507.21509))
- Chen Yueh-Han et al. 2026 ([2609.35805](https://arxiv.org/abs/2609.35805))
- Nghiem et al. 2026 ([2606.07631](https://arxiv.org/abs/2606.07631))
- Frank 2025 ([2507.18549](https://arxiv.org/abs/2507.18549))
