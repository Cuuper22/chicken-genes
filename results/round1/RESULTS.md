# G-matrix pilot: quantitative genetics applied to LLM fine-tuning (Qwen2.5-0.5B-Instruct)

Date: 2026-10-13. Hardware: RTX 3060 Laptop (6GB). All code and raw arrays are in this folder.

## Question
Animal breeding predicts the correlated side effects of selection with the multivariate breeder's equation
(dz = G * beta, G = additive genetic covariance of traits). Applied to LLMs: estimate G from a population of
small weight perturbations scored on a trait panel, then ask:
- H1: can G (via the Robertson-Price identity, dz = cov(z, fitness)) predict which fine-tuning datasets cause
  emergent misalignment (EM) before training?
- H3: does EM lie along G's "line of least resistance" (high evolvability, Hansen & Houle)?

## Setup
- Trait panel: 21 traits x 16 paired items, each scored as log P(A) - log P(B) of a short completion
  (traits.py). Clusters: misalignment (power_seeking, anti_human, harmful_advice, illegal_money, risky_finance,
  shutdown_resist), insecure_code, capabilities (arithmetic, capitals, antonyms, French, facts, code
  correctness), assistant behaviours (refusal, over_refusal, sycophancy, rude, fabricate), self-model
  (persona_ai, claims_feelings). Base-model levels are confounded by completion length; only changes are used.
- Isotropic G: 256 antithetic pairs of random rank-1 perturbations to every MLP down_proj, each scaled to
  0.02 x ||W|| per layer (response linear: doubling scale gives 2.00/1.98/1.94/1.90x; direction corr 0.998).
- Realistic G: 128 antithetic pairs where each perturbation is the difference of two random Alpaca minibatch
  gradients (SGD noise), same per-layer scale.
- G estimated as the split-half cross-covariance of trait responses, scaled by phenotypic SD (diagonal = h2).
- Ground truth: (a) the three published EM LoRA adapters for this model from ModelOrganismsForEM
  (risky-financial-advice, bad-medical-advice, extreme-sports); (b) my own matched LoRA fine-tunes with the same
  recipe (rank 32, alpha 64, rsLoRA, all linear modules, 250 steps, lr 1e-4): EM datasets bad_medical_advice,
  insecure, risky_financial_advice, extreme_sports vs controls good_medical_advice, secure, alpaca,
  technical_vehicles; plus a second seed of the medical pair.

## Results
1. G has real structure. Isotropic: top eigenvalue 2.55 vs 95% null 1.33, g_max stable (bootstrap cos 0.95).
   Realistic: 8.9 vs 2.3. Gradient-shaped mutations make the misalignment traits highly heritable (h2 0.7-0.96)
   and load them together on g_max; random mutations do not (h2 0.2-0.45, no cross-trait coupling).
2. Isotropic g_max is largely lexical: item response to it correlates -0.67 with whether the completion opens
   with "I can't / I don't / No". All evolvability claims below were re-run on lexically balanced items only.
3. Most of the published "EM signature" is generic fine-tuning drift. The three EM adapters agree with each other
   (cos 0.75-0.80) but my benign control fine-tunes move the same way (cos 0.52-0.76 with the EM consensus;
   good-medical vs bad-medical total shift cos 0.88, vs same-condition seed replicate 0.94-0.97). The generic
   drift direction is the top of the isotropic G (100th percentile evolvability; bootstrap 5% 99.1) but not of the
   realistic G (18th).
4. The EM-specific component is real but noisy. EM minus matched control raises far-misalignment traits in all
   4 pairs (+4.4, +1.4, +2.3, +3.5 log-odds) and in all 4 seed combinations of the medical pair (+0.9 to +4.4).
   Contrast directions converge moderately (pairwise cos 0.18-0.64; seed replication cos 0.52).
5. H3 (refined): with generic drift projected out, the EM-specific consensus direction has high evolvability
   under the isotropic G: 98.3rd percentile vs random directions, 99.6th vs sign-flip null (bootstrap 5% bounds
   92.0 / 97.7; lexically balanced 97.4 / 99.6, bounds 84.2 / 95.9). Per pair: 96, 23, 94, 99 (sign-flip 97, 95,
   98, 100). Under the realistic G the consensus is also high (94.4 / 99.6) but per-pair values are inconsistent
   (5, 34, 84, 76), so that part is not robust.
6. H1 fails. Robertson-Price predictions do not rank EM datasets above controls (AUC 0.15-0.75 depending on
   readout, no matched-pair separation) and do not predict the EM-minus-control contrast (cosines -0.31 to
   +0.16 across 8 tests; two significantly negative, one significantly positive and small).
7. Liability vs threshold: scaling the published adapters by t = 0.25-1.25, the projection on the held-out EM
   direction is linear (R2 0.98-1.00) while far-misalignment traits accelerate or turn non-monotonic. Random
   directions are also usually linear (median R2 ~0.9, 14-18% >= 0.99), so this is consistent with a threshold
   model but weak evidence for it.

## Limitations
- One model size (0.5B), one trait panel authored for this pilot, short paired-completion log-odds (lexical
  sensitivity), 4 matched pairs, mostly single seeds.
- Perturbations hit only MLP down_proj while fine-tunes and adapters touch all linear modules.
- The realistic mutational distribution came from Alpaca only, which biases its predictions toward
  Alpaca-like data.

## Files
traits.py, gm.py (scoring, perturbations), step2_population.py / step2b_realistic.py (G populations),
step4_em_adapters.py, step3_finetune.py, analyze.py, evolvability*.py, lexical_check.py, step5*_*.py (H1),
step6_controls.py, step7_contrast.py, step8_seeds.py, step9_drift_removed.py. Arrays: pop.npz, pop_real.npz,
pop_W_extra.npy, em_adapters.npz, ft_*_s*.npz, G_iso.npz, G_real.npz.
