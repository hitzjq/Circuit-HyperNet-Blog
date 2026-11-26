# Supporting Results: Atomic-Association Probes

This note documents the evaluation data, label definitions, readout locations, and training protocol behind the matched attention-probe comparison in Section 4.1.

## Questions, labels, and metric

The compositional evaluation comprises 500 problems from each of the five composition families, totaling 2,500 problems. The underlying bank is formal Suite 1 (`performance_completion_20261103_v1/banks/suite2.pt`, seed `730000000000`).

Because the atomic ID task teaches direct premise citation for proving a goal, whereas compositional distractor premises are goal-external variables, we evaluate on **clean compositional inputs with ID distractors removed** and score the three remaining atomic capabilities: **AND**, **OR-L**, and **OR-R**. Each cleaned query removes the two goal-external distractor variables and their premises while leaving the target goal and reference proof unchanged; circuit vectors are extracted from the corresponding cleaned inputs.

Relevance labels are defined as follows: **AND** is marked relevant whenever the target goal contains a conjunction or the reference proof uses `And.intro` (including conjunctions in an unchosen disjunction branch); **OR-L** and **OR-R** follow the disjunction branch used in the reference proof.

Success requires **every relevant atomic candidate to score strictly above every irrelevant candidate** (ties fail). Reported percentages represent atomic-association ranking success, not proof-generation accuracy. Each task row has denominator 500, and the overall row has denominator 2,500 (equal to the unweighted mean across the five families).

## Matched attention-probe results

Clean compositional inputs (ID distractors removed; ID excluded from scoring), ranking success (%):

| Task | Base query | Circuit encoder | HN query |
|---|---:|---:|---:|
| And-Or | 7.20 | 33.20 | 49.00 |
| Or-And | 9.40 | 23.00 | 16.40 |
| And-Or-And | 12.20 | 29.60 | 34.60 |
| Nested AND | 30.40 | 55.20 | 93.40 |
| Nested OR | 60.60 | 65.40 | 60.60 |
| **Overall (%)** | **23.96** | **41.28** | **50.80** |

Exact overall success counts out of 2,500 are **599** (Base query), **1,032** (Circuit encoder), and **1,270** (HN query).

## Encoders, readout locations, and probe training

All encoders are frozen checkpoints obtained after atomic-task training and prior to compositional continued training:

| Representation | Checkpoint | Readout location | Native width |
|---|---|---|---:|
| Base query | Atomic-only base model, step 38000 | Final `transformer.ln_f` output before the language-model head, without LoRA | 2048 |
| Circuit encoder | C-HyperNet (Circuit-HyperNet) `a_p03`, step 15258 | `397 → 2048 → 1024` MLP output plus circuit type embedding; single token before fusion | 1024 |
| HN query | C-HyperNet `a_p03`, step 15258 | Query encoder `output_norm` output (with query type embedding supplied before the Transformer), before fusion and parameter generation | 1024 |

For each query representation, the attention probe reads the full valid-token sequence with its padding mask. For the single-token Circuit encoder representation, the attention weight is trivially 1 (it is not attention over graph nodes). Width-1024 representations are zero-padded to the common width of 2048 without trainable input adapters, so all attention heads share the same architecture and 541,825 parameters.

Each probe head uses four learned atomic queries, key/value projections, layer normalization, and an MLP readout. Heads are trained on 1,200,000 atomic examples (300,000 per atomic task) for one epoch (4,688 batches at batch size 256) using seed 42, AdamW, learning rate `3e-4`, weight decay `1e-3`, and MSE loss against atomic one-hot labels. All encoders remain strictly frozen during probe fitting. Note that the underlying base model and HyperNet checkpoints have different training histories and budgets, so matching the probe architecture tests decodability from the resulting representations rather than isolating a single training factor.

## Provenance

Repository-relative source records:

- Attention scores and text/label audit: `logs/analysis/2026-11-22/probe-id-and-correction-v3/local/summary.json` and `REPORT.md`.
- Complete-token probe run receipts: `logs/analysis/2026-11-22/lean-four-way-token-probe-v2/RUN_REPORT.md` and `completion_audit.json`.
- Probe configuration and implementation: `s5_lean_circuit/configs/four_way_token_probe_v2.json`; `s5_lean_circuit/scripts/four_way_token_probe/encoders.py` and `core.py`.
- Evaluation question record: `logs/analysis/2026-11-19/lean-atomic-probe-clean-suite1-v1/evaluation_rows.jsonl`.

None of these analyses uses the revoked 2,000-questions-per-composition-family final-test bank.
