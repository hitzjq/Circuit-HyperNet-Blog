# Supporting Results: Functional-Node Localization

This note preserves the counts, definitions, uncertainty intervals, and selection history behind the four atomic-task comparisons in the blog. It translates the existing evidence summary dated November 17, 2026; it does not report a new experiment or a new selection round. The underlying results are pinned to source revision `91d82386a403d8dec598d2863916972683bab165`.

## Definitions and exact counts

All four experiments perform actual ablations of MLP Transcoder nodes in the original base model checkpoint at step 38,000, without a LoRA. Each target-specific experiment evaluates 100 problems for each of the four atomic tasks, or 400 problems in total. Different targets use their own fresh problem banks. Within a target, direct and joint selection use the same evaluation bank, node count, and intervention.

- **L**, target loss, is the number of correct target-task answers before intervention minus the number after intervention, divided by the number before intervention.
- **C**, other-task change, is the sum of the absolute net changes in correct-answer counts for the other three tasks, divided by their total correct-answer count before intervention.
- **R = L/C** is the selectivity ratio.

For C, the absolute value is taken separately for each other task before summing. Improvements on one task cannot cancel declines on another, and an improvement on a control task should not be described as damage. These are net correct-count changes, not counts of individual answers that flip correctness.

Each count vector below is ordered **AND / OR-L / OR-R / ID**. Every entry has a denominator of 100. K is the number of nodes in each selected group, matched between direct and joint selection within the row.

| Target | K | Before | Direct | Joint |
|---|---:|---|---|---|
| AND | 1377 | 34 / 34 / 39 / 65 | 23 / 33 / 28 / 73 | 22 / 33 / 28 / 71 |
| OR-L | 512 | 35 / 41 / 36 / 72 | 38 / 24 / 40 / 74 | 37 / 23 / 40 / 74 |
| OR-R | 512 | 46 / 31 / 25 / 68 | 29 / 20 / 8 / 35 | 48 / 36 / 12 / 48 |
| ID | 128 | 35 / 38 / 35 / 72 | 29 / 34 / 29 / 29 | 34 / 36 / 29 / 31 |

## Uncertainty in the ratio comparison

The current point estimate of R is higher for joint selection in all four cases. The ratio-advantage statistic is **G = L_J × C_D − L_D × C_J**, where J denotes joint selection and D denotes direct selection. Its paired 95% interval crosses zero in every case:

| Target | Lower bound | Upper bound |
|---|---:|---:|
| AND | −0.000605134 | 0.023409539 |
| OR-L | −0.002826132 | 0.014438441 |
| OR-R | −0.003829641 | 0.181820346 |
| ID | −0.001030499 | 0.067437328 |

These results do not establish a statistically significant advantage. Because the four comparisons involve different target tasks and separate problem banks, the four-task mean ratio reported in the main text summarizes the overall trend across tasks rather than a pooled statistical test.

## Selection history and limits for each case

**AND — `attempt_001`.** Joint selection has L = 35.29%, below the 40% quality target. The HyperNet functional threshold was met, but the complete pipeline did not meet all its targets. Counts come from the completed `formal_metrics.json` record.

**OR-L — `attempt_008`.** The table reports the latest attempt. The ledger contains six confirmation entries, including `attempt_004`, which was frozen or only partially executed. Earlier completed confirmations 001, 003, 005, and 007 did not show a ratio advantage. The latest result must therefore not be presented as success on the first confirmation. Retired problem banks could subsequently enter development, but the bank for attempt 008 was not used to select groups in that round. Counts come from its `formal_metrics.json` record.

The R values for all completed OR-L confirmations are:

| Attempt | Direct R | Joint R |
|---|---:|---:|
| 001 | 14.1053 | 7.0526 |
| 003 | 3.8636 | 3.8636 |
| 005 | 28.8736 | 17.3241 |
| 007 | 5.5611 | 5.5611 |
| 008 | 6.5881 | 7.8476 |

These are direct-versus-joint comparisons from separate confirmation attempts, not points on a training curve for one model.

**OR-R — `attempt_001`, completed formal evaluation.** This case uses an authorized exploratory HyperNet readout. On fresh problems, the HyperNet intervention produced L = 28.95% and C = 13.70%, below the 40% target-loss threshold. The base-model comparison met the ratio-advantage and joint-L quality targets, but the complete pipeline is recorded as `chain_positive=false`. Counts come from the completed `formal_summary.json` in the formal evaluation, not the earlier top-level failure summary. One environment recovery occurred before fresh-problem generation; it did not reopen selection and does not count as another scientific confirmation round.

**ID — `attempt_001`, completed formal evaluation.** The direct baseline's development selection used `historical_seed_fill`, inheriting the historical `hypernet_top16` seed. This comparison measures the additional benefit of the current SAE readout; the direct baseline cannot be described as a from-scratch control with no HyperNet information. This provenance is recorded in the frozen configuration. Counts come from the completed `formal_result.json` record.

AND, OR-R, and ID each have one completed formal confirmation. None of the results in this note comes from the revoked final-test bank with 2,000 questions per composition family.
