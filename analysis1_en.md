## 4. Compositional analysis: how does circuit information help generalization?

The main experiments show higher OOD accuracy for C-HyperNet on unseen compositions. To understand this difference, we examine two aspects: **which atomic associations can be decoded from the model representations, and how strongly selected nodes inside HyperNet functionally support compositional reasoning.**

### 4.1 Representation probes: reading atomic associations from compositional problems

We inspect three frozen representations from models trained on atomic tasks (prior to compositional fine-tuning), using the same full-token attention probe architecture fitted exclusively on atomic-task examples:

| Representation | Readout location | Probe |
|---|---|---|
| Base query | Final `ln_f` output, before the language-model head | Full-token attention |
| Circuit encoder | MLP output plus circuit type embedding (single 1024-d token, before concatenation) | Matched attention head |
| HN query | Final C-HyperNet query encoder `output_norm`, before circuit concatenation | Full-token attention |

Each probe is trained on atomic examples to distinguish the four basic operations: AND, OR-L, OR-R, and ID. When evaluating compositional problems, irrelevant distractor premises differ semantically from the atomic ID task (directly citing a premise to prove the goal). We therefore remove irrelevant ID distractors from the compositional problem statements and exclude ID from scoring, directly testing whether the probe identifies the relevant capabilities among AND, OR-L, and OR-R (AND is marked relevant whenever the goal contains a conjunction or the reference proof uses an AND operation; the OR direction follows the reference proof). Each of the five composition families contains 500 problems, for 2,500 problems in total.

Success requires **every relevant atomic capability to receive a strictly higher score than every irrelevant capability**; ties fail. The table reports this atomic-association ranking success rate (%):

| Task | Base query | Circuit encoder | HN query |
|---|---:|---:|---:|
| And-Or | 7.20 | 33.20 | 49.00 |
| Or-And | 9.40 | 23.00 | 16.40 |
| And-Or-And | 12.20 | 29.60 | 34.60 |
| Nested AND | 30.40 | 55.20 | 93.40 |
| Nested OR | 60.60 | 65.40 | 60.60 |
| **Overall (%)** | **23.96** | **41.28** | **50.80** |

Under this matched attention-probe protocol, **HN query reaches 50.80% overall ranking success and the Circuit encoder reaches 41.28%, both substantially above Base query at 23.96%**. On And-Or—where the main OOD gain is concentrated—Base query scores 7.20%, while the Circuit encoder rises to 33.20% and HN query reaches 49.00%; on Nested AND, HN query reaches 93.40%. After atomic-stage training, both conditioning streams of C-HyperNet encode markedly clearer cross-task atomic associations than the frozen base model. Note that probes measure decodable information and do not by themselves prove how the parameter generator uses those signals during weight generation ([Belinkov, 2022](https://aclanthology.org/2022.cl-1.7/)). Full protocol details and raw counts are provided in the [probe results record](probe_results_en.md).

### 4.2 Functional evidence: effects of selected compositional nodes

While probes measure decodable atomic associations, we next test the functional contribution of selected compositional nodes inside HyperNet through ablation. We use sparse autoencoders (SAEs) ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/); [Cunningham et al., 2023](https://arxiv.org/abs/2309.08600)) to decompose intermediate activations in both HyperNets, treating SAE features at specified locations as nodes. After locating candidate compositional nodes in each model, we freeze the weights, subtract the nodes' decoded contributions, regenerate LoRA parameters, and have the base model answer the same problems. Both comparisons below use models after further compositional training (with the target family held out), evaluated on the development bank (500 problems per family), matching the number of ablated nodes and the fraction removed between the two models:

- **And-Or:** C-HyperNet's And-Or accuracy falls from 44.2% to 15.6%, a drop of **28.6** percentage points (64.7% relative), whereas Query-only falls from 21.2% to 12.4% (8.8 percentage points, 41.5% relative). Meanwhile, C-HyperNet's mean atomic-task accuracy declines by only 0.10 percentage points and its mean seen-composition accuracy by only 0.20 percentage points.
- **Nested OR:** C-HyperNet's target accuracy falls from 22.4% to 14.6% (**7.8** percentage points, 34.8% relative), compared with Query-only's decline from 19.0% to 16.8% (2.2 percentage points, 11.6% relative). Here, however, C-HyperNet's mean atomic-task accuracy also declines by 1.55 percentage points (while Query-only's rises by 0.75 percentage points).

Under matched node counts and full ablation on this development bank, the node groups identified in C-HyperNet consistently show stronger functional impact on both held-out target compositions than the matched Query-only groups—in both absolute and relative drops (64.7% vs. 41.5% on And-Or; 34.8% vs. 11.6% on Nested OR). Their selectivity differs by task: the And-Or group leaves atomic and seen-composition accuracy nearly unchanged (−0.10 and −0.20 percentage points), whereas the Nested OR group incurs a trade-off on atomic tasks (−1.55 percentage points). An important explanatory boundary remains: Sections 4.1 and 4.2 provide two separate pieces of evidence—representation probes show that atomic associations are decodable before compositional training, and node ablations show that specific HyperNet features functionally support OOD compositions afterward—but we have not yet directly traced the causal link between them, namely how signals from the circuit input drive the formation or activation of these compositional nodes and translate into the generated LoRA weights. Complete counts, relative changes, and ratios are available in the [node-intervention record](composition_nodes_en.md).
