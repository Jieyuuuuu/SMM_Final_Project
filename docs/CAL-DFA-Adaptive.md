# CAL-DFA-Adaptive: Per-Country Adaptive Decoupled Feature Alignment

Cross-country zero-shot information-operation (IO) account detection.

- Model / training script: [`src/run_MultiModalGNN_CrossAttention_CrossCountry_CALDFA.py`](../src/run_MultiModalGNN_CrossAttention_CrossCountry_CALDFA.py)
- Sweep runner: [`src/run_experiments_caldfa_sweep.ps1`](../src/run_experiments_caldfa_sweep.ps1) (sweeps `--coral_weight` ∈ {0, 0.1, 0.3})
- Authoritative results: [`results/summarize_final.py`](../results/summarize_final.py) → [`results/FINAL_RESULTS.md`](../results/FINAL_RESULTS.md)

> This is one of two research directions on the same task. The other,
> [`CS-DFA`](CS-DFA.md), changes the *model architecture* (splits the five behavioural
> sub-networks into per-channel GNNs). CAL-DFA-Adaptive takes the opposite, minimal-change
> stance: it keeps the DFA model **unchanged** and only adapts the alignment strength
> per target country.

---

## 1. Motivation

The strongest existing self-contained baseline, **DFA** (Decoupled Feature Alignment),
applies CORAL alignment between the source countries' text projections with a **single,
globally-fixed strength** `λ = 1.0` (`--coral_weight`). Our per-country diagnosis showed
that one global `λ` is the wrong choice, because alignment is **not uniformly helpful**:

- For five of the six countries, strong alignment helps — DFA (`λ=1.0`) is at or above
  the no-alignment reproduction on `china`, `iran`, `UAE`, `cuba`, `russia`.
- For **Venezuela**, the *same* CORAL alignment causes **negative transfer**: DFA drops
  to `0.768`, while simply turning alignment off (`λ=0`) recovers `0.813`.
- `cuba` is separately hard for a different reason — extreme class imbalance (IO ≈ 2.3%
  positives) — which makes any choice that depends on a few validation positives fragile.

A single fixed `λ` cannot serve both regimes at once: raise it and Venezuela suffers
negative transfer; lower it and the five alignment-friendly countries lose signal.

**Hypothesis.** If we let **each target country choose its own alignment strength** `λ`
on its own validation split, we recover the gains where alignment helps and disable it
where it hurts — without touching the model, and without using any test labels.

---

## 2. Method

### 2.1 Backbone (unchanged DFA)

CAL-DFA-Adaptive reuses the DFA multimodal cross-attention model verbatim. Unlike
[`CS-DFA`](CS-DFA.md), the five behavioural sub-networks are **merged into a single
graph** and one GNN runs over it.

```
text + struct features
        │
   struct_projector        text_projector
        │                       │
        ├──×── cross-attn(text) │           ← multiplicative cross-attention gating
        │      cross-attn(struct) ──×───────┤
        ▼                       ▼
   [struct_proj ‖ text_proj] → joint_projector
        │
        ▼
   single merged-graph GNN  (GraphSAGE / GCN)
        ▼
   classifier → IO / non-IO
```

### 2.2 The one change: per-country adaptive λ (core idea)

The only modification over DFA is **not** in the model — it is in how `λ` is selected:

> For each target country, run the model at alignment strengths
> **`λ` ∈ {0, 0.1, 0.3, 1.0}** (the `--coral_weight` flag) and keep the `λ` that
> maximises that country's **target-validation** `f1_macro`. Report the test
> `f1_macro` of the selected `λ`.

`λ = 1.0` is exactly the existing DFA run, so it is reused rather than recomputed; the
sweep only adds `λ ∈ {0, 0.1, 0.3}`. The per-country selection itself lives in
`adaptive_pick()` in [`results/summarize_final.py`](../results/summarize_final.py).

This uses the **same protocol** as the existing early-stopping in this codebase, which
already selects the checkpoint by target-validation `f1_macro`. **No test labels are
used** for the choice.

### 2.3 Loss and alignment

- **Loss: BCE task loss + `λ · CORAL`.** The CORAL term aligns the text projection
  pairwise across the source countries; `λ = --coral_weight` scales it. Focal Loss is
  **not** used (the earlier diagnosis showed it hurts extreme-imbalance countries such
  as Cuba).

### 2.4 Knobs that were tried and rejected (ablation, default off)

Two intuitively-appealing additions were implemented as flags but are **off by default**
because they measurably *hurt* (see §4.2 / §5):

- `--use_layernorm` (default `0`): LayerNorm on the modality projectors and the fused
  representation.
- `--calibrate` (default off): tune the decision threshold on the target-validation
  split instead of the fixed `0.5`.

### 2.5 Evaluation protocol

True **zero-shot cross-country**: the model is trained only on the five *source*
countries and evaluated on the held-out *target* country's test split. Five random
splits, `f1_macro` reported as mean ± std. Per-subnet test slices
(`TEST_coRT`, … `TEST_tweetSim`) are logged for diagnosis. Selection (both early-stopping
and the `λ` choice) is on the target-validation split — identical across all compared
methods.

---

## 3. How to run

Unlike [`CS-DFA`](CS-DFA.md) (which is wired into `run_experiments.py --method csdfa`),
**CAL-DFA-Adaptive is run via a standalone sweep** plus a summariser. From the project
root on Windows / PowerShell, with the dataset present at
`<project_root>/data/processed/<country>/`:

```powershell
# 1. Activate the environment (.venv has torch 2.6.0+cu124)
& .\.venv\Scripts\Activate.ps1

# 2. Run the λ sweep from src/  (sweeps λ ∈ {0, 0.1, 0.3}; λ=1.0 reuses existing DFA results)
Set-Location src
.\run_experiments_caldfa_sweep.ps1

# 3. Produce the authoritative table + figures from results/
Set-Location ..\results
python summarize_final.py      # → FINAL_RESULTS.md  (per-country λ selection)
python plot_final_results.py   # → final_results_avg.png / final_results_bar.png
```

The sweep writes raw per-split logs to `results/zero-shot_CW000_<country>.txt`,
`zero-shot_CW010_*.txt`, `zero-shot_CW030_*.txt` (one per `λ`), reusing the existing
`zero-shot_DFA_*.txt` for `λ=1.0`. (These `.txt` files are UTF-16 encoded.)

---

## 4. Results

### 4.1 Main result — full zero-shot `f1_macro`

Authoritative numbers are produced by
[`results/summarize_final.py`](../results/summarize_final.py) →
[`results/FINAL_RESULTS.md`](../results/FINAL_RESULTS.md).

| Method | china | iran | UAE | cuba | russia | venezuela | **avg** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Paper (external\*) | 0.581 | 0.728 | 0.839 | 0.899 | 0.798 | 0.910 | 0.7925 |
| no-DA repro (`λ=0`) ◆ | 0.581 | 0.702 | 0.875 | 0.781 | 0.825 | 0.813 | 0.7629 |
| DANN | 0.578 | 0.705 | 0.857 | 0.794 | 0.823 | 0.792 | 0.7583 |
| CORAL | 0.585 | 0.705 | 0.878 | 0.848 | 0.823 | 0.768 | 0.7678 |
| DFA-GFM (fixed `λ=1.0`) | 0.603 | 0.706 | 0.880 | 0.856 | 0.836 | 0.768 | 0.7747 |
| **CAL-DFA-Adaptive (ours)** | **0.603** | **0.706** | **0.880** | **0.856** | **0.836** | **0.813** | **0.7823** |
| selected `λ*` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **0.0** | — |

\* Paper = IOHunter paper, Table 3 (Only-PreTrain) — **external numbers, not
reproducible in this codebase** (see §6).
◆ no-DA repro = the identical architecture with alignment turned off (`--coral_weight 0`):
our own *reproducible* baseline.

CAL-DFA-Adaptive beats the reproducible no-DA baseline (`0.7629`, **+1.9pp**) and the
previous-best fixed-`λ` DFA (`0.7747`, **+0.8pp**). The whole gain comes from **Venezuela**
auto-selecting `λ=0` (`0.768 → 0.813`) to dodge negative transfer; the other five
countries select `λ=1.0` and match DFA exactly. A single fixed `λ` cannot serve both
regimes — **per-country adaptive `λ` is what closes the gap**.

### 4.2 Ablation — two negative results

Both variants intervene on top of DFA; both *hurt*.

| Variant | mechanism | avg `f1_macro` | verdict |
|---|---|:-:|:-:|
| DFA (baseline) | fixed `λ=1.0`, no LayerNorm, fixed-0.5 threshold | 0.7747 | — |
| DFA + **LayerNorm** | `--use_layernorm 1` | **0.7023** | ❌ harmful |
| DFA + LayerNorm + **threshold calib.** | `--use_layernorm 1 --calibrate` | **0.7323** | ❌ harmful |
| **CAL-DFA-Adaptive** | per-country `λ`, no LayerNorm, no calibration | **0.7823** | ✅ adopted |

Figures (under `results/`): `final_results_avg.png` (6-country average per method),
`final_results_bar.png` (per-country bars, Venezuela's `λ=0` recovery visible), and
`ablation_bar.png` (the two negative results, China/Cuba collapse under LayerNorm).

---

## 5. Analysis

**Per-country adaptive `λ` is the contribution; the two extra mechanisms are inert or harmful.**

- **Adaptive `λ` (the win).** Five countries keep `λ=1.0` (strong alignment helps);
  only Venezuela switches to `λ=0`, recovering `0.768 → 0.813`. This is the entire
  +1.9pp over the reproducible baseline and +0.8pp over fixed-`λ` DFA. The negative
  transfer is real, not a bookkeeping artefact: a rerun confirmed DFA Venezuela ≈ `0.768`,
  and disabling alignment lifts it back to `0.813`.

- **LayerNorm hurts (`csdfa`-style scale signal broken).** `--use_layernorm 1` drops the
  average to `0.7023`. LayerNorm destroys the scale signal the cross-attention gating
  relies on to separate the rare positive class, and the two most imbalanced countries
  collapse — **Cuba `0.507`, China `0.521`**.

- **Threshold calibration overfits.** Tuning the decision threshold on the target
  validation split (`--calibrate`) recovers only part of the LayerNorm damage (`0.7323`,
  still below DFA). Cuba has ~2.3% positives, so its validation set holds only a few
  dozen positives; sweeping a threshold there overfits validation noise (Cuba precision
  variance blows up to ±0.33) and lowers the test result.

These two "intuitive but empirically harmful" outcomes are themselves useful
**negative results** for the report.

**No extra test-label use.** The adaptive `λ` choice uses the same target-validation
selection the codebase already uses for early stopping — it does not touch test labels.

---

## 6. Caveats

1. **Reproduction gap — claim "beats the reproducible baseline", not "beats the paper".**
   The paper's `0.7925` is **not reproducible in this codebase**. Reproducing the paper's
   *own no-alignment method* faithfully (identical architecture, `--coral_weight 0`)
   yields only `0.7629`, ~3pp short, with the gap concentrated in two countries:

   | Country | Paper | Our faithful repro | Δ |
   |---|:-:|:-:|:-:|
   | Venezuela | 0.910 | 0.813 | −9.7pp |
   | Cuba | 0.899 | 0.781 | −11.8pp |

   This is a **data-preprocessing / split / hyperparameter** reproduction gap, not a
   method gap — even the paper's non-alignment method does not reach `0.910` here. Report
   claims should therefore be **relative to the reproducible baseline**.

2. **Selection on the target validation set.** Like the DFA/CS-DFA baselines, both
   early-stopping and the `λ` choice maximise the *target* validation `f1_macro`. This is
   shared by every compared method, so it does not affect the *relative* conclusion, but
   absolute numbers may be optimistic.

3. **Coarse `λ` grid.** Only four values are swept (`λ ∈ {0, 0.1, 0.3, 1.0}`); a finer
   grid or continuous tuning is possible future work.

---

## 7. Takeaway

For cross-country zero-shot IO detection, a **single global alignment strength is the
wrong assumption** — CORAL alignment helps most countries but causes negative transfer on
Venezuela. Letting **each target country pick its own `λ`** on validation recovers the
gains where alignment helps and disables it where it hurts, beating the reproducible
no-DA baseline by **+1.9pp** (and fixed-`λ` DFA by +0.8pp) with **zero model changes**.
The two architectural add-ons we tried (LayerNorm, threshold calibration) are negative
results, and the headline contribution is purely the per-country adaptive `λ`.
