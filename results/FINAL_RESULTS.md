# IOHunter+ Final Zero-shot Results (Macro-F1)

Proposed method: **CAL-DFA-Adaptive** = DFA architecture + per-country validation-selected CORAL weight lambda in {0.0, 0.1, 0.3, 1.0}.

| Country | Paper* | no-DA repro (lam=0) | DANN | CORAL | DFA (lam=1.0) | **CAL-DFA-Adaptive** | sel. lam |
|---|---|---|---|---|---|---|---|
| china | 0.5814 | 0.5805 | 0.5779 | 0.5853 | 0.6026 | **0.6026** | 1.0 |
| iran | 0.7278 | 0.7019 | 0.7048 | 0.7048 | 0.7063 | **0.7063** | 1.0 |
| UAE | 0.8393 | 0.8750 | 0.8572 | 0.8784 | 0.8797 | **0.8797** | 1.0 |
| cuba | 0.8991 | 0.7813 | 0.7942 | 0.8479 | 0.8559 | **0.8559** | 1.0 |
| russia | 0.7977 | 0.8254 | 0.8233 | 0.8227 | 0.8362 | **0.8362** | 1.0 |
| venezuela | 0.9099 | 0.8133 | 0.7924 | 0.7676 | 0.7676 | **0.8133** | 0.0 |
| **AVERAGE** | 0.7925 | 0.7629 | 0.7583 | 0.7678 | 0.7747 | **0.7823** | - |

\* Paper numbers are from IOHunter Table 3 (Only-PreTrain) and are **NOT reproducible in this codebase**: our faithful no-DA reproduction (lam=0) averages 0.7629, ~3.0pp below the paper, concentrated in Venezuela (paper 0.910 vs ours 0.813) and Cuba (paper 0.899 vs ours 0.781).

## Ablations (negative results)

- **LayerNorm feature normalisation**: harmful. DFA+LN avg = 0.7023 (Cuba collapses to 0.507, China 0.521); LayerNorm destroys the scale signal the cross-attention gating relies on for the rare positive class.
- **Target-val threshold calibration**: harmful for rare-positive targets. DFA+LN+calibration avg = 0.7323; the validation positive count is too small (Cuba 2.3% IO) so the swept threshold overfits val noise.
- Both were therefore removed from the final method.

## Key finding

Domain-adaptation alignment helps China/Iran/UAE/Cuba/Russia (lam=1.0 selected) but causes **negative transfer on Venezuela** (lam=0 selected). A single fixed lambda cannot serve both; selecting lambda per target on the validation split yields **0.7823**, beating the reproducible no-DA baseline (0.7629) and fixed-lambda DFA (0.7747).
