# run_experiments_caldfa_sweep.ps1
# Reproduces the proposed CAL-DFA-Adaptive method end-to-end.
#
# CAL-DFA-Adaptive = DFA architecture (use_layernorm 0, calibrate off -- both
# alternatives were ablated and found harmful) with the text-feature CORAL
# alignment weight lambda selected PER TARGET COUNTRY on the target validation
# split. This script runs the lambda sweep; selection is then done by
# results/summarize_final.py (picks, per country, the lambda with best [VAL] F1).
#
# lambda = 1.0 is equivalent to the original DFA, whose outputs already exist as
# results/zero-shot_DFA_<country>.txt (summarize_final.py reads those for the
# lambda=1.0 candidate), so the sweep below only needs lambda in {0.0,0.1,0.3}.
# Outputs go to results/.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")
# label -> lambda (coral_weight)
$weights = [ordered]@{ "CW000" = 0.0; "CW010" = 0.1; "CW030" = 0.3 }

Write-Host "====== CAL-DFA-Adaptive: CORAL-weight (lambda) sweep ======" -ForegroundColor Green
foreach ($label in $weights.Keys) {
    $w = $weights[$label]
    foreach ($country in $countries) {
        Write-Host "---- $label (lambda=$w) target=$country ----" -ForegroundColor Cyan
        python run_MultiModalGNN_CrossAttention_CrossCountry_CALDFA.py `
            --dataset $country --device 0 --epochs 1000 --splits 5 `
            --use_layernorm 0 --coral_weight $w |
            Tee-Object -FilePath "../results/zero-shot_${label}_$country.txt"
    }
}
Write-Host "====== Sweep done. Now run:  python ../results/summarize_final.py ======" -ForegroundColor Yellow
