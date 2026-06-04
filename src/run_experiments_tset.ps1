# run_experiments_tset.ps1
# TSET-GFM: Topology-Selective Ensemble Transfer GFM
# Module A: Subnet-Importance Weighting
# Module B: Focal Loss (gamma=2, alpha=0.75) - fixes Venezuela/Cuba class collapse
# Module C: Structure-Only Transfer Mode (SOTM) - fixes Iran/Venezuela linguistic isolation

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")
$script = "run_MultiModalGNN_CrossAttention_CrossCountry_TSET.py"
$outDir = "c:\Users\minelab\Desktop\projects\ssm"

foreach ($country in $countries) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  TSET-GFM: Target = $country" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan

    $logFile = "$outDir\zero-shot_TSET_$country.txt"

    python $script `
        --dataset $country `
        --device 0 `
        --epochs 1000 `
        --lr 1e-2 `
        --early 20 `
        --check 1 `
        --gnn sage `
        --embed_type positional_degree `
        --latent 128 `
        --splits 5 `
        --val_metric f1_macro `
        --focal_gamma 2.0 `
        --focal_alpha 0.75 `
        --coral_weight 500.0 `
        --sotm_threshold 0.3 | Tee-Object -FilePath $logFile

    Write-Host ""
    Write-Host "  Saved -> $logFile" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TSET-GFM all countries complete!" -ForegroundColor Green
Write-Host "  Run: python parse_results_tset.py to compare" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
