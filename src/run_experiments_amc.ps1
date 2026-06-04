# run_experiments_amc.ps1
# Adaptive Multi-Channel GFM (AMC-GFM) Batch Runner
# Fuses 5 subnets using separate GNNs with learnable Softmax fusion.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")
$script = "run_MultiModalGNN_CrossAttention_CrossCountry_AMC.py"
$outDir = "c:\Users\minelab\Desktop\projects\ssm"

foreach ($country in $countries) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  AMC-GFM: Target = $country" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan

    $logFile = "$outDir\zero-shot_AMC_$country.txt"

    python -u $script `
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
        --coral_weight 500.0 > $logFile 2>&1

    Write-Host ""
    Write-Host "  Saved -> $logFile" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AMC-GFM all countries complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
