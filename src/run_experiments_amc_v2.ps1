# run_experiments_amc_v2.ps1
# Adaptive Multi-Channel GFM v2 (AMC-v2) Batch Runner
# Fuses 5 subnets using separate GNNs with dynamic attention and adaptive source weighting.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")
$script = "run_MultiModalGNN_CrossAttention_CrossCountry_AMC_v2.py"
$outDir = "c:\Users\minelab\Desktop\projects\ssm"

foreach ($country in $countries) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  AMC-v2 GFM: Target = $country" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan

    $logFile = "$outDir\zero-shot_AMCv2_$country.txt"

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
        --loss_type bce `
        --coral_weight 50.0 > $logFile 2>&1

    Write-Host ""
    Write-Host "  Saved -> $logFile" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AMC-v2 GFM all countries complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
