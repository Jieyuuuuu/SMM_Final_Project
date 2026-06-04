# run_experiments_multilevel.ps1
# Multi-Level Optimized Zero-Shot GFM (MSCA + GRPS + TT-GCA)
# Level 1: Multilingual Semantic Centroid Alignment (MSCA)
# Level 2: Graph-Regularized Posterior Smoothing (GRPS) via Label Propagation
# Level 3: Test-Time Graph Contrastive Adaptation (TT-GCA) via Link Prediction

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")
$script = "run_MultiModalGNN_MultiLevelOpt.py"
$outDir = "c:\Users\minelab\Desktop\projects\ssm"

foreach ($country in $countries) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Multi-Level GFM: Target = $country" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan

    $logFile = "$outDir\zero-shot_multilevel_$country.txt"

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
        --msca `
        --grps `
        --grps_alpha 0.6 `
        --grps_max_iter 30 `
        --tt_gca `
        --tt_gca_epochs 10 `
        --tt_gca_lr 1e-3 | Tee-Object -FilePath $logFile

    Write-Host ""
    Write-Host "  Saved -> $logFile" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Multi-Level Optimized all countries complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
