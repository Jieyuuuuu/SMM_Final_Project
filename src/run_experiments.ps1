# run_experiments.ps1
# 專門用來自動跑完 6 組 Baseline 與 6 組 DANN 的實驗，並將輸出結果存入對應的 .txt 檔案中。

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

# 1. 跑 6 組 Baseline 實驗
Write-Host "====== 開始執行 6 組 Baseline 實驗 ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "正在執行 Baseline - 測試目標國家: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    # 執行 Baseline 指令，並將輸出同時顯示在螢幕且寫入專案目錄下的 txt 檔案
    python run_MultiModalGNN_CrossAttention_CrossCountry.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_baseline_$country.txt"
}

# 2. 跑 6 組 DANN 實驗
Write-Host "====== 開始執行 6 組 DANN 實驗 ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "正在執行 DANN - 測試目標國家: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    # 執行 DANN 指令，並將輸出同時顯示在螢幕且寫入專案目錄下的 txt 檔案
    python run_MultiModalGNN_CrossAttention_CrossCountry_DANN.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_DANN_$country.txt"
}

Write-Host "====== 所有 12 組實驗已全數執行完畢！ ======" -ForegroundColor Yellow
