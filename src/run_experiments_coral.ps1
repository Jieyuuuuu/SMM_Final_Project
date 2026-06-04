# run_experiments_coral.ps1
# 專門用來自動跑完 6 組 CORAL 優化實驗，並將輸出結果存入對應的 .txt 檔案中。

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== 開始執行 6 組 Deep CORAL 優化實驗 ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "正在執行 Deep CORAL - 測試目標國家: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    # 執行 CORAL 指令，並將輸出同時顯示在螢幕且寫入專案目錄下的 txt 檔案
    python run_MultiModalGNN_CrossAttention_CrossCountry_CORAL.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_CORAL_$country.txt"
}

Write-Host "====== 6 組 Deep CORAL 實驗已全數執行完畢！ ======" -ForegroundColor Yellow
