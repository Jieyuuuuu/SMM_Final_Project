# run_experiments_dfa.ps1
# 專門用來自動跑完 6 組 DFA-GFM (解耦特徵對齊) 實驗，並將輸出結果存入對應的 .txt 檔案中。

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== 開始執行 6 組 DFA-GFM (解耦對齊) 優化實驗 ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "正在執行 DFA-GFM - 測試目標國家: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    # 執行 DFA 指令，並將輸出同時顯示在螢幕且寫入專案目錄下的 txt 檔案
    python run_MultiModalGNN_CrossAttention_CrossCountry_DFA.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_DFA_$country.txt"
}

Write-Host "====== 6 組 DFA-GFM 實驗已全數執行完畢！ ======" -ForegroundColor Yellow
