# run_experiments_caldfa.ps1
# 自動跑完 6 組 CAL-DFA (Calibrated Decoupled Feature Alignment) 實驗，
# 在 DFA 之上加入 LayerNorm 特徵正規化與 target-val threshold 校準，
# 並將輸出結果存入 results/ 目錄對應的 .txt 檔案中。

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== 開始執行 6 組 CAL-DFA (校準解耦對齊) 實驗 ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "正在執行 CAL-DFA - 測試目標國家: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # 執行 CAL-DFA，輸出同時顯示在螢幕並寫入 results/zero-shot_CALDFA_<country>.txt
    python run_MultiModalGNN_CrossAttention_CrossCountry_CALDFA.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../results/zero-shot_CALDFA_$country.txt"
}

Write-Host "====== 6 組 CAL-DFA 實驗已全數執行完畢！ ======" -ForegroundColor Yellow
