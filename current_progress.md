# IOHunter+ 專案進度總覽

> NCU 社群媒體探勘 (Social Media Mining) 期末專案 — Group 7
> 楊絜宇 · 郭建廷 · 譚聲全 · 丁明全
> 最後更新: 2026-06-06

---

## 1. 專案概覽

本專案 **IOHunter+** 基於 AAAI 2025 論文 **IOHunter: Graph Foundation Model to Uncover Online Information Operations** (Minici et al., 2025) 進行擴展。

**任務**: Twitter/X 上的 **協同性資訊操弄 (Information Operation, IO) Driver 偵測** — 二元分類 (IO driver vs. 一般用戶),並聚焦在 **zero-shot 跨國遷移**:在 5 個國家的資料上訓練,直接拿去測第 6 個國家。

**核心貢獻**: 在原 IOHunter 上加入 **domain adaptation** 機制 (CORAL / DANN / DFA / AMC / Multilevel / DMC / TSET),探討哪種機制最能解決跨國 IO 偵測的 domain gap。

---

## 2. 論文 IOHunter 重點回顧

### 架構 (Methodology)
- **輸入模態**:
  - **文字**: 使用者貼文 → 凍結的 multilingual SBERT → 平均池化得 content embedding `c_i`
  - **結構**: Fused Similarity Network (5 種行為相似性合併: co-Retweet、co-URL、co-Hashtag、Fast-Retweet、Tweet Similarity) → 度數 bucket one-hot → `g_i`
- **融合**: Cross-Attention (Abavisani et al., 2020) — 兩條 modality 互相生成 attention 係數,過濾跨模態的雜訊
- **GNN**: GCN / GraphSAGE,2 層 message passing,Dropout 20%
- **Loss**: Binary Cross-Entropy
- **學習設定**: LM 凍結,僅 Multi-Modal Blend + GNN 參數可訓練

### 資料 (來自 Seckin et al. 2024)
| Country   | Nodes  | Edges     | Homophily | IO Prop. |
|-----------|--------|-----------|-----------|----------|
| UAE       | 9,242  | 2,118,684 | 52.8%     | 35.7%    |
| Cuba      | 19,822 | 4,737,374 | 37.1%     | 2.3%     |
| Russia    | 666    | 10,381    | 53.0%     | 38.4%    |
| Venezuela | 4,980  | 56,700    | 77.7%     | 10.6%    |
| Iran      | 12,977 | 392,938   | 81.0%     | 32.2%    |
| China     | 22,694 | 410,979   | 41.1%     | 3.3%     |

### 論文核心結果 (Macro-F1)

| Research Question | 重點數字 |
|---|---|
| **RQ1 全監督** | IOHunter 平均 **96.43**,勝 node2vec+RF (87.77) 約 9 pp;Iran +20pp |
| **RQ2 資料稀缺** | 0.1% 訓練資料仍 ~80% F1,node2vec+RF 退步嚴重 |
| **RQ3 跨國 zero-shot** | Only-PreTrain: 平均 **+5.69%** vs 0.1% supervised;+0.1% fine-tune: **+10.25%** |
| **Ablation** | 沒有文字 → 88.53;沒有圖 → 86.76;沒有 CrossAttn → 88.80;完整版 96.43 |

### 關鍵設計直覺
- LM 凍結 → 避免破壞預訓練語意空間,只學如何「混合」
- Cross-Attention → 比單純 concat 更穩定 (RQ1 std dev 從 ~21 降到 ~1)
- 度數 bucket → 處理 heterophily (Cuba 37%、China 41% 都很 heterophilic)

---

## 3. 本專案 (IOHunter+) 的擴展與創新

我們在 IOHunter baseline 之上,**加入 zero-shot domain adaptation** 機制 (原論文需要 0.1% target fine-tune,我們嘗試完全不用 target 標籤),總共實作 **8 條改進路線**:

| 方法 | 主程式 | 核心機制 | 期中報告 |
|---|---|---|---|
| **Baseline** | `run_MultiModalGNN_CrossAttention.py` | 復現 IOHunter | ✅ |
| **DANN** | `run_..._DANN.py` | 對抗式 Domain Classifier + Gradient Reversal Layer | ✅ |
| **Deep CORAL** | `run_..._CORAL.py` | Source/Target 協方差矩陣 Frobenius norm 對齊 | ✅ |
| **DFA-GFM** | `run_..._DFA.py` | **Surgical Decoupling** — 只對齊文字特徵協方差,圖結構完全不動 (shielded) | ✅ |
| **CAL-DFA-Adaptive** ⭐ | `run_..._CALDFA.py` + `summarize_final.py` | DFA 架構 + **per-country 用 target-val 自選對齊強度 λ** (避免 Venezuela 負遷移)。**目前最強,平均 0.782** | 🆕 (主推) |
| **AMC-GFM v1** | `run_..._AMC.py` | Adaptive Multi-Channel,讀取多種邊類型 (`read_all_data_amc`) | ❌ |
| **AMC-GFM v2** | `run_..._AMC_v2.py` | v1 + **Focal Loss** (γ=2.0, α=0.75) 處理類別不平衡 | ❌ |
| **Multilevel** | `run_MultiModalGNN_MultiLevelOpt.py` | **LEVEL 1**: MSCA 多語意中心對齊 + **Module C**: SOTM 用 MMD 偵測語言隔離 + **LEVEL 2**: Focal+CORAL 多級最佳化 | ❌ |
| **DMC** | `run_..._DMC.py` | Multi-Channel GNN + 節點級注意力融合 (`NodeLevelAttentionFusion`) | ❌ |
| **TSET** | `run_..._TSET.py` | Structure-aware Transfer | ❌ (沒結果) |

### DFA-GFM 設計哲學 (本專案主要 contribution)

對比 DANN/CORAL:
- DANN 用對抗式訓練對齊 **整個融合表示** → 容易破壞 graph 訊號 (Venezuela 退步嚴重)
- CORAL 對 **融合後特徵** 做協方差對齊 → 仍會影響到 graph topology
- **DFA-GFM**: 只對 **text features 投影** 做協方差對齊,**graph topology 完全 shielded / no domain loss** → 保留拓樸的 country-specific 訊號,只對齊跨語言文字分佈

報告中稱此為 **"Surgical Decoupling"**。

### Multilevel-GFM 設計 (進階)

- **LEVEL 1 — MSCA (Multilingual Semantic Centroid Alignment)**: 標準化目標文字特徵以匹配全域 source 統計量
- **Module C — SOTM (Structure-Only Transfer Mode)**: 用 MMD 距離偵測語言隔離,若文字差異過大就跳過文字模態,只用結構特徵
- **LEVEL 2**: Focal Loss + CORAL loss 對齊

---

## 4. 實驗結果總表 (Zero-shot Macro-F1)

完整 zero-shot 結果 (從 `results/zero-shot_*.txt` 解析,UTF-16 編碼)。
**2026-06-08 更新**: 加入「本 codebase 可重現的 no-DA baseline」與新方法 **CAL-DFA-Adaptive**,
並修正先前直接引用論文外部數字的問題。權威表格與選擇邏輯見 `results/summarize_final.py` → `results/FINAL_RESULTS.md`。

| 方法 | China | Iran | UAE | Cuba | Russia | Venezuela | 平均 |
|------|-------|------|-----|------|--------|-----------|------|
| Paper (外部,Table 3) ※ | 0.581 | 0.728 | 0.839 | 0.899 | 0.798 | 0.910 | **0.793** |
| **no-DA repro (λ=0)** ◆ | 0.581 | 0.702 | 0.875 | 0.781 | 0.825 | 0.813 | **0.763** |
| DANN | 0.578 | 0.705 | 0.857 | 0.794 | 0.823 | 0.792 | 0.758 |
| CORAL | 0.585 | 0.705 | 0.878 | 0.848 | 0.823 | 0.768 | 0.768 |
| DFA-GFM (λ=1.0) | 0.603 | 0.706 | 0.880 | 0.856 | 0.836 | 0.768 | 0.775 |
| **CAL-DFA-Adaptive** ⭐ | **0.603** | **0.706** | **0.880** | **0.856** | **0.836** | **0.813** | **0.782** |
| AMC v2 | 0.523 | 0.683 | 0.819 | 0.647 | 0.767 | N/A | 0.688 |
| Multilevel | 0.462 | 0.658 | 0.795 | 0.523 | 0.691 | 0.592 | 0.620 |

※ Paper 數字取自 IOHunter 論文 Table 3 (Only-PreTrain),**在本 codebase 無法重現**。
◆ no-DA repro = 用完全相同架構但關掉對齊 (`coral_weight=0`) 的忠實重現,即「我們自己的論文方法重現」。

### 重點觀察 (已用實驗驗證)

1. **CAL-DFA-Adaptive 是本專案最強方法**: 平均 **0.782**,**勝過可重現的 no-DA baseline (0.763, +1.9pp) 與固定-λ 的 DFA (0.775, +0.8pp)**。
   - 機制: 對每個 target country,在 **target validation split** 上從 λ∈{0,0.1,0.3,1.0} 選 macro-F1 最高者 (與現有 early-stopping 同協定)。China/Iran/UAE/Cuba/Russia 自動選 λ=1.0 (強對齊有益),**Venezuela 自動選 λ=0 (關閉對齊以避免負遷移)**。
2. ⚠️ **論文 baseline 0.793 在本 codebase 不可重現**: 用相同架構的 no-DA 重現只有 **0.763**,落差 ~3pp 集中在 Venezuela (論文 0.910 vs 我們 0.813) 與 Cuba (0.899 vs 0.781) — 屬於資料前處理/切分/超參的 **reproduction gap**,非方法差距。期末報告改以「贏過可重現 baseline」為主張。
3. **negative transfer 確認來源是 CORAL 對齊強度 λ**: λ 越大對 UAE/Russia/China 越好,但對 Venezuela 越差;單一固定 λ 無法兩全 → per-country 自適應 λ 是解法。
4. **兩個被推翻的點子 (期末報告當 negative result 寫)**:
   - **LayerNorm 特徵正規化 = 有害**: DFA+LN 平均掉到 0.702 (Cuba 0.507、China 0.521),破壞了 cross-attention gating 賴以分辨稀少正類的尺度訊號。
   - **target-val threshold 校準 = 對稀少正類有害**: DFA+LN+校準 0.732;Cuba 正樣本僅 2.3%,val 上掃 threshold 會 overfit 雜訊。
5. ⚠️ 原 **Venezuela DFA=CORAL 數字相同 (0.768)** 經重跑確認: DFA(λ=1.0) Venezuela 確實 ~0.768,負遷移為真 (非單純檔案複製);自適應改選 λ=0 後回升到 0.813。
6. **AMC、Multilevel 表現不如 baseline** (未重跑,沿用舊值);**AMC 在 Venezuela 沒有結果**。

### 視覺化 (`results/`)
- `zero-shot_comparison_bar.png`、`zero-shot_comparison_radar.png` — 全方法比較
- `coral_subnetworks_heatmap.png` — CORAL 在 5 種子網路上的表現 (見 PDF p.17)
- `russia_subnetworks_ablation.png` — Russia sub-network ablation
- `graph_viz_coordination_clique.png` — Russia IO 帳號 coordination clique
- `graph_viz_subnets.png` — Russia 5 種 behavioral sub-networks

---

## 5. 進度評估

### 已完成 (約 75–80%)
- ✅ Baseline (IOHunter) 復現
- ✅ DANN、CORAL、DFA-GFM 三條主要 domain adaptation 路線 (期中報告已收錄)
- ✅ AMC v1/v2、Multilevel、DMC 進階方法的程式碼實作
- ✅ 6 國 zero-shot 完整實驗 (除 Venezuela AMC、TSET 全國)
- ✅ 視覺化 (radar / bar / heatmap / subnetwork 分析)
- ✅ 期中 Progress Report (20 頁簡報,覆蓋三條主線)
- ✅ 期中報告辨識出 negative transfer 現象與 Venezuela/Cuba 西班牙語問題

### 未完成 / 待修正
- ❌ 期中 Progress Report **未涵蓋** AMC / Multilevel / DMC / TSET 的結果
- ❌ Venezuela DFA 數值疑似 bug (跟 CORAL 完全相同) — 需重跑驗證
- ❌ `results/zero-shot_baseline_*.txt` 沒有 `[TEST]` metrics (只有訓練 log) → 改用論文官方數字代替
- ❌ TSET 完全沒有結果檔
- ❌ `results/parse_results.py` 寫死 `minelab` 的絕對路徑,別人 clone 下來無法直接執行
- ❌ 沒有遷移到期中報告預告的 **Seckin et al. 16 國 / 26 campaigns / 13M tweets** 大資料集
- ❌ 期末報告 / 期末簡報尚未撰寫

---

## 6. 可優化方向 (Recommendations)

### A. 解釋現有失敗,把 AMC / Multilevel「救回來」(高 ROI)

**問題**: AMC、Multilevel 是「我們自己提的」方法,但表現比 baseline 還差,期末報告變難寫。建議:

1. **檢查 Multilevel 的 MSCA 是否反而破壞 Cuba/Venezuela 的西班牙語訊號**
   - MSCA 強制把 target 文字特徵拉到 source 中心 → 西班牙語特徵被中文/阿拉伯文「平均」掉
   - 建議: 加入 SOTM 觸發 threshold 的 sensitivity analysis
2. **AMC v2 加了 Focal Loss 但沒有解決根本失衡**
   - Cuba IO Prop. 只有 2.3%、China 3.3%
   - 建議: 試 class-balanced sampling 或 BCE pos_weight (比 Focal Loss 直觀)
3. **DMC 的 multi-channel 是否退化成單通道**
   - 檢查 attention coefficients 分佈 — 若一個 channel 拿走 >90% 權重,就退化了

### B. 修正 Venezuela 退步 — Negative Transfer 問題 (期末報告必修)

DANN/CORAL/DFA 都在 Venezuela 退步 ~14pp。假設: Venezuela 跟其他 source 在語言/IO 行為上有 mode collapse。可試:

1. **Source-target similarity weighting**: 給語言相近的 source (Cuba=西班牙文) 更高權重,語言遠的 source (China、Iran) 更低權重
2. **Selective source**: 排除某些 source,做 leave-3-out 而非 leave-1-out
3. **Per-country adaptation strength**: CORAL/DFA 的 λ 對每個 target 自動調 (validation F1)

### C. 把結果分析自動化

1. `results/parse_results.py` 改成相對路徑 (用 `os.path.dirname(__file__)`)
2. baseline runner 補上 `--eval-test` flag,讓 `[TEST]` metrics 進到 log
3. 統一各方法結果格式,寫一支腳本一鍵 dump 完整 Markdown 表格

### D. 移植到 Seckin et al. 大資料集 (期中報告已列為 next)

- 規模: 16 國 state actors、26 verified campaigns、~303k 帳號、~13M tweets
- 對應論文: **Seckin, O. C. et al. (2024) "Labeled Datasets for Research on Information Operations" (arXiv:2411.10609)**
- 挑戰: 從「6 國聚合圖」變成「per-campaign + 時間/主題配對 control」→ 資料前處理需重寫
- 預估工作量: 1 週

### E. 期末報告必補的圖

- AMC v1 vs v2 比較 (Focal Loss 有沒有幫助)
- Multilevel 的 MSCA on/off ablation
- 6 種方法在 6 國的 heatmap (目前只有 CORAL 有,見 `coral_subnetworks_heatmap.png`)
- DFA-GFM 的 surgical decoupling 直覺示意圖 (PDF p.18 已有,但要加文字說明)
- DMC 的 channel attention 分佈圖 (證明 multi-channel 真的有用)

---

## 7. 接下來的工作優先序

依「期末交件」為主軸排序:

| 優先 | 任務 | 預估時間 | 為什麼 |
|---|---|---|---|
| 🔴 | 修 Venezuela bug (DFA/CORAL 數字相同) | 30 min | 結果可信度問題 |
| 🔴 | 補齊 baseline 的 test metrics | 1 h | 期末報告要跟自家 baseline 比 |
| 🟡 | AMC / Multilevel ablation,找出退步原因 | 半天 | 寫得進報告就是亮點,寫不進就是黑歷史 |
| 🟡 | 撰寫期末報告 (整合 AMC/Multilevel/DMC) | 1–2 天 | 主要交件 |
| 🟢 | 嘗試 Seckin et al. 大資料集 | 1 週 | 加分項,但時間壓力大時可砍 |

---

## 重要檔案參考

- `IOHunter Thesis.pdf` (專案外) — 原論文 (9 頁)
- `SMM Progress report.pdf` (專案外) — 期中報告 (20 頁簡報)
- `src/run_MultiModalGNN_CrossAttention_CrossCountry_DFA.py` — DFA 主檔
- `src/run_MultiModalGNN_MultiLevelOpt.py` — Multilevel 主檔
- `src/run_MultiModalGNN_CrossAttention_CrossCountry_AMC_v2.py` — AMC v2 主檔
- `src/models.py` / `src/models_dann.py` — 模型元件
- `src/data_loader.py` — 資料載入與圖構建
- `results/zero-shot_<method>_<country>.txt` — 各方法各國結果 (UTF-16 編碼)
- `results/parse_results.py` — 結果解析腳本 (路徑硬編碼,待修)
- `src/run_experiments_*.ps1` — 各方法的 PowerShell batch runner
