# SMM Final Project — InfoOps Detection via GFM

This repository contains source code and experiment logs for our Social Media Mining final project,
which extends the AAAI 2025 paper **"InfoOpsGFM"** with several zero-shot cross-country adaptation improvements.

## Project Structure

```
src/            ← All Python training scripts & PowerShell batch runners
  run_MultiModalGNN_CrossAttention_CrossCountry_DFA.py  ← DFA-GFM (our primary model)
  run_MultiModalGNN_CrossAttention_CrossCountry_AMC.py  ← AMC-GFM v1 (Node-Level Attention Fusion)
  run_MultiModalGNN_CrossAttention_CrossCountry_AMC_v2.py  ← AMC-GFM v2 (Adaptive Source Weighting)
  run_MultiModalGNN_CrossAttention_CrossCountry_CORAL.py   ← CORAL baseline
  run_MultiModalGNN_CrossAttention_CrossCountry_DANN.py    ← DANN baseline
  run_experiments_dfa.ps1    ← run DFA-GFM for all 6 countries
  run_experiments_amc.ps1    ← run AMC-GFM v1
  run_experiments_amc_v2.ps1 ← run AMC-GFM v2
  data_loader.py             ← Dataset loading utilities
  model_eval.py              ← Evaluation metrics helpers
  my_utils.py                ← Shared utilities (edge index, embeddings, etc.)
  models.py                  ← Base GNN model

results/        ← Experiment output logs & visualizations
  zero-shot_baseline_*.txt   ← Baseline (GFM pre-trained, no adaptation)
  zero-shot_DFA_*.txt        ← DFA-GFM results (domain feature alignment)
  zero-shot_DANN_*.txt       ← DANN results
  zero-shot_CORAL_*.txt      ← CORAL results
  zero-shot_AMC_*.txt        ← AMC-GFM v1 results
  zero-shot_multilevel_*.txt ← Multi-level optimization results
  parse_results.py           ← Parse & compare experiment results vs baseline
  parse_amc.py               ← Parse AMC-GFM specific results
```

## Original Paper

Base code from: https://github.com/mminici/InfoOpsGFM  
Paper: "Zero-Shot Influence Operation Detection via Graph Feature Mining" (AAAI 2025)

## Data

The dataset is available at the Zenodo link specified in the original repo.
Download and place under `InfoOpsGFM/data/processed/{country}/`.

## How to Run

```powershell
cd InfoOpsGFM/src

# Run DFA-GFM (our best model) for all 6 countries:
powershell -ExecutionPolicy Bypass -File .\run_experiments_dfa.ps1

# Run AMC-GFM v2 (adaptive source domain selection):
powershell -ExecutionPolicy Bypass -File .\run_experiments_amc_v2.ps1

# Parse results and compare with baseline:
python parse_results.py
python parse_amc.py
```
