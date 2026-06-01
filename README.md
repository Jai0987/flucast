# FluCast

**Disease outbreak detection from localized Reddit activity.**

FluCast detects early flu signals by applying a fine-tuned RoBERTa classifier to Reddit posts, aggregating the model's confidence scores geographically and temporally into a weighted outbreak signal, and validating against CDC ILINet ground truth. This directly addresses the core failure of Google Flu Trends (raw volume = noisy signal) by filtering for genuine first-person flu reports.

**Course:** CS610 — Advanced Artificial Intelligence, Drexel University  
**Team:** Jai Kashyap, Raymond Nguyen, Chris Karmilowicz

---

## How It Works

1. Reddit posts from city-specific subreddits are filtered by flu-related keywords.
2. A fine-tuned `roberta-base` classifier assigns each post a flu-relevance probability.
3. Those probabilities are summed per (subreddit, week) and normalized by total post volume → **weighted flu signal**.
4. The signal is compared against CDC ILINet weekly ILI% data for HHS Region 3 (mid-Atlantic).

---

## Repository Structure

```
flucast/
├── data/
│   └── filtered/               # Keyword-filtered Reddit 2018 data per metro
│       ├── NYC/
│       ├── Philadelphia/
│       └── Houston/
├── results/                    # Generated outputs (plots, CSVs)
├── weighted_training.py        # RoBERTa fine-tuning script (run in Colab)
├── aggregate_signal.py         # Weekly flu signal aggregation
├── validate_cdc.py             # CDC ILINet comparison and correlation
├── run_inference_multi_metro.py # Apply model to new metro data
├── training_combined_roberta_dataset.csv  # ~2,340 hand-labeled training posts
├── region3_2018_keyword_matches_all_roberta_labeled.csv  # Region 3 inference output
└── requirements.txt
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

> `torch` and `transformers` are only needed for `run_inference_multi_metro.py`.
> The analysis scripts (`aggregate_signal.py`, `validate_cdc.py`) only require
> `pandas`, `matplotlib`, `seaborn`, and `scipy`.

---

## Running the Pipeline

### Step 1 — Train the Classifier (already done)

The model was trained in Google Colab using `weighted_training.py`. Training data is in `training_combined_roberta_dataset.csv`. The saved model lives locally (too large for git).

To retrain:
```bash
# Upload weighted_training.py and training_combined_roberta_dataset.csv to Colab
# Run weighted_training.py — it saves the model to /content/final_roberta_flu_model/
# Download and unzip final_roberta_flu_model.zip into this directory
```

### Step 2 — Aggregate the Region 3 Signal

```bash
python aggregate_signal.py
# Output: results/region3_weekly_signal.csv
#         results/region3_weekly_signal.png
```

### Step 3 — Validate Against CDC ILINet

```bash
python validate_cdc.py
# Fetches HHS Region 3 2018 ILI data from CMU DELPHI Epidata API (internet required)
# Output: results/region3_cdc_comparison.png
#         results/region3_correlation.csv
```

### Step 4 — Run Inference on Additional Metros

```bash
# 1. Set MODEL_DIR at the top of the script to your local model path
# 2. Run:
python run_inference_multi_metro.py
# Output: results/nyc_2018_roberta_labeled.csv
#         results/philadelphia_2018_roberta_labeled.csv
#         results/houston_2018_roberta_labeled.csv

# 3. Aggregate each metro's signal:
python aggregate_signal.py results/nyc_2018_roberta_labeled.csv
python aggregate_signal.py results/philadelphia_2018_roberta_labeled.csv
python aggregate_signal.py results/houston_2018_roberta_labeled.csv
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| [Pushshift Reddit Archive](https://files.pushshift.io/reddit/) | 2018 Reddit comments and submissions |
| [CDC ILINet via CMU DELPHI Epidata](https://api.delphi.cmu.edu/epidata/) | Weekly ILI% by HHS region |

---

## Key Results

- **Classifier performance:** Fine-tuned `roberta-base` on 2,340 labeled posts with class-weighted loss to handle imbalance.
- **Region 3 correlation:** FluCast weighted signal vs. CDC wILI% — Pearson r ≈ 0.47 (p < 0.001), Spearman r ≈ 0.46 (p < 0.001) over 52 aligned weeks.
- **Geographic generalization:** Model applied without retraining to r/nyc, r/philadelphia, r/houston.
