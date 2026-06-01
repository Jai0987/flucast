# aggregate_signal.py
# Groups RoBERTa-labeled posts by (subreddit, week) and computes a normalized
# flu signal: sum(prob_flu) / total_posts per bucket.
#
# Usage:
#   python aggregate_signal.py                                    (Region 3 default)
#   python aggregate_signal.py results/nyc_2018_roberta_labeled.csv

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_INPUT = "region3_2018_keyword_matches_all_roberta_labeled.csv"

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT

_raw_stem = os.path.splitext(os.path.basename(INPUT_CSV))[0]
stem = _raw_stem.replace("_2018_keyword_matches_all_roberta_labeled", "").replace("_2018_roberta_labeled", "") or _raw_stem
OUTPUT_CSV = f"results/{stem}_weekly_signal.csv"
OUTPUT_PLOT = f"results/{stem}_weekly_signal.png"

os.makedirs("results", exist_ok=True)

df = pd.read_csv(INPUT_CSV)

df["date"] = pd.to_datetime(df["date"])
df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
df["year"] = df["date"].dt.isocalendar().year.astype(int)
df["week_start"] = df["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

agg = (
    df.groupby(["subreddit", "week_start"])
    .agg(
        raw_post_count=("roberta_prob_flu", "count"),
        relevant_post_count=("roberta_flu_label", "sum"),
        prob_flu_sum=("roberta_prob_flu", "sum"),
    )
    .reset_index()
)

agg["weighted_signal"] = agg["prob_flu_sum"] / agg["raw_post_count"]
agg = agg.drop(columns=["prob_flu_sum"])
agg = agg.sort_values(["subreddit", "week_start"]).reset_index(drop=True)

agg.to_csv(OUTPUT_CSV, index=False)
print(f"Saved signal CSV -> {OUTPUT_CSV}")
print(agg.head(10).to_string(index=False))

subreddits = sorted(agg["subreddit"].unique())
n = len(subreddits)
cols = 2
rows = (n + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5), sharex=False)
axes = axes.flatten()

for i, sub in enumerate(subreddits):
    ax = axes[i]
    sub_df = agg[agg["subreddit"] == sub].copy()
    ax.plot(sub_df["week_start"], sub_df["weighted_signal"], marker="o", linewidth=1.5, markersize=3, color="#e05c4b")
    ax.fill_between(sub_df["week_start"], sub_df["weighted_signal"], alpha=0.15, color="#e05c4b")
    ax.set_title(f"r/{sub}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Weighted Signal", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("FluCast - HHS Region 3 Weekly Weighted Flu Signal (2018)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved signal plot -> {OUTPUT_PLOT}")
plt.close()
