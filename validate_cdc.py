# validate_cdc.py
# Pulls CDC ILINet 2018 data for HHS Region 3 from CMU DELPHI Epidata,
# aligns it with the FluCast weekly signal, and computes Pearson/Spearman correlation.
#
# Outputs:
#   results/region3_cdc_comparison.png
#   results/region3_correlation.csv

import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

SIGNAL_CSV = "results/region3_weekly_signal.csv"
OUTPUT_PLOT = "results/region3_cdc_comparison.png"
OUTPUT_CORR = "results/region3_correlation.csv"

os.makedirs("results", exist_ok=True)


def fetch_cdc_data():
    print("Fetching CDC ILINet data from CMU DELPHI Epidata...")
    url = (
        "https://api.delphi.cmu.edu/epidata/fluview/"
        "?regions=hhs3&epiweeks=201740-201852"
    )
    with urllib.request.urlopen(url) as resp:
        payload = json.load(resp)

    assert payload["result"] == 1, f"Epidata API error: {payload.get('message')}"
    rows = payload["epidata"]
    print(f"  Got {len(rows)} epiweek rows from CDC.")

    cdc = pd.DataFrame(rows)[["epiweek", "wili", "ili"]]
    cdc = cdc.rename(columns={"wili": "cdc_wili", "ili": "cdc_ili"})

    cdc["year"] = (cdc["epiweek"] // 100).astype(int)
    cdc["mmwr_week"] = (cdc["epiweek"] % 100).astype(int)

    # MMWR weeks and ISO weeks are off by ~1 day, close enough for correlation
    cdc["week_start"] = pd.to_datetime(
        cdc["year"].astype(str) + "-W" + cdc["mmwr_week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )

    return cdc.drop(columns=["epiweek", "year", "mmwr_week"])


def load_flucast_signal():
    print(f"Loading FluCast signal from {SIGNAL_CSV}...")
    sig = pd.read_csv(SIGNAL_CSV, parse_dates=["week_start"])

    region = (
        sig.groupby("week_start")
        .agg(
            total_posts=("raw_post_count", "sum"),
            total_relevant=("relevant_post_count", "sum"),
        )
        .reset_index()
    )

    # reconstruct prob_sum from stored weighted_signal * post_count
    sig["prob_sum"] = sig["weighted_signal"] * sig["raw_post_count"]
    prob_by_week = sig.groupby("week_start")["prob_sum"].sum().reset_index()
    region = region.merge(prob_by_week, on="week_start")
    region["flucast_signal"] = region["prob_sum"] / region["total_posts"]
    region = region.drop(columns=["prob_sum"])

    print(f"  {len(region)} FluCast weeks loaded.")
    return region


def align_and_correlate(cdc, flucast):
    merged = pd.merge(cdc, flucast, on="week_start", how="inner")
    merged = merged.sort_values("week_start").reset_index(drop=True)
    print(f"\nAligned {len(merged)} overlapping weeks for correlation.")

    if len(merged) < 3:
        print("Too few overlapping weeks for meaningful correlation.")
        return merged, {}

    pearson_r, pearson_p = stats.pearsonr(merged["cdc_wili"], merged["flucast_signal"])
    spearman_r, spearman_p = stats.spearmanr(merged["cdc_wili"], merged["flucast_signal"])

    corr = {
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "n_weeks": len(merged),
    }
    print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4f})")
    print(f"  Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4f})")
    return merged, corr


def plot_comparison(cdc, flucast, merged, corr):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    ax1 = axes[0]
    ax1.plot(cdc["week_start"], cdc["cdc_wili"], color="#1f77b4", linewidth=2, marker="o", markersize=4)
    ax1.fill_between(cdc["week_start"], cdc["cdc_wili"], alpha=0.15, color="#1f77b4")
    ax1.set_title("CDC ILINet - HHS Region 3 Weighted ILI % (2017-18 Season)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("% ILI Visits", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.plot(flucast["week_start"], flucast["flucast_signal"], color="#e05c4b", linewidth=2, marker="o", markersize=4)
    ax2.fill_between(flucast["week_start"], flucast["flucast_signal"], alpha=0.15, color="#e05c4b")
    ax2.set_title("FluCast - HHS Region 3 Weighted Signal (Reddit 2018)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Weighted Signal", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3 = axes[2]
    if len(merged) >= 3:
        ax3.scatter(merged["cdc_wili"], merged["flucast_signal"], color="#2ca02c", alpha=0.7, edgecolors="white", linewidth=0.5, s=60)
        m, b = np.polyfit(merged["cdc_wili"], merged["flucast_signal"], 1)
        x_line = np.linspace(merged["cdc_wili"].min(), merged["cdc_wili"].max(), 100)
        ax3.plot(x_line, m * x_line + b, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
        corr_text = (
            f"Pearson r = {corr.get('pearson_r', 'N/A')}  (p = {corr.get('pearson_p', 'N/A')})\n"
            f"Spearman r = {corr.get('spearman_r', 'N/A')}  (p = {corr.get('spearman_p', 'N/A')})\n"
            f"n = {corr.get('n_weeks', len(merged))} weeks"
        )
        ax3.text(0.05, 0.95, corr_text, transform=ax3.transAxes, fontsize=9,
                 verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    ax3.set_xlabel("CDC wILI %", fontsize=9)
    ax3.set_ylabel("FluCast Weighted Signal", fontsize=9)
    ax3.set_title("FluCast Signal vs. CDC ILI % (Overlapping Weeks)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("FluCast vs. CDC ILINet - HHS Region 3 Validation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot -> {OUTPUT_PLOT}")
    plt.close()


def main():
    cdc = fetch_cdc_data()
    flucast = load_flucast_signal()
    merged, corr = align_and_correlate(cdc, flucast)

    if corr:
        corr_df = pd.DataFrame([corr])
        corr_df.to_csv(OUTPUT_CORR, index=False)
        print(f"Saved correlation stats -> {OUTPUT_CORR}")

    plot_comparison(cdc, flucast, merged, corr)
    print("\nDone.")


if __name__ == "__main__":
    main()
