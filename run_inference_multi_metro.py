# run_inference_multi_metro.py
# Runs the trained RoBERTa classifier on raw Reddit 2018 data for NYC, Philadelphia, Houston.
# Keyword filters first, then batched inference, outputs one labeled CSV per metro.
#
# Set MODEL_DIR below to wherever you unzipped final_roberta_flu_model.zip from Colab.
# After this runs, feed each output CSV into aggregate_signal.py.

import os
import re
import glob
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "final_roberta_flu_model"
BATCH_SIZE = 32
MAX_LENGTH = 256
DATA_DIR = "data/filtered"
OUTPUT_DIR = "results"

METROS = {
    "NYC":          "nyc",
    "Philadelphia": "philadelphia",
    "Houston":      "houston",
}

FLU_KEYWORDS = [
    "antiviral", "antivirals", "bad cough", "been sick all week", "body ache",
    "body aches", "called out sick", "chills", "cold and flu", "cold or flu",
    "cold/flu", "congested", "congestion", "cough", "coughing", "coughs",
    "coworkers are sick", "diarrhea", "difficulty breathing", "doctor appointment",
    "dry cough", "emergency room", "epidemic", "er visit", "everyone has it",
    "everyone is sick", "fatigue", "fever", "feverish", "flu", "get a flu shot",
    "going around", "going to the doctor", "got my flu shot", "got tested",
    "high fever", "home sick", "i am  sick", "i am sick", "i feel awful",
    "i feel sick", "i feel terrible", "i got the flu", "i have the flu",
    "i think i have the flu", "i'm sick", "i've been sick", "influenza",
    "joint pain", "malaise", "minuteclinic", "muscle aches", "muscle pain",
    "nausea", "night sweats", "oseltamivir", "out sick", "outbreak",
    "people are sick", "persistent cough", "rapid flu test", "runny nose",
    "scratchy throat", "seasonal flu", "shivering", "shivers",
    "shortness of breath", "sinus infection", "sinus pressure", "sneezing",
    "sore throat", "stomach bug", "stomach virus", "strep throat", "stuffy nose",
    "sweats", "tamiflu", "the flu", "throat hurts", "throat pain", "throwing up",
    "trouble breathing", "upper respiratory infection", "urgent care", "vaccinated",
    "vaccination", "vaccine", "viral illness", "viral infection",
    "virus going around", "vomiting", "went to the doctor", "wheezing",
]

# longest keywords first so multi-word phrases match before single words
_KW_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(kw) for kw in sorted(FLU_KEYWORDS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)


def load_metro_data(metro_dir: str, subreddit: str) -> pd.DataFrame:
    comments_files = sorted(glob.glob(os.path.join(metro_dir, "comments", "*.csv")))
    submission_files = sorted(glob.glob(os.path.join(metro_dir, "submissions", "*.csv")))

    dfs = []

    for f in comments_files:
        df = pd.read_csv(f, low_memory=False)
        df["source_type"] = "comment"
        df["text"] = df["body"].fillna("").astype(str)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    for f in submission_files:
        df = pd.read_csv(f, low_memory=False)
        df["source_type"] = "submission"
        title = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)
        selftext = df.get("selftext", pd.Series([""] * len(df))).fillna("").astype(str)
        df["text"] = (title + " " + selftext).str.strip()
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["subreddit"] = subreddit
    combined["date"] = pd.to_datetime(combined["created_utc"], unit="s", errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    return combined[["subreddit", "source_type", "date", "id", "text", "source_file", "permalink", "score"]]


def apply_keyword_filter(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["text"].apply(lambda t: bool(_KW_PATTERN.search(t)))
    filtered = df[mask].copy()
    filtered["matched_keywords"] = filtered["text"].apply(
        lambda t: ", ".join(sorted(set(m.lower() for m in _KW_PATTERN.findall(t))))
    )
    return filtered.reset_index(drop=True)


def run_inference(texts, model, tokenizer, device):
    all_preds, all_prob0, all_prob1 = [], [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_prob0.extend(probs[:, 0].tolist())
        all_prob1.extend(probs[:, 1].tolist())

    return all_preds, all_prob0, all_prob1


def process_metro(metro_name: str, subreddit: str, model, tokenizer, device):
    metro_dir = os.path.join(DATA_DIR, metro_name)
    print(f"\n{'='*60}")
    print(f"Processing {metro_name} (r/{subreddit})")
    print(f"{'='*60}")

    df = load_metro_data(metro_dir, subreddit)
    print(f"  Loaded {len(df):,} total posts")

    df = apply_keyword_filter(df)
    print(f"  After keyword filter: {len(df):,} posts")

    if len(df) == 0:
        print("  No keyword matches, skipping.")
        return

    texts = df["text"].tolist()
    print(f"  Running inference (batch_size={BATCH_SIZE})...")
    preds, prob0, prob1 = run_inference(texts, model, tokenizer, device)

    df["roberta_predicted_class"] = preds
    df["roberta_prob_not_flu"] = prob0
    df["roberta_prob_flu"] = prob1
    df["roberta_flu_label"] = preds

    out_path = os.path.join(OUTPUT_DIR, f"{metro_name.lower()}_2018_roberta_labeled.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    print(f"  Flu-relevant: {sum(preds):,} / {len(preds):,}  ({100*sum(preds)/len(preds):.1f}%)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: '{MODEL_DIR}'\n"
            "Set MODEL_DIR at the top of this script to the unzipped model path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {MODEL_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.")

    for metro_name, subreddit in METROS.items():
        process_metro(metro_name, subreddit, model, tokenizer, device)

    print("\nDone. Run aggregate_signal.py on each output CSV for weekly signals.")


if __name__ == "__main__":
    main()
