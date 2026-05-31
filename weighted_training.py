

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import shutil

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


MODEL_NAME = "roberta-base"

OUTPUT_DIR = "/content/roberta_flu_model"
FINAL_MODEL_DIR = "/content/final_roberta_flu_model"
LOG_DIR = "/content/logs"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


train_df = pd.read_csv("/content/train.csv")
val_df = pd.read_csv("/content/val.csv")
test_df = pd.read_csv("/content/test.csv")


train_df = train_df[["text", "label"]]
val_df = val_df[["text", "label"]]
test_df = test_df[["text", "label"]]


train_df = train_df.dropna(subset=["text", "label"])
val_df = val_df.dropna(subset=["text", "label"])
test_df = test_df.dropna(subset=["text", "label"])


train_df["text"] = train_df["text"].astype(str)
val_df["text"] = val_df["text"].astype(str)
test_df["text"] = test_df["text"].astype(str)


train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)


print("\nTrain label counts:")
print(train_df["label"].value_counts())

print("\nValidation label counts:")
print(val_df["label"].value_counts())

print("\nTest label counts:")
print(test_df["label"].value_counts())


negative_count = (train_df["label"] == 0).sum()
positive_count = (train_df["label"] == 1).sum()

positive_weight = negative_count / positive_count
CLASS_WEIGHTS = [1.0, positive_weight]

print("\nClass weights:")
print(CLASS_WEIGHTS)


train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])


train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        class_weights = torch.tensor(
            CLASS_WEIGHTS,
            dtype=torch.float,
            device=logits.device
        )

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=LOG_DIR,
    logging_steps=20,
    save_total_limit=2,
    report_to="none",
)


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()


print("\nValidation results:")
val_results = trainer.evaluate(val_dataset)
print(val_results)


print("\nTest results:")
test_results = trainer.evaluate(test_dataset)
print(test_results)


predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids


cm = confusion_matrix(true_labels, pred_labels)

print("\nConfusion matrix:")
print(cm)


tn, fp, fn, tp = cm.ravel()

print("\nDetailed confusion matrix:")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")


trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"\nSaved model to {FINAL_MODEL_DIR}")





shutil.make_archive(
    "/content/final_roberta_flu_model",
    "zip",
    FINAL_MODEL_DIR
)

print("Zipped model saved to /content/final_roberta_flu_model.zip")