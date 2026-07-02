#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot Text Classification on AG News using BERT + Prompt Learning.
Runs 5-shot, 10-shot, 20-shot and prints accuracy results.
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,  # 改用这个
)
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Loading AG News from local CSV files...")
train_csv_path = "./data/train.csv"
test_csv_path  = "./data/test.csv"

train_df = pd.read_csv(train_csv_path, header=None, names=["label", "title", "description"])
test_df  = pd.read_csv(test_csv_path,  header=None, names=["label", "title", "description"])

train_df["text"] = train_df["title"] + ". " + train_df["description"]
test_df["text"]  = test_df["title"] + ". " + test_df["description"]

train_df = train_df[["label", "text"]]
test_df  = test_df[["label", "text"]]

train_data = Dataset.from_pandas(train_df)
test_data  = Dataset.from_pandas(test_df)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
VERBALIZER = {0: "world", 1: "sports", 2: "business", 3: "tech"}

MODEL_PATH = "./bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

label_word_ids = {}
for label, word in VERBALIZER.items():
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    label_word_ids[label] = token_ids[0]
id_to_label = {v: k for k, v in label_word_ids.items()}

def sample_few_shot_data(dataset, k_shot):
    samples = []
    for label in range(4):
        label_data = dataset.filter(lambda x: x["label"] == label)
        selected = label_data.shuffle(seed=SEED).select(range(k_shot))
        samples.extend(selected)
    return samples

def preprocess_function(examples):
    texts = examples["text"]
    labels = examples["label"]
    prompts = [f"{text} This news is about [MASK]." for text in texts]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=128,
        return_tensors=None
    )
    mask_token_id = tokenizer.mask_token_id
    new_labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        mask_indices = [idx for idx, token_id in enumerate(input_ids) if token_id == mask_token_id]
        label_ids = [-100] * len(input_ids)
        if mask_indices:
            mask_pos = mask_indices[0]
            label_ids[mask_pos] = label_word_ids[labels[i]]
        new_labels.append(label_ids)
    tokenized["labels"] = new_labels
    return tokenized

def evaluate_model(model, tokenizer, test_dataset, batch_size=32):
    model.eval()
    all_preds = []
    all_labels = []
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        texts = batch["text"]
        true_labels = batch["label"]
        prompts = [f"{text} This news is about [MASK]." for text in texts]
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
        batch_preds = []
        for b in range(len(texts)):
            pos = mask_positions[1][mask_positions[0] == b]
            if len(pos) == 0:
                batch_preds.append(-1)
                continue
            pos = pos[0].item()
            mask_logits = logits[b, pos, :]
            pred_token_id = torch.argmax(mask_logits).item()
            pred_label = id_to_label.get(pred_token_id, -1)
            batch_preds.append(pred_label)
        all_preds.extend(batch_preds)
        all_labels.extend(true_labels)
    valid_indices = [i for i, p in enumerate(all_preds) if p != -1]
    if not valid_indices:
        return 0.0
    acc = accuracy_score(
        [all_labels[i] for i in valid_indices],
        [all_preds[i] for i in valid_indices]
    )
    return acc

shot_values = [5, 10, 20]
results = {}

for k in shot_values:
    print(f"\n========== {k}-shot ==========")
    few_train = sample_few_shot_data(train_data, k)
    print(f"Training samples: {len(few_train)} ( {k} per class )")
    few_train_dataset = Dataset.from_list(few_train)
    tokenized_train = few_train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=few_train_dataset.column_names
    )
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_args = TrainingArguments(
        output_dir=f"./results_{k}shot",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    # 使用 DataCollatorForTokenClassification，自动填充 labels
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding=True,
        max_length=128,  # 可选，与 tokenizer 的 max_length 一致
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )
    print("Training...")
    trainer.train()
    print("Evaluating on test set...")
    test_acc = evaluate_model(model, tokenizer, test_data)
    print(f"Test Accuracy: {test_acc:.4f}")
    results[k] = {"train_samples": len(few_train), "accuracy": test_acc}
    del model
    torch.cuda.empty_cache()

print("\n" + "="*50)
print("Final Results for Report:")
print("="*50)
print(f"{'Shot':<10} {'Train Samples':<15} {'Test Accuracy':<15}")
print("-"*50)
for k in shot_values:
    info = results[k]
    print(f"{k:<10} {info['train_samples']:<15} {info['accuracy']:.4f}")
print("="*50)
print("\nAll done. Results are ready for your report.")