"""
=============================================================
PELESIR PALEMBANG — NER-ONLY RETRAINING SCRIPT (Google Colab)
=============================================================
Jalankan file ini di Google Colab (dengan GPU T4).
Khusus melatih model NER (XLM-RoBERTa) dengan dataset Legacy (583 baseline).

DATASET YANG DIGUNAKAN:
- train_ner_legacy.json
- val_ner_legacy.json
- test_ner_legacy_583.json
=============================================================
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

# ============================================================
# KONFIGURASI HYPERPARAMETER & PATHS
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 25
LR = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3
WARMUP_RATIO = 0.1

HF_REPO_ID = "ZeroCG/pelesir-ner"
DATA_DIR = "ml/data/processed"
OUTPUT_DIR = "output/ner"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

NER_TAGS = [
    "O",
    "B-DESTINATION", "I-DESTINATION",
    "B-CATEGORY",    "I-CATEGORY",
    "B-LOCATION",    "I-LOCATION",
    "B-TIME",        "I-TIME",
    "B-PRICE",       "I-PRICE"
]
NER_TAG2ID = {tag: i for i, tag in enumerate(NER_TAGS)}
NER_ID2TAG = {i: tag for tag, i in NER_TAG2ID.items()}
TARGET_ENTITIES = ["DESTINATION", "TIME", "CATEGORY", "PRICE", "LOCATION"]

# Set Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# DATASET CLASS
# ============================================================
class NERDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len, tag2id):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        tags = self.data[idx]['tags']

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids()
        previous_word_idx = None
        labels = []

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(self.tag2id[tags[word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# ============================================================
# MAIN TRAINING & PLOTTING LOOP
# ============================================================
def train_ner():
    print("=" * 60)
    print("🚀 PELESIR PALEMBANG — RETRAINING MODEL NER (XLM-RoBERTa)")
    print(f"Device: {DEVICE}")
    print(f"Seed  : {SEED}")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, "train_ner_legacy.json")
    val_path = os.path.join(DATA_DIR, "val_ner_legacy.json")
    test_path = os.path.join(DATA_DIR, "test_ner_legacy_583.json")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"❌ ERROR: File {p} tidak ditemukan! Pastikan dataset sudah di-copy.")
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(NER_TAGS),
        id2label=NER_ID2TAG,
        label2id=NER_TAG2ID,
        classifier_dropout=0.3
    ).to(DEVICE)

    train_ds = NERDataset(train_path, tokenizer, MAX_LEN, NER_TAG2ID)
    val_ds = NERDataset(val_path, tokenizer, MAX_LEN, NER_TAG2ID)
    test_ds = NERDataset(test_path, tokenizer, MAX_LEN, NER_TAG2ID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and not any(nd in n for nd in no_decay)],
            "lr": LR, "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and any(nd in n for nd in no_decay)],
            "lr": LR, "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": LR * 5, "weight_decay": WEIGHT_DECAY
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\nMulai Training ({EPOCHS} Epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_true, val_pred = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
                lbls = labels.cpu().numpy()

                for i in range(len(preds)):
                    t_seq, p_seq = [], []
                    for j in range(len(preds[i])):
                        if lbls[i][j] != -100:
                            t_seq.append(NER_ID2TAG[lbls[i][j]])
                            p_seq.append(NER_ID2TAG[preds[i][j]])
                    val_true.append(t_seq)
                    val_pred.append(p_seq)

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = seq_f1(val_true, val_pred)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping dipicu di Epoch {epoch+1}")
                break

    # ============================================================
    # FINAL TEST EVALUATION (TEST_NER_LEGACY_583.JSON)
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 EVALUASI MODEL NER PADA TEST SET (583 Baseline)")
    print("=" * 60)

    best_model = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
    best_model.eval()

    test_true, test_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
            lbls = labels.cpu().numpy()

            for i in range(len(preds)):
                t_seq, p_seq = [], []
                for j in range(len(preds[i])):
                    if lbls[i][j] != -100:
                        t_seq.append(NER_ID2TAG[lbls[i][j]])
                        p_seq.append(NER_ID2TAG[preds[i][j]])
                test_true.append(t_seq)
                test_pred.append(p_seq)

    report_str = seq_report(test_true, test_pred, digits=4)
    print(report_str)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "ner_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str + "\n")

    # ============================================================
    # PLOTTING GRAFIK REPORT
    # ============================================================
    generate_reports(history, test_true, test_pred)

    print(f"\n✅ Model & Grafik NER berhasil disimpan di: {OUTPUT_DIR}")
    return OUTPUT_DIR


def generate_reports(history, test_true, test_pred):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Grafik 1: Training & Validation Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    ax1.set_title('NER: Loss Training vs Validation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs_range, history['val_f1'], 'g-o', label='Val F1 Score', linewidth=2)
    ax2.set_title('NER: Validation F1 Score Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path_loss = os.path.join(REPORTS_DIR, 'ner_training_curves.png')
    plt.savefig(plot_path_loss, dpi=300)
    plt.close()
    print(f"📈 Grafik Loss & F1 Curve disimpan: {plot_path_loss}")

    # Grafik 2: Per-Entity Performance Bar Chart
    from seqeval.metrics import classification_report as seq_report_dict
    report_dict = seq_report_dict(test_true, test_pred, output_dict=True)

    entities = [e for e in TARGET_ENTITIES if e in report_dict]
    precisions = [report_dict[e]['precision'] for e in entities]
    recalls = [report_dict[e]['recall'] for e in entities]
    f1_scores = [report_dict[e]['f1-score'] for e in entities]

    x = np.arange(len(entities))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label='Precision', color='#2b5c8f')
    plt.bar(x, recalls, width, label='Recall', color='#469b88')
    plt.bar(x + width, f1_scores, width, label='F1-Score', color='#d95f02')

    plt.xlabel('Tipe Entitas NER', fontweight='bold')
    plt.ylabel('Skor', fontweight='bold')
    plt.title('Evaluasi Performa Per-Entitas Model NER (Legacy 583)', fontsize=13, fontweight='bold')
    plt.xticks(x, entities, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(entities)):
        plt.text(x[i] - width, precisions[i] + 0.02, f"{precisions[i]:.2f}", ha='center', fontsize=8)
        plt.text(x[i], recalls[i] + 0.02, f"{recalls[i]:.2f}", ha='center', fontsize=8)
        plt.text(x[i] + width, f1_scores[i] + 0.02, f"{f1_scores[i]:.2f}", ha='center', fontsize=8)

    plt.tight_layout()
    plot_path_bar = os.path.join(REPORTS_DIR, 'ner_entity_performance.png')
    plt.savefig(plot_path_bar, dpi=300)
    plt.close()
    print(f"📊 Grafik Per-Entitas Bar Chart disimpan: {plot_path_bar}")

if __name__ == "__main__":
    train_ner()
