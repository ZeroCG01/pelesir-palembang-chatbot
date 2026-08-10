"""
train_ner.py — Stage 3: Fine-Tuning Model NER (XLM-RoBERTa) Colab-Ready
Khusus melatih ulang modul Named Entity Recognition (NER) dengan Dataset Latih Augmentasi (train_ner_aug.json).

Fitur & Jaminan Kualitas:
1. Dataset Latih: ml/data/processed/train_ner_aug.json (4.214 kalimat).
2. Dataset Validasi & Uji: val_ner_legacy.json & test_ner_legacy_583.json (100% terkunci).
3. Anti-Overfitting: Early stopping (patience=3), weight decay (0.01), dropout (0.25), warmup (0.10).
4. Loss Weighting Seimbang: CrossEntropyLoss terbobot moderat pada token LOCATION & PRICE.
5. Model Terbaik: Menyimpan checkpoint model dengan Macro F1 Validasi tertinggi.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1, precision_score as seq_prec, recall_score as seq_rec

# ============================================================
# 1. KONFIGURASI HYPERPARAMETER & ENVIRONMENT
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
MAX_EPOCHS = 15
PATIENCE = 3
LR_BACKBONE = 2e-5
LR_CLASSIFIER = 8e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_CLIP = 1.0

DATA_DIR = "ml/data/processed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_ner_aug.json")
VAL_FILE   = os.path.join(DATA_DIR, "val_ner_legacy.json")
TEST_FILE  = os.path.join(DATA_DIR, "test_ner_legacy_583.json")

OUTPUT_DIR = "output/ner"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(SEED)


# ============================================================
# 2. DATASET PYTORCH DENGAN ALIGNMENT SUB-TOKEN SUBWORD
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
                tag_str = tags[word_idx]
                labels.append(self.tag2id.get(tag_str, self.tag2id["O"]))
            else:
                # Sub-token berikutnya diabaikan dari perhitungan loss (-100)
                labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# ============================================================
# 3. PERHITUNGAN CLASS WEIGHT LOSS (MODERAT)
# ============================================================
def calculate_class_weights():
    """
    Bobot moderat untuk menyeimbangkan penalti klasifikasi tanpa menyebabkan over-prediction:
    - LOCATION: 1.30 (memberi perhatian ekstra pada batas wilayah)
    - PRICE: 1.15 (memberi perhatian ekstra pada multi-token nominal)
    - Lainnya: 1.00
    """
    weights = torch.ones(len(NER_TAGS), dtype=torch.float)
    weights[NER_TAG2ID["B-LOCATION"]] = 1.30
    weights[NER_TAG2ID["I-LOCATION"]] = 1.30
    weights[NER_TAG2ID["B-PRICE"]]    = 1.15
    weights[NER_TAG2ID["I-PRICE"]]    = 1.15
    return weights.to(DEVICE)


# ============================================================
# 4. LOOP TRAINING UTAMA
# ============================================================
def train():
    print("=" * 80)
    print("🚀 PELESIR PALEMBANG — TRAINING MODEL NER XLM-RoBERTa (STAGE 3)")
    print(f"Device    : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Seed      : {SEED} | Max Epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")
    print(f"Dataset   : Train Aug ({TRAIN_FILE}) | Val ({VAL_FILE}) | Test ({TEST_FILE})")
    print("=" * 80)

    # Validasi keberadaan file
    for p in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not os.path.exists(p):
            print(f"❌ File tidak ditemukan: {p}")
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(NER_TAGS),
        id2label=NER_ID2TAG,
        label2id=NER_TAG2ID,
        classifier_dropout=0.25
    ).to(DEVICE)

    train_ds = NERDataset(TRAIN_FILE, tokenizer, MAX_LEN, NER_TAG2ID)
    val_ds   = NERDataset(VAL_FILE, tokenizer, MAX_LEN, NER_TAG2ID)
    test_ds  = NERDataset(TEST_FILE, tokenizer, MAX_LEN, NER_TAG2ID)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📦 Total Data: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")

    # Optimizer Grouped Parameters
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and not any(nd in n for nd in no_decay)],
            "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and any(nd in n for nd in no_decay)],
            "lr": LR_BACKBONE, "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": LR_CLASSIFIER, "weight_decay": WEIGHT_DECAY
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )

    class_weights = calculate_class_weights()
    loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    best_val_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_location_f1": [],
        "val_price_f1": [],
        "val_precision": [],
        "val_recall": []
    }

    print("\n⏳ Memulai fine-tuning per-epoch...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fct(logits.view(-1, len(NER_TAGS)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluasi Validasi
        model.eval()
        val_true, val_pred = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                v_loss = loss_fct(logits.view(-1, len(NER_TAGS)), labels.view(-1))
                total_val_loss += v_loss.item()

                preds = torch.argmax(logits, dim=2).cpu().numpy()
                lbls = labels.cpu().numpy()

                for i in range(len(preds)):
                    t_seq, p_seq = [], []
                    for j in range(len(preds[i])):
                        if lbls[i][j] != -100:
                            t_seq.append(NER_ID2TAG[lbls[i][j]])
                            p_seq.append(NER_ID2TAG[preds[i][j]])
                    val_true.append(t_seq)
                    val_pred.append(p_seq)

        avg_val_loss = total_val_loss / len(val_loader)
        val_macro_f1 = seq_f1(val_true, val_pred)
        val_prec = seq_prec(val_true, val_pred)
        val_rec  = seq_rec(val_true, val_pred)

        # Hitung F1 per label pada val
        from seqeval.metrics import classification_report as seq_report_dict
        val_report = seq_report_dict(val_true, val_pred, output_dict=True)
        val_loc_f1 = val_report.get("LOCATION", {}).get("f1-score", 0.0)
        val_prc_f1 = val_report.get("PRICE", {}).get("f1-score", 0.0)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_location_f1"].append(val_loc_f1)
        history["val_price_f1"].append(val_prc_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_macro_f1*100:6.2f}% | LOC F1: {val_loc_f1*100:5.1f}% | PRC F1: {val_prc_f1*100:5.1f}%")

        # Cek Checkpoint Model Terbaik (Berdasarkan Val Macro F1)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ⭐ MODEL TERBAIK DISIMPAN (Val Macro F1: {val_macro_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early Stopping dipicu pada Epoch {epoch} (Patience {PATIENCE} tercapai).")
                break

    print("\n" + "=" * 80)
    print(f"🏆 TRAINING SELESAI | Model Terbaik dari Epoch {best_epoch} (Val Macro F1: {best_val_macro_f1*100:.2f}%)")
    print("=" * 80)

    # ============================================================
    # 5. EVALUASI FINAL PADA TEST SET (TEST_NER_LEGACY_583.JSON)
    # ============================================================
    print("\n📊 MEMUAT CHECKPOINT TERBAIK & MENGEVALUASI DATA UJI (583 Entitas)...")
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
    print("\n" + report_str)

    # Simpan hasil teks laporan
    report_file = os.path.join(REPORTS_DIR, "ner_test_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"NER EVALUATION REPORT (Stage 3 Trained Model)\n")
        f.write(f"Best Epoch: {best_epoch} | Best Val Macro F1: {best_val_macro_f1:.4f}\n\n")
        f.write(report_str)

    # ============================================================
    # 6. PLOTTING GRAFIK & LEARNING CURVE
    # ============================================================
    save_training_plots(history, test_true, test_pred)

    # Simpan metrics JSON
    metrics_json_path = os.path.join(REPORTS_DIR, "ner_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "history": history
        }, f, indent=4)

    print(f"\n💾 Model & Laporan Evaluasi Tersimpan di: {OUTPUT_DIR} dan {REPORTS_DIR}")


def save_training_plots(history, test_true, test_pred):
    epochs_range = history["epoch"]

    # 1. Loss & F1 Learning Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    ax1.set_title('Learning Curve: Loss Training vs Validation (NER)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs_range, [f * 100 for f in history['val_macro_f1']], 'g-o', label='Val Macro F1', linewidth=2)
    ax2.plot(epochs_range, [f * 100 for f in history['val_location_f1']], 'm--s', label='Val LOCATION F1', linewidth=1.5)
    ax2.plot(epochs_range, [f * 100 for f in history['val_price_f1']], 'c--^', label='Val PRICE F1', linewidth=1.5)
    ax2.set_title('Learning Curve: Validation F1 Scores (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curve_path = os.path.join(REPORTS_DIR, "ner_training_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"📈 Grafik Learning Curve tersimpan: {curve_path}")

    # 2. Per-Entity Bar Chart (Precision, Recall, F1)
    from seqeval.metrics import classification_report as seq_report_dict
    report_dict = seq_report_dict(test_true, test_pred, output_dict=True)

    entities = [e for e in TARGET_ENTITIES if e in report_dict]
    precisions = [report_dict[e]['precision'] * 100 for e in entities]
    recalls = [report_dict[e]['recall'] * 100 for e in entities]
    f1_scores = [report_dict[e]['f1-score'] * 100 for e in entities]

    x = np.arange(len(entities))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label='Precision (%)', color='#2b5c8f')
    plt.bar(x, recalls, width, label='Recall (%)', color='#469b88')
    plt.bar(x + width, f1_scores, width, label='F1-Score (%)', color='#d95f02')

    plt.xlabel('Tipe Entitas NER', fontweight='bold')
    plt.ylabel('Skor (%)', fontweight='bold')
    plt.title('Evaluasi Performa Per-Entitas Model NER pada Test Set (583 Baseline)', fontsize=13, fontweight='bold')
    plt.xticks(x, entities, fontweight='bold')
    plt.ylim(0, 115)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(entities)):
        plt.text(x[i] - width, precisions[i] + 1.5, f"{precisions[i]:.1f}%", ha='center', fontsize=8)
        plt.text(x[i], recalls[i] + 1.5, f"{recalls[i]:.1f}%", ha='center', fontsize=8)
        plt.text(x[i] + width, f1_scores[i] + 1.5, f"{f1_scores[i]:.1f}%", ha='center', fontsize=8)

    plt.tight_layout()
    bar_path = os.path.join(REPORTS_DIR, "ner_entity_performance.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"📊 Grafik Bar Chart Per-Entitas tersimpan: {bar_path}")


if __name__ == "__main__":
    train()
