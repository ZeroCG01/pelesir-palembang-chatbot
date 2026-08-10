"""
train_ner_multiseed.py — Stage 3b: Multi-Seed NER Training with Location-Aware Validation Checkpoint Selection
Melatih model XLM-RoBERTa pada 5 random seed berbeda (13, 42, 123, 2024, 7) untuk mengejar F1 LOCATION >= 0.80
tanpa data baru, tanpa leakage, dan dengan seleksi model 100% berbasis Validation Set.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
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
from seqeval.metrics import (
    classification_report as seq_report,
    f1_score as seq_f1,
    precision_score as seq_prec,
    recall_score as seq_rec
)
from seqeval.metrics import classification_report as seq_report_dict

# ============================================================
# 1. KONFIGURASI HYPERPARAMETER
# ============================================================
SEEDS = [13, 42, 123, 2024, 7]
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
VAL_TOLERANCE = 0.003  # 0.3% toleransi absolut dari Peak Val Macro-F1

DEFAULT_HF_REPO_ID = "ZeroCG/pelesir-ner"
DATA_DIR = "ml/data/processed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_ner_aug.json")
VAL_FILE   = os.path.join(DATA_DIR, "val_ner_legacy.json")
TEST_FILE  = os.path.join(DATA_DIR, "test_ner_legacy_583.json")

OUTPUT_BASE_DIR = "output/ner_multiseed"
OUTPUT_FINAL_DIR = "output/ner"
REPORTS_DIR = os.path.join(OUTPUT_BASE_DIR, "reports")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FINAL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

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


# ============================================================
# 2. DATASET PYTORCH SUB-TOKEN ALIGNMENT
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
                labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def calculate_class_weights():
    # Bobot moderat ringan (data sudah diseimbangkan di Stage 1)
    weights = torch.ones(len(NER_TAGS), dtype=torch.float)
    weights[NER_TAG2ID["B-LOCATION"]] = 1.25
    weights[NER_TAG2ID["I-LOCATION"]] = 1.25
    weights[NER_TAG2ID["B-PRICE"]]    = 1.10
    weights[NER_TAG2ID["I-PRICE"]]    = 1.10
    return weights.to(DEVICE)


# ============================================================
# 3. TRAINING & SELEKSI CHECKPOINT PER SEED
# ============================================================
def train_single_seed(seed, tokenizer, train_ds, val_ds, test_ds):
    print("\n" + "=" * 75)
    print(f"🌱 MEMULAI TRAINING NER DENGAN SEED = {seed}")
    print("=" * 75)
    set_seed(seed)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(NER_TAGS),
        id2label=NER_ID2TAG,
        label2id=NER_TAG2ID,
        classifier_dropout=0.25
    ).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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

    epoch_checkpoints = {}  # epoch -> {'val_macro_f1', 'val_loc_f1', 'model_state', 'val_loss', 'train_loss'}
    patience_counter = 0
    best_raw_val_macro = 0.0

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_macro_f1": [], "val_location_f1": [], "val_price_f1": []
    }

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

        val_rep = seq_report_dict(val_true, val_pred, output_dict=True)
        val_loc_f1 = val_rep.get("LOCATION", {}).get("f1-score", 0.0)
        val_prc_f1 = val_rep.get("PRICE", {}).get("f1-score", 0.0)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_location_f1"].append(val_loc_f1)
        history["val_price_f1"].append(val_prc_f1)

        # Simpan state dict model per-epoch ke memori/disk
        epoch_dir = os.path.join(OUTPUT_BASE_DIR, f"temp_seed_{seed}_epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)

        epoch_checkpoints[epoch] = {
            "val_macro_f1": val_macro_f1,
            "val_loc_f1": val_loc_f1,
            "val_prc_f1": val_prc_f1,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "epoch_dir": epoch_dir
        }

        print(f"Seed {seed:4d} | Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro: {val_macro_f1*100:5.2f}% | LOC F1: {val_loc_f1*100:5.1f}% | PRC F1: {val_prc_f1*100:5.1f}%")

        # Early Stopping Check (Berdasarkan Val Macro F1)
        if val_macro_f1 > best_raw_val_macro:
            best_raw_val_macro = val_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️ Early stopping dipicu di Epoch {epoch} untuk Seed {seed}")
                break

    # ============================================================
    # ATURAN SELEKSI CHECKPOINT SADAR-LOCATION (HANYA PADA VAL)
    # ============================================================
    # 1. Cari peak validation Macro F1
    peak_val_macro = max(cp["val_macro_f1"] for cp in epoch_checkpoints.values())
    
    # 2. Filter kandidat dalam toleransi 0.3% (absolut) dari peak
    candidates = [
        (ep, cp) for ep, cp in epoch_checkpoints.items()
        if cp["val_macro_f1"] >= (peak_val_macro - VAL_TOLERANCE)
    ]

    # 3. Pilih kandidat dengan validation LOCATION F1 tertinggi
    selected_epoch, best_cp = max(candidates, key=lambda x: (x[1]["val_loc_f1"], x[1]["val_macro_f1"]))

    print(f"\n🎯 [Seleksi Checkpoint Val Seed {seed}]")
    print(f"  - Peak Val Macro F1: {peak_val_macro*100:.2f}%")
    print(f"  - Jumlah Kandidat (Toleransi 0.3%): {len(candidates)} epoch ({[c[0] for c in candidates]})")
    print(f"  - Epoch Terpilih: Epoch {selected_epoch} (Val Macro: {best_cp['val_macro_f1']*100:.2f}%, Val LOC F1: {best_cp['val_loc_f1']*100:.2f}%)")

    # Salin model terbaik seed ini ke direktori permanen seed
    seed_output_dir = os.path.join(OUTPUT_BASE_DIR, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    best_seed_model = AutoModelForTokenClassification.from_pretrained(best_cp["epoch_dir"]).to(DEVICE)
    best_seed_model.save_pretrained(seed_output_dir)
    tokenizer.save_pretrained(seed_output_dir)

    # Hapus folder temporary epoch
    for ep, cp in epoch_checkpoints.items():
        if os.path.exists(cp["epoch_dir"]):
            import shutil
            shutil.rmtree(cp["epoch_dir"], ignore_errors=True)

    # ============================================================
    # EVALUASI TEST SET UNTUK SEED INI
    # ============================================================
    best_seed_model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = best_seed_model(input_ids=input_ids, attention_mask=attention_mask)
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

    test_rep = seq_report_dict(test_true, test_pred, output_dict=True)
    test_macro_f1 = seq_f1(test_true, test_pred)
    test_micro_f1 = test_rep.get("micro avg", {}).get("f1-score", 0.0)
    test_weighted_f1 = test_rep.get("weighted avg", {}).get("f1-score", 0.0)

    # Plot learning curve seed ini
    plot_seed_learning_curve(seed, history, selected_epoch)

    return {
        "seed": seed,
        "selected_epoch": selected_epoch,
        "val_macro_f1": best_cp["val_macro_f1"],
        "val_loc_f1": best_cp["val_loc_f1"],
        "val_prc_f1": best_cp["val_prc_f1"],
        "train_loss": best_cp["train_loss"],
        "val_loss": best_cp["val_loss"],
        "test_macro_f1": test_macro_f1,
        "test_micro_f1": test_micro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_report": test_rep,
        "test_true": test_true,
        "test_pred": test_pred,
        "history": history,
        "model_dir": seed_output_dir
    }


def plot_seed_learning_curve(seed, history, selected_epoch):
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    ax1.axvline(x=selected_epoch, color='k', linestyle='--', alpha=0.7, label=f'Selected (Ep {selected_epoch})')
    ax1.set_title(f'Seed {seed}: Loss Curve', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(epochs, [f*100 for f in history['val_macro_f1']], 'g-o', label='Val Macro F1')
    ax2.plot(epochs, [f*100 for f in history['val_location_f1']], 'm--s', label='Val LOC F1')
    ax2.plot(epochs, [f*100 for f in history['val_price_f1']], 'c--^', label='Val PRC F1')
    ax2.axvline(x=selected_epoch, color='k', linestyle='--', alpha=0.7, label=f'Selected (Ep {selected_epoch})')
    ax2.set_title(f'Seed {seed}: Validation F1 Curve (%)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curve_path = os.path.join(REPORTS_DIR, f"ner_curve_seed_{seed}.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()


# ============================================================
# 4. MAIN MULTI-SEED WORKFLOW
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Seed NER XLM-RoBERTa Training (Stage 3b)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push final best model to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_HF_REPO_ID, help="Hugging Face Repo ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Token")
    args = parser.parse_args()

    print("=" * 85)
    print("🚀 PELESIR PALEMBANG — STAGE 3b: MULTI-SEED NER RETRAINING (5 SEEDS)")
    print(f"Device    : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Seeds List: {SEEDS}")
    print(f"Dataset   : Train Aug (4.214) | Val Legacy (392) | Test Legacy (331 / 583 entitas)")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = NERDataset(TRAIN_FILE, tokenizer, MAX_LEN, NER_TAG2ID)
    val_ds    = NERDataset(VAL_FILE, tokenizer, MAX_LEN, NER_TAG2ID)
    test_ds   = NERDataset(TEST_FILE, tokenizer, MAX_LEN, NER_TAG2ID)

    seed_results = []
    for s in SEEDS:
        res = train_single_seed(s, tokenizer, train_ds, val_ds, test_ds)
        seed_results.append(res)

    # ============================================================
    # 5. SELEKSI MODEL FINAL (100% BERBASIS VALIDATION SET)
    # ============================================================
    # Kriteria Final: Val Macro-F1 tertinggi (Tie-break: Val Location F1)
    final_seed_res = max(
        seed_results,
        key=lambda x: (x["val_macro_f1"], x["val_loc_f1"])
    )
    final_seed = final_seed_res["seed"]

    print("\n" + "=" * 90)
    print(f"🏆 SELEKSI MODEL FINAL (BERBASIS VALIDATION SET): SEED {final_seed}")
    print(f"  - Validation Macro F1 : {final_seed_res['val_macro_f1']*100:.2f}%")
    print(f"  - Validation LOC F1   : {final_seed_res['val_loc_f1']*100:.2f}%")
    print(f"  - Selected Epoch      : Epoch {final_seed_res['selected_epoch']}")
    print(f"  - Train Loss -> Val   : {final_seed_res['train_loss']:.4f} -> {final_seed_res['val_loss']:.4f} (Gap Sehat)")
    print("=" * 90)

    # Salin model final ke output/ner/ dan output/ner_final/
    import shutil
    for out_dst in [OUTPUT_FINAL_DIR, os.path.join(OUTPUT_BASE_DIR, "final_best")]:
        os.makedirs(out_dst, exist_ok=True)
        for fname in os.listdir(final_seed_res["model_dir"]):
            src_f = os.path.join(final_seed_res["model_dir"], fname)
            dst_f = os.path.join(out_dst, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)

    # ============================================================
    # 6. ANALISIS STATISTIK LINTAS-SEED (MEAN +- STD)
    # ============================================================
    csv_rows = []
    for r in seed_results:
        rep = r["test_report"]
        for label in TARGET_ENTITIES:
            if label in rep:
                csv_rows.append({
                    "seed": r["seed"],
                    "selected_epoch": r["selected_epoch"],
                    "val_macro_f1": r["val_macro_f1"],
                    "val_loc_f1": r["val_loc_f1"],
                    "label": label,
                    "precision": rep[label]["precision"],
                    "recall": rep[label]["recall"],
                    "f1": rep[label]["f1-score"],
                    "support": rep[label]["support"],
                    "test_macro_f1": r["test_macro_f1"],
                    "test_micro_f1": r["test_micro_f1"],
                    "test_weighted_f1": r["test_weighted_f1"]
                })

    df_results = pd.DataFrame(csv_rows)
    df_results.to_csv("ner_multiseed_results.csv", index=False)
    df_results.to_csv(os.path.join(OUTPUT_BASE_DIR, "ner_multiseed_results.csv"), index=False)
    df_results.to_csv(os.path.join("output/reports", "ner_multiseed_results.csv"), index=False)

    # Hitung Mean +- Std
    summary_stats = {}
    for label in TARGET_ENTITIES:
        sub = df_results[df_results["label"] == label]
        summary_stats[label] = {
            "precision_mean": float(sub["precision"].mean()),
            "precision_std": float(sub["precision"].std()),
            "recall_mean": float(sub["recall"].mean()),
            "recall_std": float(sub["recall"].std()),
            "f1_mean": float(sub["f1"].mean()),
            "f1_std": float(sub["f1"].std()),
            "support": int(sub["support"].iloc[0])
        }

    macro_scores = [r["test_macro_f1"] for r in seed_results]
    micro_scores = [r["test_micro_f1"] for r in seed_results]
    weighted_scores = [r["test_weighted_f1"] for r in seed_results]

    summary_stats["MACRO_AVG"] = {
        "f1_mean": float(np.mean(macro_scores)), "f1_std": float(np.std(macro_scores))
    }
    summary_stats["MICRO_AVG"] = {
        "f1_mean": float(np.mean(micro_scores)), "f1_std": float(np.std(micro_scores))
    }
    summary_stats["WEIGHTED_AVG"] = {
        "f1_mean": float(np.mean(weighted_scores)), "f1_std": float(np.std(weighted_scores))
    }

    # ============================================================
    # 7. ANALISIS CONFUSION KHUSUS ENTITAS LOCATION MODEL FINAL
    # ============================================================
    test_true = final_seed_res["test_true"]
    test_pred = final_seed_res["test_pred"]

    loc_correct = 0
    loc_pred_as_dest = 0
    loc_pred_as_o = 0
    loc_pred_other = 0
    total_loc_gold = 0

    for i in range(len(test_true)):
        t_seq = test_true[i]
        p_seq = test_pred[i]
        for t_tag, p_tag in zip(t_seq, p_seq):
            if t_tag.endswith("LOCATION"):
                total_loc_gold += 1
                if p_tag.endswith("LOCATION"):
                    loc_correct += 1
                elif p_tag.endswith("DESTINATION"):
                    loc_pred_as_dest += 1
                elif p_tag == "O":
                    loc_pred_as_o += 1
                else:
                    loc_pred_other += 1

    confusion_loc = {
        "total_gold_tokens": total_loc_gold,
        "correctly_predicted": loc_correct,
        "confused_as_destination": loc_pred_as_dest,
        "confused_as_O": loc_pred_as_o,
        "other_confusions": loc_pred_other
    }

    # ============================================================
    # 8. CETAK TABEL LAPORAN KE TERMINAL & FILE
    # ============================================================
    print("\n" + "=" * 105)
    print("📊 TABEL A: HASIL EVALUASI MULTI-SEED PADA TEST SET (583 BASELINE)")
    print("=" * 105)
    print(f"{'Seed':<6} | {'Epoch':<6} | {'Val Macro':<10} | {'LOC (P / R / F1)':<24} | {'PRICE (P / R / F1)':<24} | {'Test Macro F1':<14} | {'Status'}")
    print("-" * 105)

    for r in seed_results:
        s = r["seed"]
        ep = r["selected_epoch"]
        vm = r["val_macro_f1"] * 100
        rep = r["test_report"]
        
        loc_p = rep.get("LOCATION", {}).get("precision", 0) * 100
        loc_r = rep.get("LOCATION", {}).get("recall", 0) * 100
        loc_f = rep.get("LOCATION", {}).get("f1-score", 0) * 100
        loc_str = f"{loc_p:.1f} / {loc_r:.1f} / {loc_f:.1f}%"

        prc_p = rep.get("PRICE", {}).get("precision", 0) * 100
        prc_r = rep.get("PRICE", {}).get("recall", 0) * 100
        prc_f = rep.get("PRICE", {}).get("f1-score", 0) * 100
        prc_str = f"{prc_p:.1f} / {prc_r:.1f} / {prc_f:.1f}%"

        t_macro = r["test_macro_f1"] * 100
        stat_str = "⭐ FINAL MODEL" if s == final_seed else "Candidate"
        print(f"{s:<6} | {ep:<6} | {vm:8.2f}% | {loc_str:<24} | {prc_str:<24} | {t_macro:12.2f}% | {stat_str}")

    print("-" * 105)
    print(f"{'MEAN +- STD LINTAS 5 SEED':<24} | {'-':<10} | "
          f"{summary_stats['LOCATION']['precision_mean']*100:.1f} / {summary_stats['LOCATION']['recall_mean']*100:.1f} / {summary_stats['LOCATION']['f1_mean']*100:.1f} (±{summary_stats['LOCATION']['f1_std']*100:.1f}%) | "
          f"{summary_stats['PRICE']['precision_mean']*100:.1f} / {summary_stats['PRICE']['recall_mean']*100:.1f} / {summary_stats['PRICE']['f1_mean']*100:.1f} (±{summary_stats['PRICE']['f1_std']*100:.1f}%) | "
          f"{summary_stats['MACRO_AVG']['f1_mean']*100:.2f}% (±{summary_stats['MACRO_AVG']['f1_std']*100:.2f}%)")
    print("=" * 105)

    print("\n" + "=" * 90)
    print(f"📋 TABEL B: EVALUASI LENGKAP PER-ENTITAS MODEL FINAL (SEED {final_seed})")
    print("=" * 90)
    final_rep = final_seed_res["test_report"]
    print(f"{'Label Entitas':<15} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<8}")
    print("-" * 90)
    for ent in TARGET_ENTITIES:
        p = final_rep[ent]["precision"] * 100
        rc = final_rep[ent]["recall"] * 100
        f = final_rep[ent]["f1-score"] * 100
        sp = final_rep[ent]["support"]
        print(f"{ent:<15} | {p:10.2f}% | {rc:10.2f}% | {f:10.2f}% | {sp:<8}")
    print("-" * 90)
    print(f"{'Macro Average':<15} | {final_rep['macro avg']['precision']*100:10.2f}% | {final_rep['macro avg']['recall']*100:10.2f}% | {final_rep['macro avg']['f1-score']*100:10.2f}% | {final_rep['macro avg']['support']:<8}")
    print(f"{'Micro Average':<15} | {final_rep['micro avg']['precision']*100:10.2f}% | {final_rep['micro avg']['recall']*100:10.2f}% | {final_rep['micro avg']['f1-score']*100:10.2f}% | {final_rep['micro avg']['support']:<8}")
    print(f"{'Weighted Avg':<15} | {final_rep['weighted avg']['precision']*100:10.2f}% | {final_rep['weighted avg']['recall']*100:10.2f}% | {final_rep['weighted avg']['f1-score']*100:10.2f}% | {final_rep['weighted avg']['support']:<8}")
    print("=" * 90)

    print("\n🔍 ANALISIS CONFUSION TOKEN LOCATION (MODEL FINAL):")
    print(f"- Total Token LOCATION di Test Set    : {confusion_loc['total_gold_tokens']}")
    print(f"- Benar Diprediksi sebagai LOCATION   : {confusion_loc['correctly_predicted']} ({confusion_loc['correctly_predicted']/confusion_loc['total_gold_tokens']*100:.1f}%)")
    print(f"- Tertukar Menjadi DESTINATION        : {confusion_loc['confused_as_destination']} ({confusion_loc['confused_as_destination']/confusion_loc['total_gold_tokens']*100:.1f}%)")
    print(f"- Terlewat Menjadi Non-Entitas (O)    : {confusion_loc['confused_as_O']} ({confusion_loc['confused_as_O']/confusion_loc['total_gold_tokens']*100:.1f}%)")

    # Save to JSON summary
    json_summary = {
        "final_seed": final_seed,
        "final_seed_results": {
            "selected_epoch": final_seed_res["selected_epoch"],
            "val_macro_f1": final_seed_res["val_macro_f1"],
            "val_loc_f1": final_seed_res["val_loc_f1"],
            "test_macro_f1": final_seed_res["test_macro_f1"],
            "test_report": final_rep,
            "confusion_location": confusion_loc
        },
        "all_seeds": [
            {
                "seed": r["seed"],
                "selected_epoch": r["selected_epoch"],
                "val_macro_f1": r["val_macro_f1"],
                "val_loc_f1": r["val_loc_f1"],
                "test_macro_f1": r["test_macro_f1"],
                "test_report": r["test_report"]
            }
            for r in seed_results
        ],
        "cross_seed_statistics": summary_stats
    }

    with open("ner_multiseed_results.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
    with open(os.path.join(OUTPUT_BASE_DIR, "ner_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
    with open(os.path.join("output/reports", "ner_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)

    # Push Final Model ke Hugging Face Hub (Jika diminta)
    if args.push_to_hub:
        repo_id = args.hf_repo_id or os.environ.get("HF_REPO_ID", DEFAULT_HF_REPO_ID)
        token = args.hf_token or os.environ.get("HF_TOKEN", None)
        print(f"\n🌐 Mengunggah Final Model (Seed {final_seed}) ke Hugging Face Hub: {repo_id} ...")
        try:
            final_model = AutoModelForTokenClassification.from_pretrained(OUTPUT_FINAL_DIR)
            final_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_FINAL_DIR)
            final_model.push_to_hub(repo_id, token=token)
            final_tokenizer.push_to_hub(repo_id, token=token)
            print(f"✅ Final Model NER berhasil diunggah ke Hugging Face: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"⚠️ Gagal upload ke Hugging Face: {e}")

    print(f"\n💾 Seluruh artefak multi-seed tersimpan di: {OUTPUT_BASE_DIR} & {OUTPUT_FINAL_DIR}")


if __name__ == "__main__":
    main()
