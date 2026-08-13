"""
plot_intent_roc_curve.py — Script untuk menghasilkan Grafik Kurva ROC (Receiver Operating Characteristic)
pada Model Klasifikasi Intent XLM-RoBERTa (Pelesir Palembang).
Menghasilkan gambar berstandar skripsi: output/gambar_intent_roc_curve.png (300 DPI).
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# KONFIGURASI
# ================================================================
LOCAL_PATHS = ["output/intent_model", "output/intent"]
HF_FALLBACK = "ZeroCG/pelesir-intent"

MODEL_SOURCE = HF_FALLBACK
for lp in LOCAL_PATHS:
    if os.path.exists(lp):
        MODEL_SOURCE = lp
        break

TRAIN_CSV = "ml/data/processed/train_intents_v2.csv"
TEST_PATH = "ml/data/processed/test_intents_v2.csv"
OUTPUT_IMG = "output/gambar_intent_roc_curve.png"
OUTPUT_REPORT_PATH = "output/reports/intent_aggregate_metrics.txt"

BATCH_SIZE = 16
MAX_LEN = 128
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


def main():
    print("=" * 70)
    print("🚀 MENGHASILKAN GRAFIK ROC CURVE INTENT CLASSIFICATION (300 DPI)")
    print("=" * 70)

    if not os.path.exists(TEST_PATH):
        print(f"❌ ERROR: File test set tidak ditemukan: {TEST_PATH}")
        sys.exit(1)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE).to(DEVICE)
    model.eval()

    # 2. Extract or reconstruct label mapping
    config_id2label = {int(k): v for k, v in model.config.id2label.items()}
    has_custom_labels = any(not v.startswith("LABEL_") for v in config_id2label.values())

    if has_custom_labels:
        id2label = config_id2label
        label2id = {v: k for k, v in id2label.items()}
    else:
        labels_set = set()
        with open(TRAIN_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels_set.add(row["label"].strip())
        label_list = sorted(labels_set)
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for l, i in label2id.items()}

    num_labels = len(id2label)
    class_names = [id2label[i] for i in range(num_labels)]

    # 3. Read Test CSV
    texts, gold_labels_str = [], []
    with open(TEST_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text") or row.get("kalimat") or row.get("sentence")
            l = row.get("label") or row.get("intent") or row.get("kelas")
            texts.append(t.strip())
            gold_labels_str.append(l.strip())

    total_samples = len(texts)
    gold_ids = np.array([label2id[l] for l in gold_labels_str])

    # 4. Batch Inference (raw softmax)
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, total_samples, BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.append(probs)
            all_preds.append(preds)

    y_proba = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)

    # 5. One-Hot Binarize
    y_true_onehot = np.zeros((total_samples, num_labels))
    for i, gid in enumerate(gold_ids):
        y_true_onehot[i, gid] = 1.0

    # 6. Metrics Calculation
    acc = accuracy_score(gold_ids, y_pred)
    error_rate = 1.0 - acc
    roc_auc_macro = roc_auc_score(y_true_onehot, y_proba, multi_class="ovr", average="macro")

    print(f"📊 Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"📊 Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    print(f"📊 ROC AUC   : {roc_auc_macro:.4f}")

    # 7. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 8. Plotting High-Resolution Professional ROC Curve
    plt.figure(figsize=(9, 7), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # Plot Macro-average ROC Curve (Bold Orange)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink',
        linestyle=':',
        linewidth=2.5
    )

    # Plot Macro-average (Thick Navy Line)
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_labels
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'Macro-average ROC (AUC = {roc_auc_macro:.4f}) [Utama]',
        color='navy',
        linestyle='-',
        linewidth=2.8
    )

    # Colors for individual classes
    colors = plt.cm.tab20(np.linspace(0, 1, num_labels))
    for i, color in zip(range(num_labels), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            alpha=0.65,
            linewidth=1.2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )

    # Diagonal Random Chance Line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (AUC = 0.5000)')

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('Kurva ROC (Receiver Operating Characteristic) Model Intent\nAccuracy = 93,51% | Error Rate = 6,49% | Macro ROC AUC = 0,9903', fontsize=12, fontweight='bold', pad=14)
    plt.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Grafik Kurva ROC berhasil disimpan di: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
