"""
Evaluation Script for Intent Aggregate Metrics (Accuracy, Error Rate, ROC AUC OvR)
Read-only script. Outputs raw evaluation metrics to stdout and text file.
"""
import os
import csv
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
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
OUTPUT_REPORT_PATH = "output/reports/intent_aggregate_metrics.txt"
BATCH_SIZE = 16
MAX_LEN = 128
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


def main():
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: File test set tidak ditemukan: {TEST_PATH}")
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
        # Reconstruct mapping from TRAIN_CSV sorted unique labels (standard PyTorch/HF training pipeline convention)
        if not os.path.exists(TRAIN_CSV):
            print(f"ERROR: Config model generik dan {TRAIN_CSV} tidak ditemukan.")
            sys.exit(1)
        labels_set = set()
        with open(TRAIN_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels_set.add(row["label"].strip())
        label_list = sorted(labels_set)
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for l, i in label2id.items()}

    num_labels = len(id2label)

    # 3. Read CSV Robustly
    texts = []
    gold_labels_str = []
    
    with open(TEST_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lower().strip() for fn in reader.fieldnames]
        
        text_col = None
        for candidate in ["text", "kalimat", "sentence"]:
            if candidate in fieldnames:
                text_col = [orig for orig in reader.fieldnames if orig.lower().strip() == candidate][0]
                break
                
        label_col = None
        for candidate in ["label", "intent", "kelas"]:
            if candidate in fieldnames:
                label_col = [orig for orig in reader.fieldnames if orig.lower().strip() == candidate][0]
                break
                
        if not text_col or not label_col:
            print(f"ERROR: Gagal memetakan kolom. Kolom tersedia: {reader.fieldnames}")
            sys.exit(1)
            
        for row in reader:
            texts.append(row[text_col].strip())
            gold_labels_str.append(row[label_col].strip())

    total_samples = len(texts)

    # 4. Check label validity
    unmapped_labels = set(gold_labels_str) - set(label2id.keys())
    if unmapped_labels:
        print(f"ERROR: Label pada CSV tidak dikenali model: {unmapped_labels}")
        sys.exit(1)

    gold_ids = np.array([label2id[l] for l in gold_labels_str])

    # 5. Batch Inference (raw softmax)
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

    # 6. Metrics Calculation
    acc = accuracy_score(gold_ids, y_pred)
    error_rate = 1.0 - acc
    wrong_count = int(np.sum(y_pred != gold_ids))

    # ROC AUC
    roc_auc_macro = float('nan')
    roc_auc_weighted = float('nan')

    try:
        y_true_onehot = np.zeros((total_samples, num_labels))
        for i, gid in enumerate(gold_ids):
            y_true_onehot[i, gid] = 1.0

        roc_auc_macro = roc_auc_score(
            y_true_onehot,
            y_proba,
            multi_class="ovr",
            average="macro",
            labels=list(range(num_labels))
        )
        roc_auc_weighted = roc_auc_score(
            y_true_onehot,
            y_proba,
            multi_class="ovr",
            average="weighted",
            labels=list(range(num_labels))
        )
    except Exception as e:
        print(f"WARNING: Gagal menghitung ROC AUC: {e}")
        present_classes = set(gold_ids)
        missing_classes = set(range(num_labels)) - present_classes
        print(f"Kelas yang tidak muncul pada test set: {[id2label[c] for c in missing_classes]}")

    # 7. Format Output
    output_lines = [
        "================================================================",
        "EVAL AGREGAT INTENT — MODEL TERBARU",
        "================================================================",
        f"Model source     : {MODEL_SOURCE}",
        f"Jumlah kelas     : {num_labels}",
        f"Total kalimat uji: {total_samples}",
        "----------------------------------------------------------------",
        f"Accuracy         : {acc:.4f}  ({acc*100:.2f}%)",
        f"Error Rate       : {error_rate:.4f}  ({error_rate*100:.2f}%)",
        f"Kalimat salah    : {wrong_count} / {total_samples}",
        f"ROC AUC (OvR macro)    : {roc_auc_macro:.4f}",
        f"ROC AUC (OvR weighted) : {roc_auc_weighted:.4f}",
        "================================================================"
    ]

    report_content = "\n".join(output_lines)

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content + "\n")

    print(report_content)


if __name__ == "__main__":
    main()
