"""
=============================================================
PELESIR PALEMBANG — RETRAINING SCRIPT (Untuk Google Colab)
=============================================================
Jalankan file ini di Google Colab dengan GPU T4.

CARA PAKAI:
1. Upload folder `ml/data/processed/` ke Colab (atau clone repo)
2. Jalankan script ini: `python retrain_colab.py`
3. Setelah selesai, download folder `output/` yang berisi model + artefak
4. Push ke Hugging Face Hub

FITUR BARU:
- Base model: XLM-RoBERTa (menggantikan mBERT)
- Label Smoothing (0.1) untuk anti-overconfidence
- Weight Decay (0.01) untuk anti-overfitting
- Temperature Scaling (kalibrasi confidence)
- Early Stopping + Gradient Clipping
- Evaluasi lengkap (confusion matrix, classification report, kurva loss)
=============================================================
"""

import os
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intent Config
INTENT_MODEL_NAME = "xlm-roberta-base"
INTENT_MAX_LEN = 128
INTENT_BATCH_SIZE = 16
INTENT_EPOCHS = 20
INTENT_LR = 2e-5
INTENT_WEIGHT_DECAY = 0.01
INTENT_LABEL_SMOOTHING = 0.1
INTENT_PATIENCE = 5
INTENT_WARMUP_RATIO = 0.1

# NER Config
NER_MODEL_NAME = "xlm-roberta-base"
NER_MAX_LEN = 128
NER_BATCH_SIZE = 16
NER_EPOCHS = 25
NER_LR = 2e-5
NER_WEIGHT_DECAY = 0.01
NER_PATIENCE = 3
NER_WARMUP_RATIO = 0.1

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

# Paths (sesuaikan dengan lokasi data Anda)
DATA_DIR = "ml/data/processed"
OUTPUT_DIR = "output"

# ============================================================
# SEED FIXING
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# DATASET CLASSES
# ============================================================
class IntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len, label2id):
        self.texts = []
        self.labels = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row['text'])
                self.labels.append(label2id[row['label']])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


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

        tokenized = self.tokenizer(
            tokens, is_split_into_words=True,
            padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )

        word_ids = tokenized.word_ids()
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(self.tag2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx

        return {
            'input_ids': tokenized['input_ids'].flatten(),
            'attention_mask': tokenized['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# ============================================================
# PART 1: TRAIN INTENT MODEL
# ============================================================
def train_intent():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    import seaborn as sns

    print("=" * 60)
    print("PART 1: TRAINING INTENT MODEL (XLM-RoBERTa)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data & build label mapping
    train_path = os.path.join(DATA_DIR, "train_intents_v2.csv")
    val_path = os.path.join(DATA_DIR, "val_intents_v2.csv")
    test_path = os.path.join(DATA_DIR, "test_intents_v2.csv")

    labels_set = set()
    with open(train_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            labels_set.add(row['label'])
    
    label_list = sorted(labels_set)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label_list)
    print(f"Labels ({num_labels}): {label_list}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_NAME, num_labels=num_labels
    ).to(DEVICE)

    # Datasets
    train_ds = IntentDataset(train_path, tokenizer, INTENT_MAX_LEN, label2id)
    val_ds = IntentDataset(val_path, tokenizer, INTENT_MAX_LEN, label2id)
    test_ds = IntentDataset(test_path, tokenizer, INTENT_MAX_LEN, label2id)

    train_loader = DataLoader(train_ds, batch_size=INTENT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=INTENT_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=INTENT_BATCH_SIZE, shuffle=False)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=INTENT_LR, weight_decay=INTENT_WEIGHT_DECAY)
    total_steps = len(train_loader) * INTENT_EPOCHS
    warmup_steps = int(total_steps * INTENT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=INTENT_LABEL_SMOOTHING)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    intent_save_dir = os.path.join(OUTPUT_DIR, "intent_model")
    os.makedirs(intent_save_dir, exist_ok=True)

    print(f"\nStarting training: {INTENT_EPOCHS} epochs, LR={INTENT_LR}, Label Smoothing={INTENT_LABEL_SMOOTHING}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    for epoch in range(INTENT_EPOCHS):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss_total, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss_total += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{INTENT_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(intent_save_dir)
            tokenizer.save_pretrained(intent_save_dir)
            print(f"  --> Best model saved! (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= INTENT_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}!")
                break

    # Save label mappings
    with open(os.path.join(intent_save_dir, 'id2label.json'), 'w') as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(intent_save_dir, 'label2id.json'), 'w') as f:
        json.dump(label2id, f, indent=2)

    # ---- EVALUATION ON TEST SET ----
    print("\n--- Evaluasi pada Test Set ---")
    model = AutoModelForSequenceClassification.from_pretrained(intent_save_dir).to(DEVICE)
    model.eval()

    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

    print(f"\nAccuracy:    {acc:.4f}")
    print(f"F1 (Macro):  {f1_macro:.4f}")
    print(f"F1 (Weight): {f1_weighted:.4f}")

    target_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(true_labels, pred_labels, target_names=target_names)
    print(f"\n{report}")

    # Save report
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    with open(os.path.join(reports_dir, 'intent_classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Macro: {f1_macro:.4f}\nF1 Weighted: {f1_weighted:.4f}\n\n{report}")

    # Plot Loss Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_title('Intent: Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
    ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Val Acc')
    ax2.set_title('Intent: Training vs Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'intent_training_curves.png'), dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Intent Confusion Matrix (XLM-RoBERTa)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'intent_confusion_matrix.png'), dpi=300)
    plt.close()

    print(f"\nPlots saved to {reports_dir}/")
    return intent_save_dir, id2label, label2id, acc, f1_macro


# ============================================================
# PART 2: TRAIN NER MODEL
# ============================================================
def train_ner():
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

    print("\n" + "=" * 60)
    print("PART 2: TRAINING NER MODEL (XLM-RoBERTa)")
    print("=" * 60)

    train_path = os.path.join(DATA_DIR, "train_ner_augmented_v2.json")
    val_path = os.path.join(DATA_DIR, "val_ner_v2.json")
    test_path = os.path.join(DATA_DIR, "test_ner_v2.json")

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_NAME, num_labels=len(NER_TAGS), ignore_mismatched_sizes=True, classifier_dropout=0.3
    ).to(DEVICE)

    train_ds = NERDataset(train_path, tokenizer, NER_MAX_LEN, NER_TAG2ID)
    val_ds = NERDataset(val_path, tokenizer, NER_MAX_LEN, NER_TAG2ID)
    test_ds = NERDataset(test_path, tokenizer, NER_MAX_LEN, NER_TAG2ID)

    train_loader = DataLoader(train_ds, batch_size=NER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=NER_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=NER_BATCH_SIZE, shuffle=False)

    # Differential Learning Rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {"params": [p for n, p in model.named_parameters()
                     if "classifier" not in n and not any(nd in n for nd in no_decay)],
         "lr": NER_LR, "weight_decay": NER_WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                     if "classifier" not in n and any(nd in n for nd in no_decay)],
         "lr": NER_LR, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n],
         "lr": NER_LR * 5, "weight_decay": NER_WEIGHT_DECAY},
    ]
    optimizer = AdamW(optimizer_params)

    total_steps = len(train_loader) * NER_EPOCHS
    warmup_steps = int(total_steps * NER_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    ner_save_dir = os.path.join(OUTPUT_DIR, "ner_model")
    os.makedirs(ner_save_dir, exist_ok=True)

    print(f"Starting NER training: {NER_EPOCHS} epochs")

    for epoch in range(NER_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss_total += outputs.loss.item()

        val_loss = val_loss_total / len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{NER_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(ner_save_dir)
            tokenizer.save_pretrained(ner_save_dir)
            print(f"  --> Best NER model saved!")
        else:
            patience_counter += 1
            if patience_counter >= NER_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}!")
                break

    # Save tag mappings
    with open(os.path.join(ner_save_dir, 'id2tag.json'), 'w') as f:
        json.dump(NER_ID2TAG, f, indent=2)
    with open(os.path.join(ner_save_dir, 'tag2id.json'), 'w') as f:
        json.dump(NER_TAG2ID, f, indent=2)

    # ---- NER EVALUATION ----
    print("\n--- NER Evaluasi pada Test Set ---")
    model = AutoModelForTokenClassification.from_pretrained(ner_save_dir).to(DEVICE)
    model.eval()

    all_true_tags, all_pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
            labels_np = labels.numpy()

            for i in range(len(preds)):
                true_seq, pred_seq = [], []
                for j in range(len(preds[i])):
                    if labels_np[i][j] != -100:
                        true_seq.append(NER_ID2TAG[labels_np[i][j]])
                        pred_seq.append(NER_ID2TAG[preds[i][j]])
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    ner_report = seq_report(all_true_tags, all_pred_tags)
    ner_f1 = seq_f1(all_true_tags, all_pred_tags)
    print(f"\nEntity-level F1: {ner_f1:.4f}")
    print(ner_report)

    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    with open(os.path.join(reports_dir, 'ner_classification_report.txt'), 'w') as f:
        f.write(f"Entity-level F1: {ner_f1:.4f}\n\n{ner_report}")

    # Plot NER Loss
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('NER: Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'ner_training_curves.png'), dpi=300)
    plt.close()

    return ner_save_dir, ner_f1


# ============================================================
# PART 3: TEMPERATURE SCALING (KALIBRASI CONFIDENCE)
# ============================================================
def calibrate_temperature(intent_save_dir):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("\n" + "=" * 60)
    print("PART 3: TEMPERATURE SCALING (KALIBRASI CONFIDENCE)")
    print("=" * 60)

    # Load best intent model
    tokenizer = AutoTokenizer.from_pretrained(intent_save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(intent_save_dir).to(DEVICE)
    model.eval()

    with open(os.path.join(intent_save_dir, 'label2id.json'), 'r') as f:
        label2id = json.load(f)

    val_path = os.path.join(DATA_DIR, "val_intents_v2.csv")
    val_ds = IntentDataset(val_path, tokenizer, INTENT_MAX_LEN, label2id)
    val_loader = DataLoader(val_ds, batch_size=INTENT_BATCH_SIZE, shuffle=False)

    # Collect logits and labels
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Temperature parameter (optimizable)
    temperature = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    # NLL before calibration
    nll_before = F.cross_entropy(logits, labels).item()

    def eval_closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_closure)

    T_optimal = temperature.item()
    nll_after = F.cross_entropy(logits / temperature, labels).item()

    print(f"Temperature optimal: T = {T_optimal:.4f}")
    print(f"NLL sebelum kalibrasi: {nll_before:.4f}")
    print(f"NLL sesudah kalibrasi: {nll_after:.4f}")

    # Expected Calibration Error (ECE)
    def compute_ece(logits_input, labels_input, n_bins=10):
        probs = F.softmax(logits_input, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels_input).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean().item()
                bin_conf = confidences[mask].mean().item()
                ece += mask.sum().item() * abs(bin_acc - bin_conf)
        return ece / len(labels_input)

    ece_before = compute_ece(logits, labels)
    ece_after = compute_ece(logits / T_optimal, labels)

    print(f"ECE sebelum kalibrasi: {ece_before:.4f}")
    print(f"ECE sesudah kalibrasi: {ece_after:.4f}")

    # Save calibration
    calibration = {"temperature": T_optimal}
    calib_path = os.path.join(intent_save_dir, 'calibration.json')
    with open(calib_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\nCalibration saved: {calib_path}")

    # Reliability Diagram
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    n_bins = 10

    for ax, title, logits_input in [(ax1, "Sebelum Kalibrasi", logits),
                                      (ax2, f"Sesudah Kalibrasi (T={T_optimal:.2f})", logits / T_optimal)]:
        probs = F.softmax(logits_input, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean().item())
                bin_confs.append(confidences[mask].mean().item())
            else:
                bin_accs.append(0)
                bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)

        ax.bar(range(n_bins), bin_accs, alpha=0.5, label='Accuracy', width=0.4, align='center')
        ax.bar([x + 0.4 for x in range(n_bins)], bin_confs, alpha=0.5, label='Confidence', width=0.4, align='center')
        ax.plot([0, n_bins], [0, 1], 'k--', label='Perfect calibration')
        ax.set_title(title)
        ax.set_xlabel('Bin')
        ax.set_ylabel('Proportion')
        ax.legend()
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'reliability_diagram.png'), dpi=300)
    plt.close()

    return T_optimal, ece_before, ece_after


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🚀 Pelesir Palembang — NLP Model Retraining")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print()

    # Part 1: Intent
    intent_dir, id2label, label2id, intent_acc, intent_f1 = train_intent()

    # Part 2: NER
    ner_dir, ner_f1 = train_ner()

    # Part 3: Calibration
    T, ece_before, ece_after = calibrate_temperature(intent_dir)

    # Summary
    print("\n" + "=" * 60)
    print("📊 RINGKASAN HASIL RETRAINING")
    print("=" * 60)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                   METRIK BARU vs LAMA                    ║
╠══════════════════╦═══════════╦════════════╦══════════════╣
║ Metrik           ║ LAMA      ║ BARU       ║ Perubahan    ║
╠══════════════════╬═══════════╬════════════╬══════════════╣
║ Intent Accuracy  ║ ~0.9200   ║ {intent_acc:.4f}     ║ {"↑" if intent_acc > 0.92 else "↓"} {abs(intent_acc - 0.92)*100:.1f}%       ║
║ Intent F1 Macro  ║ ~0.9100   ║ {intent_f1:.4f}     ║ {"↑" if intent_f1 > 0.91 else "↓"} {abs(intent_f1 - 0.91)*100:.1f}%       ║
║ NER Entity F1    ║ ~0.8800   ║ {ner_f1:.4f}     ║ {"↑" if ner_f1 > 0.88 else "↓"} {abs(ner_f1 - 0.88)*100:.1f}%       ║
║ ECE (Kalibrasi)  ║ N/A       ║ {ece_after:.4f}     ║ ↓ {(ece_before-ece_after)*100:.1f}% from {ece_before:.4f}  ║
║ Temperature (T)  ║ 1.00      ║ {T:.4f}     ║              ║
╚══════════════════╩═══════════╩════════════╩══════════════╝
    """)

    print(f"\n📁 Artefak tersimpan di folder: {OUTPUT_DIR}/")
    print(f"   - {intent_dir}/ (model intent + calibration.json)")
    print(f"   - {ner_dir}/ (model NER)")
    print(f"   - {OUTPUT_DIR}/reports/ (grafik & laporan)")
    print(f"\n🔜 Langkah selanjutnya:")
    print(f"   1. Download folder '{OUTPUT_DIR}/'")
    print(f"   2. Push intent model ke HF Hub: ZeroCG/pelesir-intent")
    print(f"   3. Push NER model ke HF Hub: ZeroCG/pelesir-ner")
    print(f"   4. Update engine.py untuk Temperature Scaling")


if __name__ == "__main__":
    main()
