import os
import re
import csv
import json
from collections import defaultdict, Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ID_DIALECT_REVERSE = {
    r'\bapo\b': 'apa', r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak',
    r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana', r'\bnian\b': 'sangat',
    r'\bkau\b': 'kamu', r'\blemak\b': 'enak', r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga',
    r'\bberapo\b': 'berapa', r'\bngapo\b': 'kenapa', r'\bbae\b': 'saja', r'\byo\b': 'ya',
    r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
}

def normalize_text(text):
    t = str(text).lower()
    for pat, repl in ID_DIALECT_REVERSE.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return set(t.split())

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def run_validation():
    print("=== (1) UKURAN DATASET FINAL & DISTRIBUSI KELAS TEST INTENT ===")
    
    tr_intent_aug = list(csv.DictReader(open("processed/train_intents_v2.csv", 'r', encoding='utf-8')))
    tr_intent_raw = list(csv.DictReader(open("processed/train_intents_raw_v2.csv", 'r', encoding='utf-8')))
    val_intent = list(csv.DictReader(open("processed/val_intents_v2.csv", 'r', encoding='utf-8')))
    test_intent = list(csv.DictReader(open("processed/test_intents_v2.csv", 'r', encoding='utf-8')))

    print(f"Intent Train (Augmented) : {len(tr_intent_aug)} sampel")
    print(f"Intent Train (Raw Base)  : {len(tr_intent_raw)} sampel")
    print(f"Intent Val               : {len(val_intent)} sampel")
    print(f"Intent Test              : {len(test_intent)} sampel")

    print("\n--- DISTRIBUSI SAMPEL PER KELAS DI TEST_INTENTS_V2.CSV ---")
    test_df = pd.DataFrame(test_intent)
    counts = test_df['label'].value_counts()
    labels = sorted(list(set(r['label'] for r in tr_intent_aug)))
    for lbl in labels:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:22s}: {cnt:3d} sampel")

    print("\n=== (2) RECONFIRM POST-DEDUP TEST VS AUGMENTED TRAIN (train_intents_v2.csv) ===")
    train_aug_exact = set(r['text'].strip().lower() for r in tr_intent_aug)
    train_aug_wordsets = [normalize_text(r['text']) for r in tr_intent_aug]
    
    dropped_test_count = 0
    for r in test_intent:
        txt = r['text'].strip()
        if txt.lower() in train_aug_exact:
            dropped_test_count += 1
            continue
        t_w = normalize_text(txt)
        if any(jaccard_similarity(t_w, tr_w) > 0.85 for tr_w in train_aug_wordsets):
            dropped_test_count += 1
            
    print(f"Jumlah Sampel Test Dibuang di Pasca-Dedup vs Augmented Train: {dropped_test_count} sampel (Target ~0)")

    print("\n=== (3) PROXY MODEL (TF-IDF + LOGISTIC REGRESSION) EVALUATION ===")
    train_df = pd.DataFrame(tr_intent_aug)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr = vec.fit_transform(train_df['text'])
    X_te = vec.transform(test_df['text'])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, train_df['label'])

    preds = clf.predict(X_te)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')

    cm = confusion_matrix(test_df['label'], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print("\n--- CONFUSION MATRIX LENGKAP ---")
    print(cm_df.to_string())

if __name__ == "__main__":
    run_validation()
