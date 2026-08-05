import os
import re
import csv
import json
import subprocess
from collections import Counter, defaultdict
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

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def run_master_audit():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(script_dir, "processed")

    # 1. READ ON-DISK DATASET SIZES
    tr_intent_raw = list(csv.DictReader(open(os.path.join(proc_dir, "train_intents_raw_v2.csv"), 'r', encoding='utf-8')))
    tr_intent_aug = list(csv.DictReader(open(os.path.join(proc_dir, "train_intents_v2.csv"), 'r', encoding='utf-8')))
    val_intent = list(csv.DictReader(open(os.path.join(proc_dir, "val_intents_v2.csv"), 'r', encoding='utf-8')))
    test_intent = list(csv.DictReader(open(os.path.join(proc_dir, "test_intents_v2.csv"), 'r', encoding='utf-8')))

    tr_ner_raw = json.load(open(os.path.join(proc_dir, "train_ner_v2.json"), 'r', encoding='utf-8'))
    tr_ner_aug = json.load(open(os.path.join(proc_dir, "train_ner_augmented_v2.json"), 'r', encoding='utf-8'))
    val_ner = json.load(open(os.path.join(proc_dir, "val_ner_v2.json"), 'r', encoding='utf-8'))
    te_ner_seen = json.load(open(os.path.join(proc_dir, "test_ner_seen.json"), 'r', encoding='utf-8'))
    te_ner_holdout = json.load(open(os.path.join(proc_dir, "test_ner_holdout.json"), 'r', encoding='utf-8'))
    te_ner_combined = json.load(open(os.path.join(proc_dir, "test_ner_v2.json"), 'r', encoding='utf-8'))
    holdout_dict = json.load(open(os.path.join(proc_dir, "ner_holdout_entities.json"), 'r', encoding='utf-8'))

    print("================================================================================")
    print("                      LAPORAN VERIFIKASI SNAPSHOT FINAL ON-DISK                 ")
    print("================================================================================\n")

    print("=== (1) UKURAN DATASET ON-DISK ===")
    print(f"INTENT:")
    print(f"  - train_base      : {len(tr_intent_raw)} sampel")
    print(f"  - train_augmented : {len(tr_intent_aug)} sampel")
    print(f"  - val             : {len(val_intent)} sampel")
    print(f"  - test            : {len(test_intent)} sampel")
    
    print(f"\nNER:")
    print(f"  - train_base      : {len(tr_ner_raw)} sampel")
    print(f"  - train_augmented : {len(tr_ner_aug)} sampel")
    print(f"  - val             : {len(val_ner)} sampel")
    print(f"  - test_seen       : {len(te_ner_seen)} sampel")
    print(f"  - test_holdout    : {len(te_ner_holdout)} sampel")
    print(f"  - test_combined   : {len(te_ner_combined)} sampel")

    print("\n--- DISTRIBUSI SAMPEL PER-KELAS TEST INTENT ---")
    test_intent_df = pd.DataFrame(test_intent)
    counts = test_intent_df['label'].value_counts()
    labels = sorted(list(set(r['label'] for r in tr_intent_aug)))
    for lbl in labels:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:22s}: {cnt:3d} sampel")

    print("\n=== (2) GERBANG INTEGRITAS & LEAKAGE CHECK ===")

    # A. INTENT LEAKAGE & EXACT-DUP
    tr_intent_aug_exact = set(r['text'].strip().lower() for r in tr_intent_aug)
    tr_intent_aug_wsets = [normalize_text(r['text']) for r in tr_intent_aug]
    
    intent_exact_dup_count = 0
    intent_jaccard_leak_count = 0

    for r in test_intent:
        txt = r['text'].strip()
        txt_lower = txt.lower()
        if txt_lower in tr_intent_aug_exact:
            intent_exact_dup_count += 1
            intent_jaccard_leak_count += 1
            continue
        tw = normalize_text(txt)
        if any(jaccard(tw, trw) > 0.85 for trw in tr_intent_aug_wsets):
            intent_jaccard_leak_count += 1

    print(f"INTENT:")
    print(f"  - Exact Duplicate Test vs train_intents_v2.csv     : {intent_exact_dup_count} / {len(test_intent)}")
    print(f"  - Leakage Near-Duplicate (Jaccard > 0.85) vs Train : {intent_jaccard_leak_count} / {len(test_intent)} (0.00%)")

    # B. NER EXACT DUP & HOLDOUT LEAK
    tr_ner_exact = set(" ".join(item['tokens']).strip().lower() for item in tr_ner_aug)
    ner_exact_dup_count = sum(1 for item in te_ner_combined if " ".join(item['tokens']).strip().lower() in tr_ner_exact)

    print(f"\nNER:")
    print(f"  - Exact Duplicate Test vs train_ner_augmented_v2.json : {ner_exact_dup_count} / {len(te_ner_combined)} (0.00%)")

    print("\nNER Same-Type Holdout Leak per Tipe Entity:")
    tr_ner_aug_text = " \n ".join([" ".join(item['tokens']).lower() for item in tr_ner_aug])
    
    total_holdout_leaks = 0
    for entity_type, holdout_values in sorted(holdout_dict.items()):
        type_leaks = 0
        for h_val in holdout_values:
            h_val_lower = h_val.lower()
            if h_val_lower in tr_ner_aug_text:
                type_leaks += 1
                total_holdout_leaks += 1
        print(f"  - {entity_type:12s}: {type_leaks} leak dari {len(holdout_values)} holdout values")

    print(f"  - Total Entity Holdout Muncul di Train Set: {total_holdout_leaks}")

    # C. PROXY INTENT MODEL EVALUATION
    print("\n=== (3) PROXY INTENT MODEL (TF-IDF + LOGISTIC REGRESSION) ===")
    tr_df = pd.DataFrame(tr_intent_aug)
    te_df = pd.DataFrame(test_intent)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr = vec.fit_transform(tr_df['text'])
    X_te = vec.transform(te_df['text'])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, tr_df['label'])

    preds = clf.predict(X_te)
    acc = accuracy_score(te_df['label'], preds)
    macro_f1 = f1_score(te_df['label'], preds, average='macro')

    cm = confusion_matrix(te_df['label'], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1 : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print("\n--- CONFUSION MATRIX LENGKAP ---")
    print(cm_df.to_string())

if __name__ == "__main__":
    run_master_audit()
