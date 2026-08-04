"""
Stage 0 & Stage 1 Audit Script — Pelesir Palembang NLP Dataset
Melakukan Audit Statis (Stage 0) dan Evaluasi Model Proxy CPU (Stage 1)
"""
import os
import re
import csv
import json
from collections import Counter, defaultdict

# -------------------------------------------------------------------
# STAGE 0: AUDIT STATIS DATASET
# -------------------------------------------------------------------

ID_DIALECT_REVERSE = {
    r'\bapo\b': 'apa',
    r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak', r'\bndak\b': 'tidak',
    r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana',
    r'\bnian\b': 'sangat', r'\bbanget\b': 'sangat',
    r'\bkau\b': 'kamu',
    r'\bambo\b': 'aku',
    r'\blemak\b': 'enak',
    r'\bpacak\b': 'bisa',
    r'\bhargo\b': 'harga',
    r'\bberapo\b': 'berapa', r'\bbrapa\b': 'berapa',
    r'\bngapo\b': 'kenapa', r'\bknp\b': 'kenapa',
    r'\bkalo\b': 'kalau', r'\bkalu\b': 'kalau', r'\bmen\b': 'kalau',
    r'\bbae\b': 'saja',
    r'\byo\b': 'ya',
    r'\bkatek\b': 'tidak ada',
    r'\bpegi\b': 'pergi',
    r'\bjingok\b': 'lihat'
}

def normalize_text(text):
    t = text.lower()
    for pat, repl in ID_DIALECT_REVERSE.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return set(t.split())

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def run_stage_0():
    print("=" * 70)
    print("STAGE 0: AUDIT STATIS LEVEL DATASET")
    print("=" * 70)
    
    train_path = "ml/data/processed/train_intents_v2.csv"
    val_path   = "ml/data/processed/val_intents_v2.csv"
    test_path  = "ml/data/processed/test_intents_v2.csv"
    
    if not os.path.exists(train_path):
        print(f"❌ File {train_path} tidak ditemukan!")
        return
        
    train_rows, val_rows, test_rows = [], [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        train_rows = list(csv.DictReader(f))
    with open(val_path, 'r', encoding='utf-8') as f:
        val_rows = list(csv.DictReader(f))
    with open(test_path, 'r', encoding='utf-8') as f:
        test_rows = list(csv.DictReader(f))
        
    print(f"📊 Jumlah Sampel: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}")
    
    # 1. Near-Duplicate / Dialect Jaccard Leakage Check
    print("\n🔍 [1/4] Pengecekan Kebocoran Halus (Jaccard Similarity > 0.85 per Label)...")
    train_sets_by_label = defaultdict(list)
    for r in train_rows:
        train_sets_by_label[r['label']].append(normalize_text(r['text']))
        
    near_leaks = 0
    total_test = len(test_rows)
    for r in test_rows:
        test_set = normalize_text(r['text'])
        label = r['label']
        for tr_set in train_sets_by_label[label]:
            sim = jaccard_similarity(test_set, tr_set)
            if sim > 0.85 and sim < 1.0: # high similarity non-exact
                near_leaks += 1
                break
                
    pct_leak = (near_leaks / max(1, total_test)) * 100
    print(f"   Hasil: {near_leaks}/{total_test} ({pct_leak:.2f}%) kalimat test memiliki kemiripan > 85% dengan train set.")
    if pct_leak < 5.0:
        print("   ✅ BEBAS KEBOCORAN HALUS (Leakage < 5%)")
    else:
        print("   ⚠️  PERINGATAN: Ada potensi leakage halus tinggi antar split.")

    # 2. Rasio Pattern Unik per Intent Label
    print("\n🔍 [2/4] Rasio Pattern Unik per Label (Signature Pattern Diversity)...")
    ALL_DEST = ["bkb", "ampera", "kemaro", "punti kayu", "jakabaring", "siguntang", "balaputra dewa", "museum", "masjid agung"]
    def get_sig(t):
        s = t.lower()
        for d in ALL_DEST:
            s = s.replace(d, "[DEST]")
        s = re.sub(r'\b\d+\b', '[NUM]', s)
        return s
        
    sigs_per_label = defaultdict(set)
    count_per_label = defaultdict(int)
    for r in train_rows:
        lbl = r['label']
        count_per_label[lbl] += 1
        sigs_per_label[lbl].add(get_sig(r['text']))
        
    for lbl in sorted(count_per_label.keys()):
        u = len(sigs_per_label[lbl])
        n = count_per_label[lbl]
        ratio = (u / max(1, n)) * 100
        print(f"   {lbl:22s}: {u:4d} pattern unik / {n:4d} sampel ({ratio:.1f}%)")

    # 3. Korelasi Entitas "Balaputra Dewa" & Entitas Utama
    print("\n🔍 [3/4] Evaluasi Distribusi Entitas 'Balaputra Dewa' Across Intents...")
    balaputra_dist = Counter()
    for r in train_rows:
        if "balaputra dewa" in r['text'].lower():
            balaputra_dist[r['label']] += 1
    for lbl, cnt in balaputra_dist.most_common():
        pct = (cnt / sum(balaputra_dist.values())) * 100
        print(f"   Balaputra Dewa -> {lbl:22s}: {cnt:3d} kali ({pct:.1f}%)")

    # 4. Keseimbangan Kelas
    print("\n🔍 [4/4] Keseimbangan Kelas Train Set...")
    for lbl, cnt in sorted(count_per_label.items()):
        print(f"   {lbl:22s}: {cnt:4d} sampel")

# -------------------------------------------------------------------
# STAGE 1: EVALUASI PROXY MODEL (TF-IDF + LOGISTIC REGRESSION)
# -------------------------------------------------------------------

def run_stage_1():
    print("\n" + "=" * 70)
    print("STAGE 1: EVALUASI PROXY MODEL (CPU BASELINE)")
    print("=" * 70)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    import pandas as pd
    
    train_df = pd.read_csv("ml/data/processed/train_intents_v2.csv")
    test_df  = pd.read_csv("ml/data/processed/test_intents_v2.csv")
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df['text'])
    X_test  = vec.transform(test_df['text'])
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_df['label'])
    
    preds = clf.predict(X_test)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')
    
    print(f"🤖 Logistic Regression Baseline (TF-IDF n-gram 1-2):")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    print("\n💡 INTERPRETASI STAGE 1:")
    if macro_f1 > 0.95:
        print("   ⚠️  Baseline kata sederhana dapat F1 > 95%. Test set sangat mudah ditebak permukaan kata.")
    elif macro_f1 >= 0.70 and macro_f1 <= 0.90:
        print("   ✅ IDEAL! Model bag-of-words dapat ~70-90%. Test set cukup kaya, butuh pemahaman semantik XLM-R untuk capai 95%+.")
    else:
        print(f"   ℹ️  Baseline F1: {macro_f1:.4f}")

if __name__ == "__main__":
    run_stage_0()
    run_stage_1()
