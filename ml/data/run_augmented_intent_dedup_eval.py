import os
import re
import csv
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

def normalize_words(text):
    t = str(text).lower()
    for pat, repl in ID_DIALECT_REVERSE.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return set(t.split())

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def run_evaluation():
    proc_dir = "processed"
    tr_aug_path = os.path.join(proc_dir, "train_intents_v2.csv")
    tr_raw_path = os.path.join(proc_dir, "train_intents_raw_v2.csv")
    te_path     = os.path.join(proc_dir, "test_intents_v2.csv")

    df_tr_aug = pd.read_csv(tr_aug_path)
    df_tr_raw = pd.read_csv(tr_raw_path)
    df_te     = pd.read_csv(te_path)

    # -------------------------------------------------------------------
    # 3. CEK LEAKAGE 3.372 SAMPEL AUGMENTASI VS TEST
    # -------------------------------------------------------------------
    print("=== (3) DETEKSI LEAKAGE SAMPEL AUGMENTASI (3.372 BARS) VS TEST SET ===")
    raw_texts_set = set(df_tr_raw['text'].str.strip().str.lower())
    
    # Filter sampel yang murni hasil augmentasi (tidak ada di raw)
    aug_only_rows = df_tr_aug[~df_tr_aug['text'].str.strip().str.lower().isin(raw_texts_set)]
    print(f"Jumlah sampel murni hasil augmentasi di train_intents_v2.csv: {len(aug_only_rows)}")

    test_wordsets = [(row['text'], row['label'], normalize_words(row['text'])) for _, row in df_te.iterrows()]
    
    aug_leaks = []
    for _, a_row in aug_only_rows.iterrows():
        a_txt = a_row['text']
        a_lbl = a_row['label']
        a_words = normalize_words(a_txt)
        
        for t_txt, t_lbl, t_words in test_wordsets:
            j_score = jaccard(a_words, t_words)
            if j_score > 0.85:
                aug_leaks.append((a_txt, a_lbl, t_txt, t_lbl, j_score))

    print(f"Total Pasangan Sampel Augmentasi -> Test yang Jaccard > 0.85: {len(aug_leaks)}")
    print("\nContoh 10 Pasangan Leakage (Augmented Train -> Test):")
    for idx, (a_t, a_l, t_t, t_l, score) in enumerate(aug_leaks[:10], 1):
        print(f"{idx:2d}. AUG TRAIN [{a_l}] : \"{a_t}\"")
        print(f"    TEST [{t_l}]      : \"{t_t}\" | Jaccard: {score:.4f}")

    # -------------------------------------------------------------------
    # 1. PASCA-DEDUP INTENT TEST VS TRAIN_INTENTS_V2.CSV (AUGMENTED 9.813)
    # -------------------------------------------------------------------
    print("\n=== (1) PASCA-DEDUP TEST SET VS TRAIN_INTENTS_V2.CSV (9.813 BARS) ===")
    tr_aug_exact = set(df_tr_aug['text'].str.strip().str.lower())
    tr_aug_wordsets = [normalize_words(t) for t in df_tr_aug['text']]

    clean_test_rows = []
    dropped_test_count = 0

    for _, row in df_te.iterrows():
        txt = row['text'].strip()
        txt_lower = txt.lower()

        if txt_lower in tr_aug_exact:
            dropped_test_count += 1
            continue

        w_set = normalize_words(txt)
        is_near_dup = False
        for tr_w in tr_aug_wordsets:
            if jaccard(w_set, tr_w) > 0.85:
                is_near_dup = True
                break

        if is_near_dup:
            dropped_test_count += 1
        else:
            clean_test_rows.append(row)

    df_te_clean = pd.DataFrame(clean_test_rows)
    print(f"Sampel Test Dibuang : {dropped_test_count} ({len(df_te)} -> {len(df_te_clean)})")
    
    # Save clean test set if items were dropped, or save to temporary evaluation dataframe
    df_te_clean.to_csv("processed/test_intents_v2_dedup.csv", index=False)

    # -------------------------------------------------------------------
    # 2. RE-RUN PROXY MODEL TRAINED ON train_intents_v2.csv VS CLEAN TEST SET
    # -------------------------------------------------------------------
    print("\n=== (2) PROXY MODEL (TF-IDF + LOGREG) DI train_intents_v2.csv VS CLEAN TEST SET ===")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr = vec.fit_transform(df_tr_aug['text'])
    X_te = vec.transform(df_te_clean['text'])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, df_tr_aug['label'])

    preds = clf.predict(X_te)
    acc = accuracy_score(df_te_clean['label'], preds)
    macro_f1 = f1_score(df_te_clean['label'], preds, average='macro')

    labels = sorted(list(set(df_tr_aug['label'])))
    cm = confusion_matrix(df_te_clean['label'], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(f"Ukuran Train Augmented : {len(df_tr_aug)} sampel")
    print(f"Ukuran Test Final      : {len(df_te_clean)} sampel")
    print(f"Accuracy               : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1               : {macro_f1:.4f} ({macro_f1*100:.2f}%)")

    print("\n--- CONFUSION MATRIX LENGKAP ---")
    print(cm_df.to_string())

if __name__ == "__main__":
    run_evaluation()
