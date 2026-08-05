import os
import csv
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

ID_DIALECT_REVERSE = {
    r'\bapo\b': 'apa', r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak',
    r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana', r'\bnian\b': 'sangat',
    r'\bkau\b': 'kamu', r'\blemak\b': 'enak', r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga',
    r'\bberapo\b': 'berapa', r'\bngapo\b': 'kenapa', r'\bbae\b': 'saja', r'\byo\b': 'ya',
    r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
}

def normalize_words(text):
    t = text.lower()
    for pat, repl in ID_DIALECT_REVERSE.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return set(t.split())

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def run_all_checks():
    proc_dir = "processed"
    
    # -------------------------------------------------------------------
    # 1 & 1b. PATHS IN retrain_colab.py
    # -------------------------------------------------------------------
    print("=== (1 & 1b) PATH FILE TRAIN INTENT & NER DI retrain_colab.py ===")
    colab_path = "../training/retrain_colab.py"
    with open(colab_path, "r", encoding="utf-8") as f:
        colab_lines = f.readlines()
        
    intent_train_line = ""
    ner_train_line = ""
    for idx, line in enumerate(colab_lines, 1):
        if 'train_intents' in line and 'train_path' in line:
            intent_train_line = f"Baris {idx}: {line.strip()}"
        if 'train_ner' in line and 'train_path' in line:
            ner_train_line = f"Baris {idx}: {line.strip()}"
            
    print("Kode Path Train Intent:")
    print("  ", intent_train_line)
    
    intent_csv_path = os.path.join(proc_dir, "train_intents_v2.csv")
    intent_raw_path = os.path.join(proc_dir, "train_intents_raw_v2.csv")
    intent_df_colab = pd.read_csv(intent_csv_path)
    intent_df_raw = pd.read_csv(intent_raw_path)
    print(f"  Nama File: train_intents_v2.csv | Jumlah Baris: {len(intent_df_colab)}")
    print(f"  Nama File (Raw): train_intents_raw_v2.csv | Jumlah Baris: {len(intent_df_raw)}")
    
    # Evaluasi Proxy Model di train_intents_v2.csv (9.813 baris)
    test_df = pd.read_csv(os.path.join(proc_dir, "test_intents_v2.csv"))
    vec_colab = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr_c = vec_colab.fit_transform(intent_df_colab['text'])
    X_te_c = vec_colab.transform(test_df['text'])
    clf_c = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_c.fit(X_tr_c, intent_df_colab['label'])
    preds_c = clf_c.predict(X_te_c)
    acc_c = accuracy_score(test_df['label'], preds_c)
    f1_c = f1_score(test_df['label'], preds_c, average='macro')
    
    print(f"\nEvaluasi Proxy Model di train_intents_v2.csv ({len(intent_df_colab)} baris vs test {len(test_df)} baris):")
    print(f"  Accuracy : {acc_c:.4f} ({acc_c*100:.2f}%)")
    print(f"  Macro F1 : {f1_c:.4f} ({f1_c*100:.2f}%)")

    print("\nKode Path Train NER:")
    print("  ", ner_train_line)
    ner_path = os.path.join(proc_dir, "train_ner_augmented_v2.json")
    with open(ner_path, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    print(f"  Nama File: train_ner_augmented_v2.json | Jumlah Baris: {len(ner_data)}")
    print(f"  Konfirmasi = train_ner_augmented_v2.json: {os.path.basename(ner_path) == 'train_ner_augmented_v2.json'}")

    # -------------------------------------------------------------------
    # 2 & 2b. 9 KALIMAT TEST ask_recommendation DIPREDIKSI goodbye + NEAREST NEIGHBOR TRAIN
    # -------------------------------------------------------------------
    print("\n=== (2 & 2b) 9 KALIMAT TEST ask_recommendation DIPREDIKSI goodbye + NEAREST NEIGHBOR TRAIN ===")
    
    # Train on raw dataset for error tracing
    vec_raw = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr_raw = vec_raw.fit_transform(intent_df_raw['text'])
    X_te_raw = vec_raw.transform(test_df['text'])
    clf_raw = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_raw.fit(X_tr_raw, intent_df_raw['label'])
    preds_raw = clf_raw.predict(X_te_raw)
    
    rec_indices = test_df[(test_df['label'] == 'ask_recommendation') & (preds_raw == 'goodbye')].index
    print(f"Jumlah sampel test ask_recommendation -> goodbye: {len(rec_indices)}")
    
    for idx_num, idx in enumerate(rec_indices, 1):
        test_txt = test_df.loc[idx, 'text']
        test_vec = vec_raw.transform([test_txt])
        
        # Calculate Cosine Sim vs all train samples
        sims = cosine_similarity(test_vec, X_tr_raw).flatten()
        nn_idx = np.argmax(sims)
        nn_sim = sims[nn_idx]
        nn_txt = intent_df_raw.loc[nn_idx, 'text']
        nn_label = intent_df_raw.loc[nn_idx, 'label']
        
        print(f"\n{idx_num}. TEKS TEST  : \"{test_txt}\"")
        print(f"   PREDIKSI   : goodbye (True: ask_recommendation)")
        print(f"   NN TRAIN   : \"{nn_txt}\" (Label: {nn_label}, Cosine Sim: {nn_sim:.4f})")

    # -------------------------------------------------------------------
    # 3. SAMPEL TRAIN ask_recommendation DENGAN FRASA PENUTUP
    # -------------------------------------------------------------------
    print("\n=== (3) SAMPEL TRAIN ask_recommendation DENGAN KATA PENUTUP ('udah', 'yaudah', 'gitu aja', 'makasih') ===")
    keywords = ["udah", "yaudah", "gitu aja", "makasih"]
    matching_train = []
    for idx, row in intent_df_raw[intent_df_raw['label'] == 'ask_recommendation'].iterrows():
        txt = row['text'].lower()
        if any(kw in txt for kw in keywords):
            matching_train.append(row['text'])
            
    print(f"Total Sampel Train ask_recommendation Berisi Kata Penutup: {len(matching_train)}")
    for i, t in enumerate(matching_train[:10], 1):
        print(f"  {i}. \"{t}\"")

    # -------------------------------------------------------------------
    # 4. LABEL NOISE / OVERLAP TRAIN ask_recommendation VS goodbye
    # -------------------------------------------------------------------
    print("\n=== (4) LABEL NOISE TRAIN: ask_recommendation VS goodbye (Jaccard > 0.85 / Exact) ===")
    rec_train = intent_df_raw[intent_df_raw['label'] == 'ask_recommendation']['text'].tolist()
    gb_train  = intent_df_raw[intent_df_raw['label'] == 'goodbye']['text'].tolist()
    
    gb_wordsets = [(txt, normalize_words(txt)) for txt in gb_train]
    
    overlaps = []
    for r_txt in rec_train:
        r_set = normalize_words(r_txt)
        for g_txt, g_set in gb_wordsets:
            j_score = jaccard(r_set, g_set)
            if j_score > 0.85:
                overlaps.append((r_txt, g_txt, j_score))
                
    print(f"Total Pasangan Overlap/Noise ask_recommendation vs goodbye (Jaccard > 0.85): {len(overlaps)}")
    for i, (r_txt, g_txt, score) in enumerate(overlaps[:10], 1):
        print(f"  {i}. REC: \"{r_txt}\" | GOODBYE: \"{g_txt}\" | Jaccard: {score:.4f}")

if __name__ == "__main__":
    run_all_checks()
