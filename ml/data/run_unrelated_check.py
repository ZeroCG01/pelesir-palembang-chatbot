import os
import re
import csv
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def run_unrelated_check():
    proc_dir = "processed"
    intent_raw_path = os.path.join(proc_dir, "train_intents_raw_v2.csv")
    test_path = os.path.join(proc_dir, "test_intents_v2.csv")
    
    intent_df_raw = pd.read_csv(intent_raw_path)
    test_df = pd.read_csv(test_path)
    
    vec_raw = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr_raw = vec_raw.fit_transform(intent_df_raw['text'])
    X_te_raw = vec_raw.transform(test_df['text'])
    clf_raw = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_raw.fit(X_tr_raw, intent_df_raw['label'])
    preds_raw = clf_raw.predict(X_te_raw)
    
    # Check ask_recommendation predicted as ask_unrelated (9 samples)
    rec_indices = test_df[(test_df['label'] == 'ask_recommendation') & (preds_raw == 'ask_unrelated')].index
    print(f"=== (2 & 2b) 9 KALIMAT TEST ask_recommendation DIPREDIKSI ask_unrelated + NEAREST NEIGHBOR TRAIN ===")
    print(f"Jumlah sampel test ask_recommendation -> ask_unrelated: {len(rec_indices)}")
    
    for idx_num, idx in enumerate(rec_indices, 1):
        test_txt = test_df.loc[idx, 'text']
        test_vec = vec_raw.transform([test_txt])
        
        sims = cosine_similarity(test_vec, X_tr_raw).flatten()
        nn_idx = np.argmax(sims)
        nn_sim = sims[nn_idx]
        nn_txt = intent_df_raw.loc[nn_idx, 'text']
        nn_label = intent_df_raw.loc[nn_idx, 'label']
        
        print(f"\n{idx_num}. TEKS TEST  : \"{test_txt}\"")
        print(f"   PREDIKSI   : ask_unrelated (True: ask_recommendation)")
        print(f"   NN TRAIN   : \"{nn_txt}\" (Label: {nn_label}, Cosine Sim: {nn_sim:.4f})")

    print("\n=== (3) SAMPEL TRAIN ask_recommendation DENGAN KATA PENUTUP/IMPLISIT ===")
    keywords = ["udah", "yaudah", "gitu aja", "makasih", "sore", "tenang", "healing"]
    matching_train = []
    for idx, row in intent_df_raw[intent_df_raw['label'] == 'ask_recommendation'].iterrows():
        txt = row['text'].lower()
        if any(kw in txt for kw in keywords):
            matching_train.append(row['text'])
            
    print(f"Total Sampel Train ask_recommendation Berisi Kata Kunci Santai/Implisit: {len(matching_train)}")
    for i, t in enumerate(matching_train[:10], 1):
        print(f"  {i}. \"{t}\"")

    print("\n=== (4) LABEL NOISE TRAIN: ask_recommendation VS ask_unrelated (Jaccard > 0.85 / Exact) ===")
    rec_train = intent_df_raw[intent_df_raw['label'] == 'ask_recommendation']['text'].tolist()
    unrel_train = intent_df_raw[intent_df_raw['label'] == 'ask_unrelated']['text'].tolist()
    
    unrel_wordsets = [(txt, normalize_words(txt)) for txt in unrel_train]
    
    overlaps = []
    for r_txt in rec_train:
        r_set = normalize_words(r_txt)
        for u_txt, u_set in unrel_wordsets:
            j_score = jaccard(r_set, u_set)
            if j_score > 0.85:
                overlaps.append((r_txt, u_txt, j_score))
                
    print(f"Total Pasangan Overlap/Noise ask_recommendation vs ask_unrelated (Jaccard > 0.85): {len(overlaps)}")
    for i, (r_txt, u_txt, score) in enumerate(overlaps[:10], 1):
        print(f"  {i}. REC: \"{r_txt}\" | UNRELATED: \"{u_txt}\" | Jaccard: {score:.4f}")

if __name__ == "__main__":
    run_unrelated_check()
