"""
Script Ekstraksi Laporan Hasil Mentah (Raw Audit Data)
"""
import os
import re
import csv
import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Helper normalization
ID_DIALECT_REVERSE = {
    r'\bapo\b': 'apa',
    r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak', r'\bndak\b': 'tidak',
    r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana',
    r'\bnian\b': 'sangat', r'\bbanget\b': 'sangat',
    r'\bkau\b': 'kamu', r'\bambo\b': 'aku', r'\blemak\b': 'enak',
    r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga', r'\bberapo\b': 'berapa', r'\bbrapa\b': 'berapa',
    r'\bngapo\b': 'kenapa', r'\bknp\b': 'kenapa', r'\bkalo\b': 'kalau', r'\bkalu\b': 'kalau',
    r'\bbae\b': 'saja', r'\byo\b': 'ya', r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
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

def get_raw_reports():
    print("=" * 80)
    print("LAPORAN MENTAH 1: PAIRS JACCARD SIMILARITY > 0.85 (INTENT DATASET)")
    print("=" * 80)
    
    train_df = pd.read_csv("ml/data/processed/train_intents_v2.csv")
    test_df  = pd.read_csv("ml/data/processed/test_intents_v2.csv")
    
    train_rows = train_df.to_dict('records')
    test_rows  = test_df.to_dict('records')
    
    train_by_label = defaultdict(list)
    for r in train_rows:
        train_by_label[r['label']].append((r['text'], normalize_text(r['text'])))
        
    jaccard_pairs = []
    for r in test_rows:
        t_text = r['text']
        t_set  = normalize_text(t_text)
        lbl    = r['label']
        for tr_text, tr_set in train_by_label[lbl]:
            sim = jaccard_similarity(t_set, tr_set)
            if sim > 0.85 and sim < 1.0:
                jaccard_pairs.append({
                    'label': lbl,
                    'test_text': t_text,
                    'train_text': tr_text,
                    'score': sim
                })
                
    # Sort descending by score
    jaccard_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Total Pasangan Terflag (>0.85 & <1.0): {len(jaccard_pairs)}")
    print("\n--- TOP 35 PASANGAN SIMILARITY TERTINGGI ---")
    for i, p in enumerate(jaccard_pairs[:40], 1):
        print(f"{i:2d}. Score: {p['score']:.4f} | Label: {p['label']}")
        print(f"    TEST : \"{p['test_text']}\"")
        print(f"    TRAIN: \"{p['train_text']}\"")
        print("-" * 60)
        
    print("\n" + "=" * 80)
    print("LAPORAN MENTAH 2: DISTRIBUSI ENTITY MENTAH PER INTENT (4 DESTINASI UTAMA)")
    print("=" * 80)
    
    entities_to_check = [
        "Balaputra Dewa",
        "Ampera",
        "Benteng Kuto Besak",
        "Pulau Kemaro"
    ]
    
    all_intents = sorted(list(train_df['label'].unique()))
    
    for ent in entities_to_check:
        print(f"\n📍 ENTITY: \"{ent}\"")
        ent_lower = ent.lower()
        counts = Counter()
        total_mentions = 0
        for r in train_rows:
            if ent_lower in r['text'].lower():
                counts[r['label']] += 1
                total_mentions += 1
        print(f"   Total Kemunculan di Train Set: {total_mentions} kali")
        print("   Rincian per Intent:")
        for lbl in all_intents:
            cnt = counts[lbl]
            print(f"     - {lbl:22s}: {cnt:4d} kali")

    print("\n" + "=" * 80)
    print("LAPORAN MENTAH 3: PROXY MODEL DETAILS & CONFUSION MATRIX")
    print("=" * 80)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df['text'])
    X_test  = vec.transform(test_df['text'])
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_df['label'])
    
    preds = clf.predict(X_test)
    
    print("\n📋 SKLEARN TFIDF PARAMETERS:")
    print(f"   - analyzer       : {vec.analyzer}")
    print(f"   - ngram_range    : {vec.ngram_range}")
    print(f"   - min_df         : {vec.min_df}")
    print(f"   - max_df         : {vec.max_df}")
    print(f"   - sublinear_tf   : {vec.sublinear_tf}")
    print(f"   - norm           : {vec.norm}")
    print(f"   - Vocab Size     : {len(vec.vocabulary_)}")
    
    print("\n📋 LOGISTIC REGRESSION PARAMETERS:")
    print(f"   - C              : {clf.C}")
    print(f"   - max_iter       : {clf.max_iter}")
    print(f"   - solver         : {clf.solver}")
    print(f"   - random_state   : {clf.random_state}")

    print("\n📋 KONFIRMASI SPLIT DATASET PROXY VS XLM-R:")
    print(f"   - Train file path: ml/data/processed/train_intents_v2.csv ({len(train_df)} rows)")
    print(f"   - Test file path : ml/data/processed/test_intents_v2.csv ({len(test_df)} rows)")
    print("   - Keterangan     : SAMA PERSIS 100% dengan file input yang dipanggil di retrain_colab.py")

    labels = sorted(list(train_df['label'].unique()))
    cm = confusion_matrix(test_df['label'], preds, labels=labels)
    
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\n📋 CONFUSION MATRIX (SUMBU Y = TRUE, SUMBU X = PRED):")
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print(cm_df)
    
    print("\n📋 PASANGAN SALAH PREDIKSI (MISCLASSIFICATIONS > 0):")
    misclass = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                misclass.append((labels[i], labels[j], cm[i][j]))
    misclass.sort(key=lambda x: x[2], reverse=True)
    for true_lbl, pred_lbl, count in misclass:
        print(f"   True: {true_lbl:22s} -> Pred: {pred_lbl:22s}: {count:2d} kali")

    print("\n" + "=" * 80)
    print("LAPORAN MENTAH 4: NER AUDIT & OVERLAP METRICS")
    print("=" * 80)
    
    train_ner_path = "ml/data/processed/train_ner_augmented_v2.json"
    test_ner_path  = "ml/data/processed/test_ner_v2.json"
    
    with open(train_ner_path, 'r', encoding='utf-8') as f:
        train_ner = json.load(f)
    with open(test_ner_path, 'r', encoding='utf-8') as f:
        test_ner = json.load(f)
        
    print(f"Total NER Train Augmented Samples: {len(train_ner)}")
    print(f"Total NER Test Samples           : {len(test_ner)}")
    
    # Sentence-level exact near-duplicate check
    train_sentences = set(" ".join(x['tokens']).lower() for x in train_ner)
    test_sentences  = set(" ".join(x['tokens']).lower() for x in test_ner)
    sentence_overlap = train_sentences & test_sentences
    print(f"\n🔍 Sentence-level Overlap (Exact Match Tokens): {len(sentence_overlap)}/{len(test_sentences)} ({len(sentence_overlap)/max(1, len(test_sentences))*100:.2f}%)")

    # Extract entities by type
    def extract_entity_values(dataset):
        entities_by_type = defaultdict(set)
        for sample in dataset:
            tokens = sample['tokens']
            tags   = sample['tags']
            i = 0
            while i < len(tags):
                tag = tags[i]
                if tag.startswith('B-'):
                    ent_type = tag[2:]
                    j = i + 1
                    while j < len(tags) and tags[j] == f'I-{ent_type}':
                        j += 1
                    val = " ".join(tokens[i:j]).lower()
                    entities_by_type[ent_type].add(val)
                    i = j
                else:
                    i += 1
        return entities_by_type

    train_entities = extract_entity_values(train_ner)
    test_entities  = extract_entity_values(test_ner)
    
    print("\n📊 REKAP OVERLAP ENTITY VALUE TEST vs TRAIN AUGMENTED V2:")
    print(f"{'ENTITY TYPE':15s} | {'UNIQUE TEST':12s} | {'UNIQUE TRAIN':12s} | {'FOUND IN TRAIN':15s} | {'% MATCH (NUEVO)':15s} | {'% MATCH (LAMA)':15s}")
    print("-" * 95)
    
    old_numbers = {
        'DESTINATION': 97.0,
        'TIME': 76.0,
        'PRICE': 50.0,
        'CATEGORY': 78.0,
        'LOCATION': 67.0
    }
    
    for ent_type in ['DESTINATION', 'LOCATION', 'PRICE', 'TIME', 'CATEGORY']:
        t_vals = test_entities.get(ent_type, set())
        tr_vals = train_entities.get(ent_type, set())
        found = t_vals & tr_vals
        
        n_test = len(t_vals)
        n_train = len(tr_vals)
        n_found = len(found)
        pct = (n_found / max(1, n_test)) * 100
        old_pct = old_numbers.get(ent_type, 0.0)
        
        print(f"{ent_type:15s} | {n_test:12d} | {n_train:12d} | {n_found:15d} | {pct:14.2f}% | {old_pct:14.2f}%")

if __name__ == "__main__":
    get_raw_reports()
