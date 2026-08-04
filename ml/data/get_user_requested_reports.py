"""
Script Ekstraksi Laporan Mentah (User Prompt 4-Point Request)
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

# Normalization & Jaccard
ID_DIALECT_REVERSE = {
    r'\bapo\b': 'apa', r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak', r'\bndak\b': 'tidak',
    r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana', r'\bnian\b': 'sangat', r'\bbanget\b': 'sangat',
    r'\bkau\b': 'kamu', r'\bambo\b': 'aku', r'\blemak\b': 'enak', r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga',
    r'\bberapo\b': 'berapa', r'\bbrapa\b': 'berapa', r'\bngapo\b': 'kenapa', r'\bknp\b': 'kenapa',
    r'\bkalo\b': 'kalau', r'\bkalu\b': 'kalau', r'\bmen\b': 'kalau', r'\bbae\b': 'saja', r'\byo\b': 'ya',
    r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
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

def run_user_report():
    print("=" * 80)
    print("ITEM 1: LEAKAGE CHECK (JACCARD SIMILARITY > 0.85) SESUDAH FIX FILLERS GET_SIGNATURE()")
    print("=" * 80)
    
    train_df = pd.read_csv("ml/data/processed/train_intents_v2.csv")
    test_df  = pd.read_csv("ml/data/processed/test_intents_v2.csv")
    
    train_rows = train_df.to_dict('records')
    test_rows  = test_df.to_dict('records')
    
    train_by_label = defaultdict(list)
    for r in train_rows:
        train_by_label[r['label']].append(normalize_text(r['text']))
        
    near_leaks = 0
    total_test = len(test_rows)
    for r in test_rows:
        test_set = normalize_text(r['text'])
        label = r['label']
        for tr_set in train_by_label[label]:
            sim = jaccard_similarity(test_set, tr_set)
            if sim > 0.85 and sim < 1.0:
                near_leaks += 1
                break
                
    pct_leak = (near_leaks / max(1, total_test)) * 100
    print(f"Hasil Leakage Sesudah Fix Fillers get_signature(): {near_leaks}/{total_test} ({pct_leak:.2f}%)")

    print("\n" + "=" * 80)
    print("ITEM 2: 30 SAMPLE EXACT-DUPLICATE SENTENCES (TRAIN NER AUGMENTED VS TEST NER)")
    print("=" * 80)
    
    with open("ml/data/processed/train_ner_augmented_v2.json", 'r', encoding='utf-8') as f:
        train_ner = json.load(f)
    with open("ml/data/processed/test_ner_v2.json", 'r', encoding='utf-8') as f:
        test_ner = json.load(f)
        
    train_sent_map = {}
    for sample in train_ner:
        s = " ".join(sample['tokens']).lower()
        if s not in train_sent_map:
            train_sent_map[s] = sample
            
    exact_duplicates = []
    for sample in test_ner:
        s = " ".join(sample['tokens']).lower()
        if s in train_sent_map:
            # extract entity types
            entities = set()
            for tag in sample['tags']:
                if tag.startswith('B-'):
                    entities.add(tag[2:])
            exact_duplicates.append({
                'sentence': " ".join(sample['tokens']),
                'entities': list(entities) if entities else ["tidak ada entity"]
            })
            
    print(f"Total Exact Duplicate Sentences: {len(exact_duplicates)}/{len(test_ner)}")
    print("\n--- 30 SAMPLE EXACT DUPLICATE SENTENCES ---")
    for i, dup in enumerate(exact_duplicates[:30], 1):
        ents_str = ", ".join(dup['entities'])
        print(f"{i:2d}. Kalimat: \"{dup['sentence']}\"")
        print(f"    Entities: [{ents_str}]")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("ITEM 3: DETAIL MISCLASSIFICATION ask_location_access -> ask_destination_info")
    print("=" * 80)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df['text'])
    X_test  = vec.transform(test_df['text'])
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_df['label'])
    
    preds = clf.predict(X_test)
    test_df['pred'] = preds
    
    labels = sorted(list(train_df['label'].unique()))
    cm = confusion_matrix(test_df['label'], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\n📋 CONFUSION MATRIX PROXY MODEL (SUMBU Y = TRUE, SUMBU X = PRED):")
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print(cm_df)

    misclass_df = test_df[(test_df['label'] == 'ask_location_access') & (test_df['pred'] == 'ask_destination_info')]
    print(f"\nTotal Misclassifications ask_location_access -> ask_destination_info: {len(misclass_df)}/{len(test_df[test_df['label']=='ask_location_access'])}")
    
    print("\n--- DETAIL KALIMAT TEST & NEAREST NEIGHBOR DARI TRAIN ---")
    for idx, row in enumerate(misclass_df.to_dict('records'), 1):
        t_text = row['text']
        t_vec = vec.transform([t_text])
        sims = cosine_similarity(t_vec, X_train)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_tr_text = train_df.iloc[best_idx]['text']
        best_tr_lbl  = train_df.iloc[best_idx]['label']
        
        print(f"{idx:2d}. TEST (True: ask_location_access | Pred: ask_destination_info):")
        print(f"    \"{t_text}\"")
        print(f"    NEAREST TRAIN (Similarity: {best_sim:.4f} | Label: {best_tr_lbl}):")
        print(f"    \"{best_tr_text}\"")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("ITEM 4: ENTITY HOLDOUT NER OVERLAP METRICS (20% HOLDOUT EXCLUSIVELY TEST)")
    print("=" * 80)
    
    with open("ml/data/processed/ner_holdout_entities.json", 'r', encoding='utf-8') as f:
        holdout_dict = json.load(f)
        
    holdout_all = set()
    for v_list in holdout_dict.values():
        holdout_all.update(v_list)
        
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
    
    print(f"\n📋 A. OVERLAP UNTUK HOLDOUT ENTITIES (20% TERISOLASI DI TEST SET):")
    print(f"{'ENTITY TYPE':15s} | {'HOLDOUT TEST':12s} | {'FOUND IN TRAIN':15s} | {'% MATCH (HOLDOUT)':18s}")
    print("-" * 75)
    for ent_type in ['DESTINATION', 'LOCATION', 'PRICE', 'TIME', 'CATEGORY']:
        h_vals = set(holdout_dict.get(ent_type, []))
        tr_vals = train_entities.get(ent_type, set())
        found = h_vals & tr_vals
        pct = (len(found) / max(1, len(h_vals))) * 100
        print(f"{ent_type:15s} | {len(h_vals):12d} | {len(found):15d} | {pct:17.2f}%")

    print(f"\n📋 B. OVERLAP KESELURUHAN ENTITY VALUE (HOLDOUT + NON-HOLDOUT):")
    print(f"{'ENTITY TYPE':15s} | {'TOTAL TEST':12s} | {'FOUND IN TRAIN':15s} | {'% MATCH OVERALL':18s}")
    print("-" * 75)
    for ent_type in ['DESTINATION', 'LOCATION', 'PRICE', 'TIME', 'CATEGORY']:
        t_vals = test_entities.get(ent_type, set())
        tr_vals = train_entities.get(ent_type, set())
        found = t_vals & tr_vals
        pct = (len(found) / max(1, len(t_vals))) * 100
        print(f"{ent_type:15s} | {len(t_vals):12d} | {len(found):15d} | {pct:17.2f}%")

if __name__ == "__main__":
    run_user_report()
