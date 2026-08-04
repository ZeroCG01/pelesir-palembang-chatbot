"""
Script Ekstraksi Laporan Verifikasi Akhir (User Prompt 5-Point Verification)
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

def run_verification():
    print("=" * 80)
    print("POIN 1: KOMPOSISI TEST NER (HOLDOUT SUBSET VS SEEN-ENTITY SUBSET)")
    print("=" * 80)
    
    with open("ml/data/processed/ner_holdout_entities.json", 'r', encoding='utf-8') as f:
        holdout_dict = json.load(f)
    holdout_all = set(v.lower() for v_list in holdout_dict.values() for v in v_list)
    
    with open("ml/data/processed/test_ner_v2.json", 'r', encoding='utf-8') as f:
        test_ner = json.load(f)
    with open("ml/data/processed/train_ner_augmented_v2.json", 'r', encoding='utf-8') as f:
        train_ner = json.load(f)
        
    holdout_subset = []
    seen_subset = []
    
    for sample in test_ner:
        tokens = sample['tokens']
        tags   = sample['tags']
        has_holdout = False
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith('B-'):
                ent_type = tag[2:]
                j = i + 1
                while j < len(tags) and tags[j] == f'I-{ent_type}':
                    j += 1
                val = " ".join(tokens[i:j]).lower()
                if val in holdout_dict.get(ent_type, []):
                    has_holdout = True
                    break
                i = j
            else:
                i += 1
        if has_holdout:
            holdout_subset.append(sample)
        else:
            seen_subset.append(sample)
            
    print(f"Total Test Samples NER         : {len(test_ner)}")
    print(f"  - Subset (a) Holdout-Entity  : {len(holdout_subset)} sampel ({len(holdout_subset)/len(test_ner)*100:.2f}%)")
    print(f"  - Subset (b) Seen-Entity     : {len(seen_subset)} sampel ({len(seen_subset)/len(test_ner)*100:.2f}%)")

    print("\n" + "=" * 80)
    print("POIN 2: TOKEN-LEVEL HOLDOUT LEAK CHECK (SURFACE TOKEN/SUBSTRING DI TRAIN SET)")
    print("=" * 80)
    
    # Train set tokens & entity spans
    train_tokens_flat = []
    train_entity_spans = [] # (val_lower, ent_type, sentence_str)
    train_sentences_list = []
    
    for sample in train_ner:
        s_tokens = [t.lower() for t in sample['tokens']]
        s_tags   = sample['tags']
        sent_str = " ".join(sample['tokens'])
        train_sentences_list.append((s_tokens, s_tags, sent_str))
        
    leaks_by_type = defaultdict(list)
    leaks_count_by_type = Counter()
    
    for ent_type, h_values in holdout_dict.items():
        for h_val in h_values:
            h_val_lower = h_val.lower()
            h_tokens = h_val_lower.split()
            
            # Cek kemunculan di train set
            found_in_train = False
            leak_context = None
            
            for s_tokens, s_tags, sent_str in train_sentences_list:
                # 1. Cek apakah holdout value muncul sebagai substring di entity lain
                # 2. Cek apakah token holdout muncul saat tag == 'O'
                i = 0
                while i < len(s_tags):
                    tag = s_tags[i]
                    if tag.startswith('B-'):
                        etype = tag[2:]
                        j = i + 1
                        while j < len(s_tags) and s_tags[j] == f'I-{etype}':
                            j += 1
                        ent_val = " ".join(s_tokens[i:j])
                        if h_val_lower in ent_val and h_val_lower != ent_val:
                            found_in_train = True
                            leak_context = f"Substring di [{etype}] \"{ent_val}\" dalam: \"{sent_str}\""
                            break
                        i = j
                    else:
                        tok = s_tokens[i]
                        if tok in h_tokens and len(tok) > 2: # non-trivial word match in 'O'
                            found_in_train = True
                            leak_context = f"Token '{tok}' di-tag 'O' dalam: \"{sent_str}\""
                            break
                        i += 1
                if found_in_train:
                    break
                    
            if found_in_train:
                leaks_count_by_type[ent_type] += 1
                leaks_by_type[ent_type].append((h_val, leak_context))

    print(f"{'ENTITY TYPE':15s} | {'HOLDOUT VALUES':15s} | {'TOKEN LEAK TO TRAIN':20s} | {'% LEAK':10s}")
    print("-" * 70)
    for ent_type in ['DESTINATION', 'LOCATION', 'PRICE', 'TIME', 'CATEGORY']:
        h_vals = holdout_dict.get(ent_type, [])
        l_cnt = leaks_count_by_type[ent_type]
        pct = (l_cnt / max(1, len(h_vals))) * 100
        print(f"{ent_type:15s} | {len(h_vals):15d} | {l_cnt:20d} | {pct:9.2f}%")

    print("\n--- 10 CONTOH PASANGAN TOKEN LEAK (HOLDOUT VALUE -> KONTEKS TRAIN) ---")
    example_idx = 1
    for ent_type, leak_list in leaks_by_type.items():
        for h_val, ctx in leak_list:
            if example_idx > 10:
                break
            print(f"{example_idx:2d}. [{ent_type}] Holdout: \"{h_val}\"")
            print(f"    Context: {ctx}")
            print("-" * 60)
            example_idx += 1

    print("\n" + "=" * 80)
    print("POIN 3: 7 KALIMAT TEST ask_ticket_price -> ask_destination_info + NEAREST TRAIN NEIGHBOR")
    print("=" * 80)
    
    train_df = pd.read_csv("ml/data/processed/train_intents_v2.csv")
    test_df  = pd.read_csv("ml/data/processed/test_intents_v2.csv")
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df['text'])
    X_test  = vec.transform(test_df['text'])
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_df['label'])
    
    preds = clf.predict(X_test)
    test_df['pred'] = preds
    
    misclass_price = test_df[(test_df['label'] == 'ask_ticket_price') & (test_df['pred'] == 'ask_destination_info')]
    print(f"Total Misclassifications ask_ticket_price -> ask_destination_info: {len(misclass_price)} sampel\n")
    
    for idx, row in enumerate(misclass_price.to_dict('records'), 1):
        t_text = row['text']
        t_vec = vec.transform([t_text])
        sims = cosine_similarity(t_vec, X_train)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_tr_text = train_df.iloc[best_idx]['text']
        best_tr_lbl  = train_df.iloc[best_idx]['label']
        
        print(f"{idx:2d}. TEST (True: ask_ticket_price | Pred: ask_destination_info):")
        print(f"    \"{t_text}\"")
        print(f"    NEAREST TRAIN (Similarity: {best_sim:.4f} | Label: {best_tr_lbl}):")
        print(f"    \"{best_tr_text}\"")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("POIN 4: 7 KALIMAT TEST provide_feedback -> goodbye + NEAREST TRAIN NEIGHBOR")
    print("=" * 80)
    
    misclass_fb = test_df[(test_df['label'] == 'provide_feedback') & (test_df['pred'] == 'goodbye')]
    print(f"Total Misclassifications provide_feedback -> goodbye: {len(misclass_fb)} sampel\n")
    
    for idx, row in enumerate(misclass_fb.to_dict('records'), 1):
        t_text = row['text']
        t_vec = vec.transform([t_text])
        sims = cosine_similarity(t_vec, X_train)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_tr_text = train_df.iloc[best_idx]['text']
        best_tr_lbl  = train_df.iloc[best_idx]['label']
        
        print(f"{idx:2d}. TEST (True: provide_feedback | Pred: goodbye):")
        print(f"    \"{t_text}\"")
        print(f"    NEAREST TRAIN (Similarity: {best_sim:.4f} | Label: {best_tr_lbl}):")
        print(f"    \"{best_tr_text}\"")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("POIN 5: HASIL DEDUP EXACT-DUPLICATE NER DENGAN CASEFOLD()")
    print("=" * 80)
    
    train_sent_casefold = set(" ".join(x['tokens']).casefold() for x in train_ner)
    test_sent_casefold  = set(" ".join(x['tokens']).casefold() for x in test_ner)
    
    exact_casefold_dups = train_sent_casefold & test_sent_casefold
    print(f"Jumlah Exact-Duplicate NER dengan casefold(): {len(exact_casefold_dups)}/{len(test_ner)} ({len(exact_casefold_dups)/len(test_ner)*100:.4f}%)")

if __name__ == "__main__":
    run_verification()
