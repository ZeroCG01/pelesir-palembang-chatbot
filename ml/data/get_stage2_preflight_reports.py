"""
Script Ekstraksi Laporan Pre-Flight Stage 2 (5-Point User Request)
"""
import os
import re
import csv
import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

def run_preflight_reports():
    print("=" * 80)
    print("POIN 1: UKURAN & KOMPOSISI TEST NER TERPISAH (SEEN VS HOLDOUT)")
    print("=" * 80)
    
    with open("ml/data/processed/train_ner_v2.json", 'r', encoding='utf-8') as f:
        tr_ner = json.load(f)
    with open("ml/data/processed/val_ner_v2.json", 'r', encoding='utf-8') as f:
        va_ner = json.load(f)
    with open("ml/data/processed/test_ner_seen.json", 'r', encoding='utf-8') as f:
        te_seen = json.load(f)
    with open("ml/data/processed/test_ner_holdout.json", 'r', encoding='utf-8') as f:
        te_holdout = json.load(f)
    with open("ml/data/processed/test_ner_v2.json", 'r', encoding='utf-8') as f:
        te_combined = json.load(f)
    with open("ml/data/processed/train_ner_augmented_v2.json", 'r', encoding='utf-8') as f:
        tr_augmented = json.load(f)
        
    print(f"Train Base NER        : {len(tr_ner)} sampel")
    print(f"Train Augmented NER   : {len(tr_augmented)} sampel")
    print(f"Val NER               : {len(va_ner)} sampel")
    print(f"Test Subset (a) Seen  : {len(te_seen)} sampel ({len(te_seen)/len(te_combined)*100:.1f}% dari test)")
    print(f"Test Subset (b) Holdout: {len(te_holdout)} sampel ({len(te_holdout)/len(te_combined)*100:.1f}% dari test)")
    print(f"Combined Test NER     : {len(te_combined)} sampel")

    print("\n" + "=" * 80)
    print("POIN 2 & 3: KEBOCORAN SAME-TYPE HOLDOUT LEAK (TARGET = 0)")
    print("=" * 80)
    
    with open("ml/data/processed/ner_holdout_entities.json", 'r', encoding='utf-8') as f:
        holdout_dict = json.load(f)
        
    # Extract train entity values per type
    train_entities_by_type = defaultdict(set)
    for sample in tr_augmented:
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
                train_entities_by_type[ent_type].add(val)
                i = j
            else:
                i += 1

    same_type_leaks = []
    print(f"{'ENTITY TYPE':15s} | {'HOLDOUT VALUES':15s} | {'SAME-TYPE LEAK':15s} | {'STATUS':10s}")
    print("-" * 65)
    
    for ent_type in ['DESTINATION', 'LOCATION', 'PRICE', 'TIME', 'CATEGORY']:
        h_vals = holdout_dict.get(ent_type, [])
        tr_vals = train_entities_by_type.get(ent_type, set())
        leak_cnt = 0
        for h in h_vals:
            h_lower = h.lower()
            # Strict same-type equality or substring match
            if any(h_lower == tr or h_lower in tr for tr in tr_vals):
                leak_cnt += 1
                same_type_leaks.append((ent_type, h, [tr for tr in tr_vals if h_lower == tr or h_lower in tr]))
        status = "✅ 0% Leak" if leak_cnt == 0 else "❌ LEAK!"
        print(f"{ent_type:15s} | {len(h_vals):15d} | {leak_cnt:15d} | {status:10s}")

    print(f"\nTotal Same-Type Leaks: {len(same_type_leaks)}")
    if same_type_leaks:
        print("Sample Same-Type Leaks:")
        for etype, hval, tr_matches in same_type_leaks:
            print(f"  [{etype}] Holdout \"{hval}\" matches Train entity: {tr_matches}")

    print("\n" + "=" * 80)
    print("POIN 4: STACKED FILLER CLEANUP REPORT (BEFORE VS AFTER)")
    print("=" * 80)
    
    # Run dataset generator & pipeline cleanly
    import generate_intent_dataset_v2
    import generate_ner_dataset_v2
    
    # Read intent dataset & check stacked fillers before/after
    df_raw = pd.read_csv("ml/data/processed/train_intents_raw_v2.csv")
    cleaned_rows = 0
    examples = []
    
    for idx, row in df_raw.iterrows():
        t_orig = row['text']
        t_clean = generate_intent_dataset_v2.clean_stacked_fillers(t_orig)
        if t_orig != t_clean:
            cleaned_rows += 1
            if len(examples) < 10:
                examples.append((t_orig, t_clean))
                
    print(f"Total Baris Intent Dibersihkan dari Stacked Fillers: {cleaned_rows} baris")
    print("\n--- 10 CONTOH BEFORE / AFTER STACKED FILLER CLEANUP ---")
    for idx, (b, a) in enumerate(examples, 1):
        print(f"{idx:2d}. BEFORE : \"{b}\"")
        print(f"    AFTER  : \"{a}\"")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("POIN 5: OVERLAP LABEL TERIMA KASIH (provide_feedback VS goodbye)")
    print("=" * 80)
    
    df_intent_tr = pd.read_csv("ml/data/processed/train_intents_v2.csv")
    fb_thanks = df_intent_tr[(df_intent_tr['label'] == 'provide_feedback') & (df_intent_tr['text'].str.contains('terima kasih|makasih', case=False, na=False))]
    gb_thanks = df_intent_tr[(df_intent_tr['label'] == 'goodbye') & (df_intent_tr['text'].str.contains('terima kasih|makasih', case=False, na=False))]
    
    print(f"Jumlah sampel 'terima kasih' di provide_feedback : {len(fb_thanks)}")
    print(f"Jumlah sampel 'terima kasih' di goodbye          : {len(gb_thanks)}")
    
    print("\n--- SAMPEL KALIMAT provide_feedback (TERIMA KASIH) ---")
    for idx, t in enumerate(fb_thanks['text'].tolist()[:8], 1):
        print(f" {idx}. \"{t}\"")
        
    print("\n--- SAMPEL KALIMAT goodbye (TERIMA KASIH) ---")
    for idx, t in enumerate(gb_thanks['text'].tolist()[:8], 1):
        print(f" {idx}. \"{t}\"")

if __name__ == "__main__":
    run_preflight_reports()
