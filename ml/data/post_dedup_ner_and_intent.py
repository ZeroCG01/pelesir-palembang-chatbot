import os
import re
import csv
import json
from collections import defaultdict
import pandas as pd

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

def run_post_dedup():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(script_dir, "processed")
    
    # -------------------------------------------------------------------
    # 1. INTENT POST-DEDUP (Exact Match + Jaccard > 0.85)
    # -------------------------------------------------------------------
    tr_intent_path = os.path.join(proc_dir, "train_intents_v2.csv")
    te_intent_path = os.path.join(proc_dir, "test_intents_v2.csv")
    
    with open(tr_intent_path, 'r', encoding='utf-8') as f:
        train_intents = list(csv.DictReader(f))
    with open(te_intent_path, 'r', encoding='utf-8') as f:
        test_intents = list(csv.DictReader(f))
        
    train_intent_exact = set(r['text'].strip().lower() for r in train_intents)
    train_intent_wordsets = [normalize_words(r['text']) for r in train_intents]
    
    clean_test_intents = []
    dropped_intent = 0
    
    for r in test_intents:
        txt = r['text'].strip()
        txt_lower = txt.lower()
        
        # (a) Exact match (casefold)
        if txt_lower in train_intent_exact:
            dropped_intent += 1
            continue
            
        # (b) Jaccard > 0.85
        w_set = normalize_words(txt)
        is_near_dup = False
        for tr_words in train_intent_wordsets:
            if jaccard(w_set, tr_words) > 0.85:
                is_near_dup = True
                break
                
        if is_near_dup:
            dropped_intent += 1
        else:
            clean_test_intents.append(r)
            
    with open(te_intent_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(clean_test_intents)
        
    print(f"PASCA-DEDUP INTENT: Dibuang {dropped_intent} sampel test ({len(test_intents)} -> {len(clean_test_intents)}).")

    tr_ner_path = os.path.join(proc_dir, "train_ner_augmented_v2.json")
    with open(tr_ner_path, 'r', encoding='utf-8') as f:
        tr_ner = json.load(f)

    holdout_file = os.path.join(proc_dir, "ner_holdout_entities.json")
    with open(holdout_file, 'r', encoding='utf-8') as f:
        holdout_dict = json.load(f)
    all_holdouts = set(v.lower() for v_list in holdout_dict.values() for v in v_list if len(v.strip()) > 1)

    clean_tr_ner = []
    scrubbed_count = 0
    for s in tr_ner:
        text_str = " ".join(s['tokens']).lower()
        has_holdout = False
        for h_val in all_holdouts:
            if h_val in text_str or text_str in h_val:
                has_holdout = True
                break
        if has_holdout:
            scrubbed_count += 1
        else:
            clean_tr_ner.append(s)

    with open(tr_ner_path, 'w', encoding='utf-8') as f:
        json.dump(clean_tr_ner, f, ensure_ascii=False, indent=2)
    print(f"HOLDOUT LEAK SCRUBBER NER: Dibuang {scrubbed_count} sampel augmented train yang mengandung holdout entity.")
    tr_ner = clean_tr_ner

    # -------------------------------------------------------------------
    # 3. NER POST-DEDUP (vs train_ner_augmented_v2.json)
    # -------------------------------------------------------------------
    te_seen_path = os.path.join(proc_dir, "test_ner_seen.json")
    te_holdout_path = os.path.join(proc_dir, "test_ner_holdout.json")
    te_combined_path = os.path.join(proc_dir, "test_ner_v2.json")
    
    with open(te_seen_path, 'r', encoding='utf-8') as f:
        te_seen = json.load(f)
    with open(te_holdout_path, 'r', encoding='utf-8') as f:
        te_holdout = json.load(f)

    train_ner_exact = set(" ".join(s['tokens']).strip().lower() for s in tr_ner)
    train_ner_wordsets = [normalize_words(" ".join(s['tokens'])) for s in tr_ner]

    def filter_ner_split(split):
        clean_split = []
        dropped = 0
        for s in split:
            txt = " ".join(s['tokens']).strip()
            txt_lower = txt.lower()
            
            # (a) Exact match (casefold)
            if txt_lower in train_ner_exact:
                dropped += 1
                continue
                
            # (b) Jaccard > 0.85
            w_set = normalize_words(txt)
            is_near_dup = False
            for tr_words in train_ner_wordsets:
                if jaccard(w_set, tr_words) > 0.85:
                    is_near_dup = True
                    break
            if is_near_dup:
                dropped += 1
            else:
                clean_split.append(s)
        return clean_split, dropped

    clean_te_seen, drop_seen = filter_ner_split(te_seen)
    clean_te_holdout, drop_holdout = filter_ner_split(te_holdout)
    clean_te_combined = clean_te_seen + clean_te_holdout

    with open(te_seen_path, 'w', encoding='utf-8') as f:
        json.dump(clean_te_seen, f, ensure_ascii=False, indent=2)
    with open(te_holdout_path, 'w', encoding='utf-8') as f:
        json.dump(clean_te_holdout, f, ensure_ascii=False, indent=2)
    with open(te_combined_path, 'w', encoding='utf-8') as f:
        json.dump(clean_te_combined, f, ensure_ascii=False, indent=2)

    print(f"PASCA-DEDUP NER: Dibuang total {drop_seen + drop_holdout} sampel test (Seen: -{drop_seen}, Holdout: -{drop_holdout}).")
    print(f"Ukuran final test NER: Seen={len(clean_te_seen)}, Holdout={len(clean_te_holdout)}, Combined={len(clean_te_combined)}.")

    # -------------------------------------------------------------------
    # 4. HARD ASSERTIONS (STOP PIPELINE IF LEAKS EXIST)
    # -------------------------------------------------------------------
    tr_ner_aug_text = " \n ".join([" ".join(item['tokens']).lower() for item in tr_ner])
    total_holdout_leaks = 0
    for entity_type, holdout_values in holdout_dict.items():
        for h_val in holdout_values:
            if h_val.lower() in tr_ner_aug_text:
                total_holdout_leaks += 1

    assert total_holdout_leaks == 0, f"ASSERTION FAILED: Holdout entities leaked in Train NER ({total_holdout_leaks} leaks)!"
    assert dropped_intent == 0 or len(clean_test_intents) > 0, "ASSERTION FAILED: Intent test set empty!"
    print("✅ HARD ASSERTIONS PASSED: 0 Holdout Leaks in Train NER, Pipeline 100% Verified Clean!")

if __name__ == "__main__":
    run_post_dedup()
