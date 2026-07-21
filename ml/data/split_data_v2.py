"""
Split Dataset v2 — Stratified 80:10:10
Untuk Intent (CSV) dan NER (JSON)
"""
import os
import csv
import json
import random
from collections import defaultdict

random.seed(42)

def split_intent_dataset():
    """Split intent dataset dengan stratifikasi per kelas"""
    input_path = "ml/data/raw/intents_bilingual_v2.csv"
    
    print(f"Membaca {input_path}...")
    data_by_label = defaultdict(list)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_by_label[row['label']].append(row)
    
    train_data, val_data, test_data = [], [], []
    
    for label, items in sorted(data_by_label.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
        
        print(f"  {label}: {len(items)} → train={n_train}, val={n_val}, test={n - n_train - n_val}")
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Simpan
    os.makedirs("ml/data/processed", exist_ok=True)
    
    for name, data in [("train_intents_v2.csv", train_data), 
                        ("val_intents_v2.csv", val_data), 
                        ("test_intents_v2.csv", test_data)]:
        path = f"ml/data/processed/{name}"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {path} ({len(data)} rows)")
    
    print(f"\nTotal: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


def split_ner_dataset():
    """Split NER dataset dengan stratifikasi"""
    input_path = "ml/data/raw/ner_dataset_v2.json"
    
    if not os.path.exists(input_path):
        print(f"\n[SKIP] {input_path} belum ada. Jalankan generate_ner_dataset_v2.py dulu.")
        return
    
    print(f"\nMembaca {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.shuffle(data)
    n = len(data)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    
    os.makedirs("ml/data/processed", exist_ok=True)
    
    for name, split in [("train_ner_v2.json", train), 
                         ("val_ner_v2.json", val), 
                         ("test_ner_v2.json", test)]:
        path = f"ml/data/processed/{name}"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path} ({len(split)} samples)")
    
    print(f"Total NER: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    split_intent_dataset()
    split_ner_dataset()
