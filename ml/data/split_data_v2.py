"""
Split Dataset v2 — Anti Overfitting (Group Split by Template Signature)
Memastikan tidak ada template kalimat yang sama antara Train dan Test.
"""
import os
import csv
import json
import random
import re
from collections import defaultdict

random.seed(42)

# Daftar entitas untuk membuat signature
ALL_DEST = [
    "Benteng Kuto Besak", "Jembatan Ampera", "Pulau Kemaro", "Monpera",
    "Museum Sultan Mahmud Badaruddin II", "Kambang Iwak", "Hutan Wisata Punti Kayu",
    "Masjid Agung Palembang", "Kampung Kapitan", "Kampung Arab Al-Munawar",
    "Taman Kambang Iwak Besak", "Palembang Trade Center", "Palembang Icon",
    "Palembang Square", "Jakabaring Sport City", "Taman Kebon Rojo",
    "Museum Balaputra Dewa", "Al Quran Al Akbar", "Bukit Siguntang",
    "Masjid Cheng Ho", "Fantasy Island", "Amanzi Waterpark",
    "OPI Mall", "Pasar 16 Ilir", "Kawah Tengkurep",
    "BKB", "SMB", "SMB II", "PTC", "PIM", "Ampera", "Kemaro", "Monpera",
    "Punti Kayu", "Kambang Iwak", "KI", "Al-Munawar", "Kampung Kapitan",
    "Jakabaring", "Siguntang", "Amanzi",
]

def get_signature(text):
    """Menghapus entitas spesifik agar tersisa hanya kerangka template (grammar)"""
    sig = text.lower()
    # Sort dari terpanjang agar replace tidak terpotong
    for ent in sorted(ALL_DEST, key=len, reverse=True):
        pattern = r'\b' + re.escape(ent.lower()) + r'\b'
        sig = re.sub(pattern, "[DEST]", sig)
    # Hilangkan angka dan harga
    sig = re.sub(r'\b\d+\b', '[NUM]', sig)
    sig = re.sub(r'rp\s?\d+', '[PRICE]', sig)
    return sig.strip()

def group_split(items, get_text_func, ratios=(0.8, 0.1, 0.1)):
    """Membagi data berdasarkan kerangka template (Group Split)"""
    groups = defaultdict(list)
    for item in items:
        sig = get_signature(get_text_func(item))
        groups[sig].append(item)
    
    # Shuffle grup agar distribusinya acak
    group_list = list(groups.values())
    random.shuffle(group_list)
    
    n_groups = len(group_list)
    if n_groups < 3:
        print(f"⚠️  WARNING: Hanya {n_groups} grup template unik. Split tidak dijamin representatif.")
    
    train, val, test = [], [], []
    n_total = len(items)
    t_target = n_total * ratios[0]
    v_target = n_total * ratios[1]
    
    for g in group_list:
        if len(train) < t_target:
            train.extend(g)
        elif len(val) < v_target:
            val.extend(g)
        else:
            test.extend(g)
            
    # Jika test kosong karena grupnya sedikit, paksa pindahkan dari train
    if not test and len(train) > 1:
        test.extend(group_list[-1])
        train = [x for g in group_list[:-1] for x in g]
        group_list = group_list[:-1]
        
    # Jika val kosong
    if not val and len(train) > 1:
        val.extend(group_list[-1])
        train = [x for g in group_list[:-1] for x in g]
        
    return train, val, test

def verify_no_overlap(train_data, val_data, test_data, get_text_func, label=""):
    sig_train = set(get_signature(get_text_func(x)) for x in train_data)
    sig_val   = set(get_signature(get_text_func(x)) for x in val_data)
    sig_test  = set(get_signature(get_text_func(x)) for x in test_data)
    ott = sig_train & sig_test
    otv = sig_train & sig_val
    if ott or otv:
        print(f"⚠️  [{label}] Overlap train-test: {len(ott)}, train-val: {len(otv)}")
        if ott: print("   Contoh bocor:", list(ott)[:3])
    return len(ott) == 0 and len(otv) == 0

def split_intent_dataset():
    input_path = "ml/data/raw/intents_bilingual_v2.csv"
    print(f"Membaca {input_path} (Anti-Overfitting Mode)...")
    
    data_by_label = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_by_label[row['label']].append(row)
    
    train_data, val_data, test_data = [], [], []
    
    for label, items in sorted(data_by_label.items()):
        # Split per label berdasarkan signature
        tr, va, te = group_split(items, lambda x: x['text'])
        
        # VERIFIKASI TIDAK ADA KEBOCORAN (OVERLAP)
        is_clean = verify_no_overlap(tr, va, te, lambda x: x['text'], label=label)
        if not is_clean:
            print(f"❌ ERROR: Terdeteksi kebocoran data pada intent '{label}'. Proses dibatalkan.")
            return

        train_data.extend(tr)
        val_data.extend(va)
        test_data.extend(te)
        print(f"  {label}: {len(items)} → train={len(tr)}, val={len(va)}, test={len(te)}")
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
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
    print(f"\nTotal Intent: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")


def split_ner_dataset():
    input_path = "ml/data/raw/ner_dataset_v2.json"
    if not os.path.exists(input_path):
        return
        
    print(f"\nMembaca {input_path} (Anti-Overfitting Mode)...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tr, va, te = group_split(data, lambda x: " ".join(x['tokens']))
    
    # VERIFIKASI TIDAK ADA KEBOCORAN (OVERLAP)
    is_clean = verify_no_overlap(tr, va, te, lambda x: " ".join(x['tokens']), label="NER")
    if not is_clean:
        print(f"❌ ERROR: Terdeteksi kebocoran data pada NER. Proses dibatalkan.")
        return
        
    random.shuffle(tr)
    random.shuffle(va)
    random.shuffle(te)
    
    os.makedirs("ml/data/processed", exist_ok=True)
    for name, split in [("train_ner_v2.json", tr), 
                        ("val_ner_v2.json", va), 
                        ("test_ner_v2.json", te)]:
        path = f"ml/data/processed/{name}"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path} ({len(split)} samples)")
    print(f"Total NER: train={len(tr)}, val={len(va)}, test={len(te)}")


if __name__ == "__main__":
    split_intent_dataset()
    split_ner_dataset()

