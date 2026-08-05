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
# Expanded ALL_DEST & ALL_ENTITIES mask pool dari synonym pools
ALL_DEST = [
    "Benteng Kuto Besak", "Jembatan Ampera", "Pulau Kemaro", "Monpera",
    "Museum Sultan Mahmud Badaruddin II", "Kambang Iwak", "Hutan Wisata Punti Kayu",
    "Masjid Agung Palembang", "Kampung Kapitan", "Kampung Arab Al-Munawar",
    "Taman Kambang Iwak Besak", "Palembang Trade Center", "Palembang Icon",
    "Palembang Square", "Jakabaring Sport City", "Taman Kebon Rojo",
    "Museum Balaputra Dewa", "Al Quran Al Akbar", "Bukit Siguntang",
    "Masjid Cheng Ho", "Fantasy Island", "Amanzi Waterpark",
    "OPI Mall", "Pasar 16 Ilir", "Kawah Tengkurep",
    "Hotel Novotel Palembang", "Hotel Novotel", "Novotel", "Novotel PTC",
    "Hotel Aryaduta Palembang", "Hotel Aryaduta", "Aryaduta",
    "Hotel Excelton", "Excelton", "The Zuri Palembang", "The Zuri", "Zuri Palembang",
    "Pempek Candy", "Pempek Beringin", "Pempek Nony 168", "Pempek Nony",
    "Martabak HAR", "Mie Celor 26 Ilir", "Mie Celor Syafeiz", "Model H. Dowa",
    "Restoran River Side", "River Side", "Dermaga Point", "Dermaga Point BKB",
    "Rumah Adat Dekranasda", "Dekranasda", "Sanggar Tari Rumah Elok", "Rumah Elok",
    "Taman Purbakala Sriwijaya", "Museum A.K. Gani", "Museum AK Gani", "AK Gani",
    "Zainal Songket", "Griya Agung", "BKB", "SMB", "SMB II", "PTC", "PIM", "Ampera",
    "Kemaro", "Monpera", "Punti Kayu", "KI", "Al-Munawar", "Jakabaring", "Siguntang", "Amanzi"
]

def get_signature(text):
    """Menghapus filler words, entitas, dialek, dan prefix/suffix agar tersisa hanya kerangka template murni"""
    sig = text.lower()
    
    # 1. Normalisasi duplikasi kata beruntun (mis. "wisata wisata" -> "wisata")
    sig = re.sub(r'\b(\w+)\s+\1\b', r'\1', sig)

    # 2. Hapus filler words diperluas di manapun dalam kalimat (word boundary)
    FILLERS = [
        r'\bdong\b', r'\bdeh\b', r'\bnih\b', r'\bsih\b', r'\bkak\b', r'\bya\b', 
        r'\bmin\b', r'\bbang\b', r'\bhalo\b', r'\bpermisi\b', r'\bmau nanya\b', 
        r'\bmaaf mau tanya\b', r'\beh\b', r'\bplease\b', r'\bthanks\b', r'\bthank you\b',
        r'\btolong\b', r'\bcoba\b', r'\bkira-kira\b', r'\byuk\b', r'\bkk\b', r'\bka\b',
        r'\byaa\b', r'\bgaes\b', r'\bguys\b', r'\bbtw\b'
    ]
    for flr in FILLERS:
        sig = re.sub(flr, '', sig, flags=re.IGNORECASE)

    # 3. Normalisasi dialek
    ID_DIALECT_REVERSE = {
        r'\bapo\b': 'apa', r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak',
        r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana', r'\bnian\b': 'sangat',
        r'\bkau\b': 'kamu', r'\blemak\b': 'enak', r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga',
        r'\bberapo\b': 'berapa', r'\bngapo\b': 'kenapa', r'\bbae\b': 'saja', r'\byo\b': 'ya',
        r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
    }
    for pat, repl in ID_DIALECT_REVERSE.items():
        sig = re.sub(pat, repl, sig)

    # 4. Replace destinasi, angka, dan entitas lainnya
    for ent in sorted(ALL_DEST, key=len, reverse=True):
        pattern = r'\b' + re.escape(ent.lower()) + r'\b'
        sig = re.sub(pattern, "[DEST]", sig)
        
    # Expand to other entity types to prevent group split fragmentation
    PRICE_WORDS = ['murah', 'terjangkau', 'ekonomis', 'harga pas', 'budget friendly', 'merakyat', 
                   'gratis', 'free', 'tidak berbayar', 'tanpa biaya', 'bebas biaya', 'gratisan', 'cuma-cuma',
                   'mahal', 'premium', 'luxury', 'mehul', 'kuras kantong', 'harga', 'tarif', 'biaya', 'tiket', 'htm',
                   'budget', 'anggaran', 'modal', 'promo', 'diskon', 'potongan harga', 'hemat', 'ribu', 'rupiah', 'ribuan']
    TIME_WORDS = ['pagi', 'pagi hari', 'pagi-pagi', 'siang', 'siang hari', 'tengah hari', 'sore', 'sore hari', 'petang',
                  'malam', 'malam hari', 'malem', 'weekend', 'akhir pekan', 'sabtu minggu', 'sabtu-minggu', 
                  'hari ini', 'sekarang', 'saat ini', 'besok', 'besoknya', 'esok hari', 'libur', 'hari libur', 'tanggal merah',
                  'jam']
    CATEGORY_WORDS = ['wisata alam', 'destinasi alam', 'tempat alam', 'wisata sejarah', 'destinasi sejarah', 'tempat bersejarah',
                      'kuliner', 'makanan', 'tempat makan', 'restaurant', 'restoran', 'wisata keluarga', 'destinasi keluarga', 
                      'tempat keluarga', 'wisata budaya', 'destinasi budaya', 'tempat budaya', 'hotel', 'penginapan', 'akomodasi', 
                      'tempat menginap', 'cafe', 'kafe', 'coffee shop', 'ngopi', 'museum', 'masjid', 'taman', 'wahana air', 
                      'spot foto', 'pusat oleh-oleh']
                      
    for w in sorted(CATEGORY_WORDS, key=len, reverse=True):
        sig = re.sub(r'\b' + re.escape(w) + r'\b', '[CATEGORY]', sig)
    for w in sorted(TIME_WORDS, key=len, reverse=True):
        sig = re.sub(r'\b' + re.escape(w) + r'\b', '[TIME]', sig)
    for w in sorted(PRICE_WORDS, key=len, reverse=True):
        sig = re.sub(r'\b' + re.escape(w) + r'\b', '[PRICE]', sig)
        
    sig = re.sub(r'\b\d+\b', '[NUM]', sig)
    sig = re.sub(r'rp\s?\d+', '[PRICE]', sig)
    return re.sub(r'\s+', ' ', sig).strip()

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "raw", "intents_base_v2.csv")
    print(f"Membaca {input_path} (Strict Signature Group Split Mode)...")
    
    data_by_label = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_by_label[row['label']].append(row)
    
    train_base_data, val_data, test_data = [], [], []
    
    for label, items in sorted(data_by_label.items()):
        # Split per label berdasarkan signature (75% Train, 12.5% Val, 12.5% Test)
        tr, va, te = group_split(items, lambda x: x['text'], ratios=(0.75, 0.125, 0.125))
        
        # VERIFIKASI TIDAK ADA KEBOCORAN TEMPLATE
        is_clean = verify_no_overlap(tr, va, te, lambda x: x['text'], label=label)
        if not is_clean:
            print(f"❌ ERROR: Terdeteksi kebocoran data pada intent '{label}'. Proses dibatalkan.")
            return

        train_base_data.extend(tr)
        val_data.extend(va)
        test_data.extend(te)
        print(f"  {label}: {len(items)} → train_base={len(tr)}, val={len(va)}, test={len(te)}")

    # Augmentasi HANYA untuk split Train Base
    from generate_intent_dataset_v2 import augment_intent_train
    train_tuples = [(r['text'], r['label']) for r in train_base_data]
    augmented_tuples = augment_intent_train(train_tuples, target_per_class=500)
    train_augmented_data = [{'text': t, 'label': l} for t, l in augmented_tuples]

    # PASCA-DEDUP TEST SET TERHADAP TRAIN AUGMENTED (train_intents_v2.csv)
    def norm_words(text):
        t = text.lower()
        ID_DIALECT_REVERSE = {
            r'\bapo\b': 'apa', r'\bdak\b': 'tidak', r'\bgak\b': 'tidak', r'\benggak\b': 'tidak',
            r'\bcakmano\b': 'bagaimana', r'\bgimana\b': 'bagaimana', r'\bnian\b': 'sangat',
            r'\bkau\b': 'kamu', r'\blemak\b': 'enak', r'\bpacak\b': 'bisa', r'\bhargo\b': 'harga',
            r'\bberapo\b': 'berapa', r'\bngapo\b': 'kenapa', r'\bbae\b': 'saja', r'\byo\b': 'ya',
            r'\bkatek\b': 'tidak ada', r'\bpegi\b': 'pergi', r'\bjingok\b': 'lihat'
        }
        for pat, repl in ID_DIALECT_REVERSE.items():
            t = re.sub(pat, repl, t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return set(t.split())

    def jaccard(set1, set2):
        if not set1 or not set2: return 0.0
        return len(set1 & set2) / float(len(set1 | set2))

    train_aug_exact = set(r['text'].strip().lower() for r in train_augmented_data)
    train_aug_wordsets = [norm_words(r['text']) for r in train_augmented_data]

    filtered_test_data = []
    dropped_test_count = 0
    for r in test_data:
        txt = r['text'].strip()
        if txt.lower() in train_aug_exact:
            dropped_test_count += 1
            continue
        t_words = norm_words(txt)
        is_near_dup = False
        for tr_words in train_aug_wordsets:
            if jaccard(t_words, tr_words) > 0.85:
                is_near_dup = True
                break
        if is_near_dup:
            dropped_test_count += 1
        else:
            filtered_test_data.append(r)
            
    test_data = filtered_test_data
    print(f"  Pasca-Dedup Intent Test vs Augmented Train: Dibuang {dropped_test_count} sampel test.")
    
    out_dir = os.path.join(script_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [("train_intents_v2.csv", train_augmented_data),
                       ("train_intents_raw_v2.csv", train_base_data), 
                       ("val_intents_v2.csv", val_data), 
                       ("test_intents_v2.csv", test_data)]:
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label'])
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {path} ({len(data)} rows)")
    print(f"\nTotal Intent: train_augmented={len(train_augmented_data)}, train_raw={len(train_base_data)}, val={len(val_data)}, test={len(test_data)}")


def split_ner_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "raw", "ner_dataset_v2.json")
    if not os.path.exists(input_path):
        return
        
    print(f"\nMembaca {input_path} (Anti-Overfitting + Entity Holdout 20% Mode)...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Ekstrak semua entity value unik per entitas
    entities_by_type = defaultdict(set)
    for sample in data:
        tokens = sample['tokens']
        tags = sample['tags']
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

    # 2. Holdout Entities (20% untuk DESTINATION & LOCATION sebagai open-class)
    # Untuk PRICE, TIME, CATEGORY, jika di-holdout, buang superset dari train agar same-type leak = 0
    rng = random.Random(42)
    holdout_by_type = {}
    
    for ent_type, val_set in sorted(entities_by_type.items()):
        val_list = sorted(list(val_set))
        rng.shuffle(val_list)
        # Holdout open-class (DESTINATION, LOCATION) 20%, closed-class (PRICE, TIME, CATEGORY) 15%
        rate = 0.20 if ent_type in ['DESTINATION', 'LOCATION'] else 0.15
        n_hold = max(1, int(len(val_list) * rate))
        hold_set = set(val_list[:n_hold])
        holdout_by_type[ent_type] = hold_set
        print(f"  Holdout {ent_type:12s}: {len(hold_set)}/{len(val_list)} values ({len(hold_set)/len(val_list)*100:.1f}%) -> {sorted(list(hold_set))[:2]}...")

    # Purge superset di train_entities untuk closed-class leak prevention
    # Jika h_val adalah substring dari ent_val bertipe SAMA, hapus h_val dari holdout set
    for ent_type, h_set in list(holdout_by_type.items()):
        all_vals = entities_by_type[ent_type]
        purged_h_set = set()
        for h in h_set:
            # Check if h is strict substring of another non-holdout value of the SAME TYPE
            other_vals = all_vals - h_set
            if any(h != o and h in o for o in other_vals):
                continue # Purge from holdout to avoid same-type leak
            purged_h_set.add(h)
        holdout_by_type[ent_type] = purged_h_set

    # Simpan holdout_entities ke JSON agar augment_ner.py tidak memakai holdout ini saat augmentasi
    holdout_path = os.path.join(script_dir, "processed", "ner_holdout_entities.json")
    with open(holdout_path, 'w', encoding='utf-8') as f:
        json.dump({k: list(v) for k, v in holdout_by_type.items()}, f, ensure_ascii=False, indent=2)

    # 3. Pisahkan sampel yang mengandung Holdout Entities (Global Check across all types)
    all_holdout_values_global = set(v.lower() for v_list in holdout_by_type.values() for v in v_list)
    all_holdout_samples = []
    train_val_pool = []
    
    for sample in data:
        tokens = sample['tokens']
        tags = sample['tags']
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
                if val in all_holdout_values_global:
                    has_holdout = True
                    break
                i = j
            else:
                i += 1
                
        if has_holdout:
            all_holdout_samples.append(sample)
        else:
            train_val_pool.append(sample)

    rng.shuffle(all_holdout_samples)
    # Cap holdout test set to max 500 samples (Zero-Shot Test Set)
    test_ner_holdout = all_holdout_samples[:500]
    # OPTION A: Drop all surplus holdout samples (do NOT add to train_val_pool) to prevent collision & leakage
    print(f"  [OPTION A] Dropping {len(all_holdout_samples) - len(test_ner_holdout)} surplus holdout samples.")

    # 4. Group Split train_val_pool: 80% Train, 10% Val, 10% Test Seen
    tr, va, te_seen = group_split(train_val_pool, lambda x: " ".join(x['tokens']), ratios=(0.80, 0.10, 0.10))
    te_combined = te_seen + test_ner_holdout

    random.shuffle(tr)
    random.shuffle(va)
    random.shuffle(te_seen)
    random.shuffle(test_ner_holdout)
    random.shuffle(te_combined)
    
    out_dir = os.path.join(script_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    for name, split in [("train_ner_v2.json", tr), 
                        ("val_ner_v2.json", va), 
                        ("test_ner_seen.json", te_seen),
                        ("test_ner_holdout.json", test_ner_holdout),
                        ("test_ner_v2.json", te_combined)]:
        path = os.path.join(out_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path} ({len(split)} samples)")
    print(f"Total NER: train={len(tr)}, val={len(va)}, test_seen={len(te_seen)}, test_holdout={len(test_ner_holdout)}, combined_test={len(te_combined)}")


if __name__ == "__main__":
    split_intent_dataset()
    split_ner_dataset()

