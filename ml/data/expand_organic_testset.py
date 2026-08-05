import os
import csv
import json
import re
from collections import Counter, defaultdict

# -------------------------------------------------------------------
# 1. LOAD DATA TRAINING & EXISTING ORGANIC DATA
# -------------------------------------------------------------------
train_intent_file = "ml/data/processed/train_intents_v2.csv"
train_ner_file    = "ml/data/processed/train_ner_v2.json"
train_ner_aug_file= "ml/data/processed/train_ner_augmented_v2.json"
holdout_file      = "ml/data/processed/ner_holdout_entities.json"

intent_organik_file = "ml/data/processed/test_intents_organik.csv"
ner_organik_file    = "ml/data/processed/test_ner_organik.json"

train_intent_texts = set()
with open(train_intent_file, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        train_intent_texts.add(r['text'].strip().lower())

tr_ner = json.load(open(train_ner_file, encoding='utf-8'))
tr_ner_aug = json.load(open(train_ner_aug_file, encoding='utf-8'))
holdouts = json.load(open(holdout_file, encoding='utf-8'))

train_ner_texts = set(' '.join(s['tokens']).strip().lower() for s in tr_ner + tr_ner_aug)

train_entities = set()
for s in tr_ner + tr_ner_aug:
    tokens = s['tokens']
    tags = s['tags']
    curr = []
    for t, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if curr: train_entities.add(' '.join(curr).strip().lower())
            curr = [t]
        elif tag.startswith('I-') and curr:
            curr.append(t)
        else:
            if curr: train_entities.add(' '.join(curr).strip().lower())
            curr = []
    if curr:
        train_entities.add(' '.join(curr).strip().lower())

for lst in holdouts.values():
    for v in lst:
        train_entities.add(v.strip().lower())

# Read existing organic data
existing_intents = list(csv.DictReader(open(intent_organik_file, encoding='utf-8')))
existing_intent_texts = set(r['text'].strip().lower() for r in existing_intents)

existing_ner = json.load(open(ner_organik_file, encoding='utf-8'))
existing_ner_texts = set(' '.join(s['tokens']).strip().lower() for s in existing_ner)

# -------------------------------------------------------------------
# 2. DEFINISI SAMPEL INTENT BARU (INFORMAL, DIALEK PALEMBANG, TYPO)
# -------------------------------------------------------------------
new_intent_samples = [
    # ask_category (3 baru)
    {"text": "min ado tempat wisata sejarah apo bae yo di plg?", "label": "ask_category"},
    {"text": "spotted spot instagramable yg lg hits dong kka", "label": "ask_category"},
    {"text": "rekomen sentra kerajinan songket yg bagus dmano", "label": "ask_category"},

    # ask_destination_info (3 baru)
    {"text": "infokan seputar Danau OPI dunk bang", "label": "ask_destination_info"},
    {"text": "jingok foto Kampung Gelam cakmano elok idak?", "label": "ask_destination_info"},
    {"text": "spot Taman Sekanak Molek tu isinyo apo bae sih", "label": "ask_destination_info"},

    # ask_facilities (3 baru)
    {"text": "di Jembatan Musi IV ado parkiran motor dak min", "label": "ask_facilities"},
    {"text": "Rumah Syahab ado tempat sholat samo toilet dk?", "label": "ask_facilities"},
    {"text": "fasilitas kursi ama resto apung dsekitar situ lengkap dk", "label": "ask_facilities"},

    # ask_hidden_gems (3 baru)
    {"text": "spill tempat nugas sepi yg katek uwang tau di kenten dong", "label": "ask_hidden_gems"},
    {"text": "wisata sejarah malam yg belom viral dmano yo", "label": "ask_hidden_gems"},
    {"text": "kalo nak cari tempat santai pinggir sungai yg sepi kmano", "label": "ask_hidden_gems"},

    # ask_location_access (3 baru)
    {"text": "rute naik angkot ke Kelenteng Chandra Nadi drmana yo", "label": "ask_location_access"},
    {"text": "pake gmaps nak ke Masjid Al-Mahdi lewat jalan mano bang", "label": "ask_location_access"},
    {"text": "lokasi Dermaga Keramasan tu daerah mana sih min", "label": "ask_location_access"},

    # ask_lrt_destinations (3 baru)
    {"text": "naik stasiun lrt mana yg deket ama Pulau Borang", "label": "ask_lrt_destinations"},
    {"text": "kalo mo ke Kawasan Sekanak Besolek pake lrt bisa gak", "label": "ask_lrt_destinations"},
    {"text": "setop lrt terdekat dr Monumen Dwikora tuh dmana ya", "label": "ask_lrt_destinations"},

    # ask_operating_hours (3 baru)
    {"text": "jm brp biasanya spot foto Rumah Syahab buka bang", "label": "ask_operating_hours"},
    {"text": "toko oleh-oleh khas plg buka jam brapa pas malam minggu", "label": "ask_operating_hours"},
    {"text": "Dermaga Keramasan sampe jam 3 sore masih buka dk", "label": "ask_operating_hours"},

    # ask_recommendation (3 baru)
    {"text": "rekomen tempat wisata keluarga murah meriah dsekitar plg", "label": "ask_recommendation"},
    {"text": "kalo nak nongkrong santai senja hari mending kmano ya", "label": "ask_recommendation"},
    {"text": "tolong spil tempat makan makanan tradisional khas plg bwt besok", "label": "ask_recommendation"},

    # ask_ticket_price (3 baru)
    {"text": "tikt msk ke Danau OPI brp duit skrg min", "label": "ask_ticket_price"},
    {"text": "hargo mlebu Kampung Gelam mahal dk yo per org", "label": "ask_ticket_price"},
    {"text": "masuk ke Kelenteng Chandra Nadi keno biaya brp", "label": "ask_ticket_price"},

    # ask_unrelated (3 baru)
    {"text": "kito nak meli baju kaos dmano yg murah nian", "label": "ask_unrelated"},
    {"text": "skor bola sriwijaya fc td malam brp yo mang", "label": "ask_unrelated"},
    {"text": "cuaca palembang lg ujan terus nih males keluar", "label": "ask_unrelated"},

    # goodbye (3 baru)
    {"text": "makasih infonyo yo mang, kito pamit dulu", "label": "goodbye"},
    {"text": "mksih bnyk kka, sy mau lanjut jalan ya bye", "label": "goodbye"},
    {"text": "ok sip tengkiu min, smoga harimu menyenangkan", "label": "goodbye"},

    # greet (3 baru)
    {"text": "malam min, lg online dk nih?", "label": "greet"},
    {"text": "halo kka bot, biso bantu aku dk", "label": "greet"},
    {"text": "spadaa minn, mau nanya2 dunk", "label": "greet"},

    # provide_feedback (3 baru)
    {"text": "jawaban bot ny mantap nian, fast respon pulok!", "label": "provide_feedback"},
    {"text": "infonya agak rancu min, coba perbaiki lg ya", "label": "provide_feedback"},
    {"text": "suka bgt sm pelayanan chatbot nyo, josss", "label": "provide_feedback"},
]

# -------------------------------------------------------------------
# 3. DEFINISI SAMPEL NER BARU (NOVEL ENTITIES, SPACE-TOKENIZED)
# -------------------------------------------------------------------
new_ner_samples = [
    # PRICE Focus (10+ new mentions)
    {"tokens": ["tiket", "masuk", "Danau", "OPI", "cuma", "7.500", "per", "orang"],
     "tags": ["O", "O", "B-DESTINATION", "I-DESTINATION", "O", "B-PRICE", "O", "O"]},
    
    {"tokens": ["biaya", "parkir", "di", "Dermaga", "Keramasan", "sekitar", "12", "ribu"],
     "tags": ["O", "O", "O", "B-DESTINATION", "I-DESTINATION", "O", "B-PRICE", "I-PRICE"]},
    
    {"tokens": ["mlebu", "Kampung", "Gelam", "keno", "Rp8.000", "bae"],
     "tags": ["O", "B-DESTINATION", "I-DESTINATION", "O", "B-PRICE", "O"]},
    
    {"tokens": ["karcis", "Rumah", "Syahab", "tigo", "puluh", "ribu", "mang"],
     "tags": ["O", "B-DESTINATION", "I-DESTINATION", "B-PRICE", "I-PRICE", "I-PRICE", "O"]},
    
    {"tokens": ["masuk", "Masjid", "Al-Mahdi", "tarifnyo", "8rb"],
     "tags": ["O", "B-DESTINATION", "I-DESTINATION", "O", "B-PRICE"]},
    
    {"tokens": ["tiket", "wahana", "cuma", "15.000", "rupiah"],
     "tags": ["O", "O", "O", "B-PRICE", "I-PRICE"]},
    
    {"tokens": ["masuk", "Kelenteng", "Chandra", "Nadi", "cuma", "segawe"],
     "tags": ["O", "B-DESTINATION", "I-DESTINATION", "I-DESTINATION", "O", "B-PRICE"]},
    
    {"tokens": ["hargo", "makanan", "sekitar", "Rp", "12.000", "di", "Sungai", "Lais"],
     "tags": ["O", "O", "O", "B-PRICE", "I-PRICE", "O", "B-LOCATION", "I-LOCATION"]},
    
    {"tokens": ["sewa", "perahu", "ke", "Pulau", "Borang", "keno", "35rb"],
     "tags": ["O", "O", "O", "B-DESTINATION", "I-DESTINATION", "O", "B-PRICE"]},
    
    {"tokens": ["karcis", "masuk", "Taman", "Sekanak", "Molek", "75", "ribu"],
     "tags": ["O", "O", "B-DESTINATION", "I-DESTINATION", "I-DESTINATION", "B-PRICE", "I-PRICE"]},
    
    {"tokens": ["bayar", "masuk", "cuma", "Rp", "7.500", "aja"],
     "tags": ["O", "O", "O", "B-PRICE", "I-PRICE", "O"]},

    # LOCATION Focus (10+ new mentions)
    {"tokens": ["tempat", "makan", "enak", "di", "daerah", "Sungai", "Lais", "dmano"],
     "tags": ["O", "O", "O", "O", "O", "B-LOCATION", "I-LOCATION", "O"]},
    
    {"tokens": ["ada", "tempat", "ngopi", "kekinian", "di", "Sako", "gak", "min"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "O", "O"]},
    
    {"tokens": ["cari", "oleh-oleh", "khas", "di", "Sukarami"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-LOCATION"]},
    
    {"tokens": ["lokasi", "toko", "suvenir", "di", "Kalidoni", "sebelah", "mana"],
     "tags": ["O", "O", "O", "O", "B-LOCATION", "O", "O"]},
    
    {"tokens": ["Rumah", "makan", "lesehan", "sekitar", "Seberang", "Ulu", "II", "yg", "buka", "subuh"],
     "tags": ["B-CATEGORY", "I-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "I-LOCATION", "I-LOCATION", "O", "O", "B-TIME"]},
    
    {"tokens": ["deket", "Simpang", "Polda", "ada", "tempat", "nugas", "dk"],
     "tags": ["O", "B-LOCATION", "I-LOCATION", "O", "B-CATEGORY", "I-CATEGORY", "O"]},
    
    {"tokens": ["daerah", "Tangga", "Takat", "banyak", "makanan", "tradisional"],
     "tags": ["O", "B-LOCATION", "I-LOCATION", "O", "B-CATEGORY", "I-CATEGORY"]},
    
    {"tokens": ["di", "Simpang", "Charitas", "macet", "pas", "jam", "3", "sore"],
     "tags": ["O", "B-LOCATION", "I-LOCATION", "O", "O", "B-TIME", "I-TIME", "I-TIME"]},
    
    {"tokens": ["spot", "nongkrong", "di", "Bumi", "Srijaya", "rame", "nian"],
     "tags": ["O", "O", "O", "B-LOCATION", "I-LOCATION", "O", "O"]},
    
    {"tokens": ["cari", "taman", "bermain", "di", "Lebung", "Gajah"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "I-LOCATION"]},
    
    {"tokens": ["daerah", "Talang", "Kelapa", "banyak", "kedai", "kopi", "senja"],
     "tags": ["O", "B-LOCATION", "I-LOCATION", "O", "B-CATEGORY", "I-CATEGORY", "I-CATEGORY"]},
    
    {"tokens": ["toko", "kue", "di", "Duku", "buka", "jam", "11", "siang"],
     "tags": ["O", "O", "O", "B-LOCATION", "O", "B-TIME", "I-TIME", "I-TIME"]},

    # CATEGORY Focus (10+ new mentions)
    {"tokens": ["cari", "spot", "instagramable", "di", "Jembatan", "Musi", "VI"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-DESTINATION", "I-DESTINATION", "I-DESTINATION"]},
    
    {"tokens": ["rekomen", "wisata", "sejarah", "malam", "di", "Tanjung", "Siapi-api"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "I-LOCATION"]},
    
    {"tokens": ["dimana", "pusat", "oleh-oleh", "khas", "kain", "songket"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "I-CATEGORY", "O", "O"]},
    
    {"tokens": ["butuh", "tempat", "santai", "dekat", "Jembatan", "Musi", "IV"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-DESTINATION", "I-DESTINATION", "I-DESTINATION"]},
    
    {"tokens": ["beli", "makanan", "tradisional", "di", "Kawasan", "Pasar", "26"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "I-LOCATION", "I-LOCATION"]},
    
    {"tokens": ["spot", "wisata", "bahari", "di", "Pulau", "Borang"],
     "tags": ["O", "B-CATEGORY", "I-CATEGORY", "O", "B-DESTINATION", "I-DESTINATION"]},
    
    {"tokens": ["sentra", "kerajinan", "songket", "di", "Seberang", "Ulu", "II"],
     "tags": ["B-CATEGORY", "I-CATEGORY", "I-CATEGORY", "O", "B-LOCATION", "I-LOCATION", "I-LOCATION"]},
    
    {"tokens": ["makan", "di", "restoran", "apung", "tarif", "75", "ribu"],
     "tags": ["O", "O", "B-CATEGORY", "I-CATEGORY", "O", "B-PRICE", "I-PRICE"]},
    
    {"tokens": ["taman", "bermain", "anak", "di", "Sukarami", "buka", "subuh"],
     "tags": ["B-CATEGORY", "I-CATEGORY", "O", "O", "B-LOCATION", "O", "B-TIME"]},
    
    {"tokens": ["tempat", "nugas", "semalaman", "penuh", "di", "Kalidoni"],
     "tags": ["B-CATEGORY", "I-CATEGORY", "B-TIME", "I-TIME", "O", "B-LOCATION"]},

    # Additional DESTINATION & TIME combinations
    {"tokens": ["Monumen", "Dwikora", "buka", "pas", "malam", "minggu"],
     "tags": ["B-DESTINATION", "I-DESTINATION", "O", "O", "B-TIME", "I-TIME"]},
    
    {"tokens": ["foto", "di", "Kampung", "Gelam", "pas", "senja", "hari"],
     "tags": ["O", "O", "B-DESTINATION", "I-DESTINATION", "O", "B-TIME", "I-TIME"]},
    
    {"tokens": ["ke", "Kelenteng", "Chandra", "Nadi", "pas", "isya", "nanti"],
     "tags": ["O", "B-DESTINATION", "I-DESTINATION", "I-DESTINATION", "O", "B-TIME", "I-TIME"]},
    
    {"tokens": ["rute", "ke", "Masjid", "Al-Mahdi", "bisa", "jam", "7", "malam"],
     "tags": ["O", "O", "B-DESTINATION", "I-DESTINATION", "O", "B-TIME", "I-TIME", "I-TIME"]},
    
    {"tokens": ["Taman", "Sekanak", "Molek", "buka", "jam", "5", "subuh"],
     "tags": ["B-DESTINATION", "I-DESTINATION", "I-DESTINATION", "O", "B-TIME", "I-TIME", "I-TIME"]}
]

# -------------------------------------------------------------------
# 4. VERIFIKASI KEBARUAN & ATURAN ASSERTION
# -------------------------------------------------------------------
print("=== VERIFIKASI KEBARUAN ENTITAS (NOVELTY REPORT) ===")
checked_entities = []
all_new_entities = []

for item in new_ner_samples:
    tokens = item['tokens']
    tags = item['tags']
    curr = []
    curr_tag = None
    for t, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if curr:
                phrase = ' '.join(curr).strip()
                all_new_entities.append((phrase, curr_tag))
            curr = [t]
            curr_tag = tag[2:]
        elif tag.startswith('I-') and curr_tag == tag[2:]:
            curr.append(t)
        else:
            if curr:
                phrase = ' '.join(curr).strip()
                all_new_entities.append((phrase, curr_tag))
            curr = []
            curr_tag = None
    if curr:
        phrase = ' '.join(curr).strip()
        all_new_entities.append((phrase, curr_tag))

entity_collision_count = 0
for phrase, etype in all_new_entities:
    p_low = phrase.lower()
    is_novel = p_low not in train_entities
    status = "NOVEL" if is_novel else "BERTABRAKAN"
    if not is_novel:
        entity_collision_count += 1
    checked_entities.append((phrase, etype, status))

for phrase, etype, status in checked_entities:
    print(f"  [{etype:11s}] \"{phrase}\" -> Status: {status}")

print(f"\nTotal Entitas Baru Dicek: {len(checked_entities)}")
print(f"Total Bertabrakan dgn Data Latih: {entity_collision_count}")
assert entity_collision_count == 0, f"ASSERTION FAILED: {entity_collision_count} entitas bertabrakan dengan data latih!"

# -------------------------------------------------------------------
# 5. GABUNGKAN & SIMPAN FILE DATA BARU
# -------------------------------------------------------------------
final_intents = existing_intents + new_intent_samples
final_ner = existing_ner + new_ner_samples

with open(intent_organik_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['text', 'label'])
    writer.writeheader()
    writer.writerows(final_intents)

with open(ner_organik_file, 'w', encoding='utf-8') as f:
    json.dump(final_ner, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------------
# 6. HASIL VALIDASI MENTAH (RAW PRINTOUT ONLY)
# -------------------------------------------------------------------
VALID_INTENTS = {
    "ask_category", "ask_destination_info", "ask_facilities", "ask_hidden_gems",
    "ask_location_access", "ask_lrt_destinations", "ask_operating_hours", "ask_recommendation",
    "ask_ticket_price", "ask_unrelated", "goodbye", "greet", "provide_feedback"
}

VALID_TAGS = {
    "O", "B-DESTINATION", "I-DESTINATION", "B-CATEGORY", "I-CATEGORY",
    "B-LOCATION", "I-LOCATION", "B-TIME", "I-TIME", "B-PRICE", "I-PRICE"
}

print("\n============================================================")
print("             HASIL VALIDASI MENTAH INTENT CSV               ")
print("============================================================")
print(f"File                  : {intent_organik_file}")
print(f"Jumlah Total Baris    : {len(final_intents)}")

intent_counts = Counter(r['label'] for r in final_intents)
print("\nJumlah Sampel per Kelas:")
for lbl in sorted(VALID_INTENTS):
    cnt = intent_counts[lbl]
    print(f"  {lbl:22s}: {cnt:2d} sampel")
    assert cnt >= 5, f"ASSERTION FAILED: Label '{lbl}' kurang dari 5 sampel ({cnt})"

# Assert no duplicate text
intent_texts = [r['text'].strip().lower() for r in final_intents]
assert len(intent_texts) == len(set(intent_texts)), "ASSERTION FAILED: Ada text duplikat di test_intents_organik.csv!"

# Assert no text in train_intents_v2.csv
train_intent_collisions = set(intent_texts) & train_intent_texts
assert len(train_intent_collisions) == 0, f"ASSERTION FAILED: {len(train_intent_collisions)} text sama dengan train!"
print("Assert Intent Validations Passed: Semua label valid, no duplicates, 0 leak vs train_intents_v2.csv!")


print("\n============================================================")
print("              HASIL VALIDASI MENTAH NER JSON                ")
print("============================================================")
print(f"File                  : {ner_organik_file}")
print(f"Jumlah Total Kalimat  : {len(final_ner)}")

entity_counts = Counter()
ner_texts = []

for idx, s in enumerate(final_ner):
    tokens = s['tokens']
    tags = s['tags']
    text_str = ' '.join(tokens).strip().lower()
    ner_texts.append(text_str)

    # Assert len(tokens) == len(tags)
    assert len(tokens) == len(tags), f"ASSERTION FAILED: Sentence #{idx} len(tokens) != len(tags) ({len(tokens)} vs {len(tags)})"

    # Assert valid tags and BIO consistency
    prev_tag = "O"
    for j, tag in enumerate(tags):
        assert tag in VALID_TAGS, f"ASSERTION FAILED: Tag '{tag}' tidak valid di kalimat #{idx}"
        if tag.startswith("I-"):
            expected_b = "B-" + tag[2:]
            expected_i = "I-" + tag[2:]
            assert prev_tag in (expected_b, expected_i), f"ASSERTION FAILED: Invalid BIO sequence '{prev_tag}' -> '{tag}' di kalimat #{idx}"
        prev_tag = tag

        if tag.startswith("B-"):
            entity_counts[tag[2:]] += 1

print("\nJumlah Kemunculan Entitas per Tipe:")
for etype in ["DESTINATION", "LOCATION", "PRICE", "TIME", "CATEGORY"]:
    cnt = entity_counts[etype]
    print(f"  {etype:12s}: {cnt:2d} entitas")

assert entity_counts["PRICE"] >= 10, f"ASSERTION FAILED: PRICE < 10 ({entity_counts['PRICE']})"
assert entity_counts["LOCATION"] >= 10, f"ASSERTION FAILED: LOCATION < 10 ({entity_counts['LOCATION']})"
assert entity_counts["CATEGORY"] >= 10, f"ASSERTION FAILED: CATEGORY < 10 ({entity_counts['CATEGORY']})"

# Assert no duplicates & no train overlap
assert len(ner_texts) == len(set(ner_texts)), "ASSERTION FAILED: Ada kalimat duplikat di test_ner_organik.json!"
ner_train_collisions = set(ner_texts) & train_ner_texts
assert len(ner_train_collisions) == 0, f"ASSERTION FAILED: {len(ner_train_collisions)} kalimat NER sama dengan train!"

print("Assert NER Validations Passed: len(tokens)==len(tags), BIO valid, weak types >=10, 0 train duplicates!")
