"""
Generator NER Dataset v2 — Tambah singkatan + PRICE + bilingual
Memperluas dataset NER yang sudah ada dengan:
1. Singkatan destinasi (BKB, SMB, PTC, dll) yang di-tag BIO
2. Lebih banyak contoh PRICE entity
3. Contoh bilingual EN
"""
import json
import random
import os

random.seed(42)

# Singkatan → nama lengkap (untuk variasi)
ABBREVIATIONS = {
    "BKB": "Benteng Kuto Besak",
    "SMB": "Sultan Mahmud Badaruddin",
    "SMB II": "Sultan Mahmud Badaruddin II",
    "PTC": "Palembang Trade Center",
    "PIM": "Palembang Icon",
    "KI": "Kambang Iwak",
}

DESTINATIONS = [
    ("Benteng Kuto Besak", ["Benteng", "Kuto", "Besak"]),
    ("Jembatan Ampera", ["Jembatan", "Ampera"]),
    ("Pulau Kemaro", ["Pulau", "Kemaro"]),
    ("Monpera", ["Monpera"]),
    ("Kambang Iwak", ["Kambang", "Iwak"]),
    ("Hutan Wisata Punti Kayu", ["Hutan", "Wisata", "Punti", "Kayu"]),
    ("Masjid Agung Palembang", ["Masjid", "Agung", "Palembang"]),
    ("Kampung Kapitan", ["Kampung", "Kapitan"]),
    ("Kampung Arab Al-Munawar", ["Kampung", "Arab", "Al-Munawar"]),
    ("Jakabaring Sport City", ["Jakabaring", "Sport", "City"]),
    ("Museum Balaputra Dewa", ["Museum", "Balaputra", "Dewa"]),
    ("Al Quran Al Akbar", ["Al", "Quran", "Al", "Akbar"]),
    ("Bukit Siguntang", ["Bukit", "Siguntang"]),
    ("Masjid Cheng Ho", ["Masjid", "Cheng", "Ho"]),
    ("Fantasy Island", ["Fantasy", "Island"]),
    ("Amanzi Waterpark", ["Amanzi", "Waterpark"]),
    ("Kawah Tengkurep", ["Kawah", "Tengkurep"]),
]

ABBR_TOKENS = {
    "BKB": ["BKB"],
    "SMB": ["SMB"],
    "SMB II": ["SMB", "II"],
    "PTC": ["PTC"],
    "PIM": ["PIM"],
    "KI": ["KI"],
    "Ampera": ["Ampera"],
    "Kemaro": ["Kemaro"],
    "Monpera": ["Monpera"],
    "Punti Kayu": ["Punti", "Kayu"],
    "Jakabaring": ["Jakabaring"],
    "Siguntang": ["Siguntang"],
    "Amanzi": ["Amanzi"],
}

CATEGORIES = {
    "wisata alam": ["wisata", "alam"],
    "wisata sejarah": ["wisata", "sejarah"],
    "wisata kuliner": ["wisata", "kuliner"],
    "wisata religi": ["wisata", "religi"],
    "wisata budaya": ["wisata", "budaya"],
    "taman": ["taman"],
    "museum": ["museum"],
}

PRICES = [
    (["Rp", "10.000"], "Rp 10.000"),
    (["Rp", "15.000"], "Rp 15.000"),
    (["Rp", "20.000"], "Rp 20.000"),
    (["Rp", "25.000"], "Rp 25.000"),
    (["Rp", "5.000"], "Rp 5.000"),
    (["Rp", "50.000"], "Rp 50.000"),
    (["gratis"], "gratis"),
    (["10", "ribu"], "10 ribu"),
    (["20", "ribu"], "20 ribu"),
    (["lima", "ribu"], "lima ribu"),
    (["50", "ribuan"], "50 ribuan"),
    (["sepuluh", "ribu"], "sepuluh ribu"),
    (["dua", "puluh", "ribu"], "dua puluh ribu"),
    (["30", "ribu"], "30 ribu"),
    (["15", "ribuan"], "15 ribuan"),
    (["100", "ribu"], "100 ribu"),
    (["seratus", "ribu"], "seratus ribu"),
]

TIMES = [
    (["pagi"], "pagi"),
    (["siang"], "siang"),
    (["sore"], "sore"),
    (["malam"], "malam"),
    (["jam", "8"], "jam 8"),
    (["jam", "9", "pagi"], "jam 9 pagi"),
    (["jam", "10"], "jam 10"),
    (["hari", "Minggu"], "hari Minggu"),
    (["hari", "Sabtu"], "hari Sabtu"),
    (["weekend"], "weekend"),
    (["24", "jam"], "24 jam"),
]

def make_bio_tags(tokens, entity_ranges):
    """Buat BIO tags dari daftar token dan entity ranges"""
    tags = ["O"] * len(tokens)
    for start, end, label in entity_ranges:
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return tags


def generate_price_sentences():
    """Generate kalimat dengan entity PRICE yang beragam"""
    samples = []
    
    for dest_name, dest_tokens in DESTINATIONS + [(k, v) for k, v in ABBR_TOKENS.items()]:
        for price_tokens, _ in random.sample(PRICES, min(5, len(PRICES))):
            # "Harga tiket {dest} {price}"
            tokens = ["Harga", "tiket"] + dest_tokens + price_tokens
            entities = [
                (2, 2 + len(dest_tokens), "DESTINATION"),
                (2 + len(dest_tokens), 2 + len(dest_tokens) + len(price_tokens), "PRICE"),
            ]
            samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
            
            # "Berapa harga tiket {dest}?"
            tokens2 = ["Berapa", "harga", "tiket"] + dest_tokens + ["?"]
            entities2 = [(3, 3 + len(dest_tokens), "DESTINATION")]
            samples.append({"tokens": tokens2, "tags": make_bio_tags(tokens2, entities2)})
            
            # "Tiket masuk {dest} {price}"
            tokens3 = ["Tiket", "masuk"] + dest_tokens + price_tokens
            entities3 = [
                (2, 2 + len(dest_tokens), "DESTINATION"),
                (2 + len(dest_tokens), 2 + len(dest_tokens) + len(price_tokens), "PRICE"),
            ]
            samples.append({"tokens": tokens3, "tags": make_bio_tags(tokens3, entities3)})
    
    # Kalimat harga tanpa destinasi
    for price_tokens, _ in PRICES:
        tokens = ["Berapa", "harganya", "?"]
        samples.append({"tokens": tokens, "tags": ["O"] * len(tokens)})
        
        tokens2 = ["Harganya"] + price_tokens
        entities2 = [(1, 1 + len(price_tokens), "PRICE")]
        samples.append({"tokens": tokens2, "tags": make_bio_tags(tokens2, entities2)})
    
    return samples


def generate_abbreviation_sentences():
    """Generate kalimat dengan singkatan destinasi"""
    samples = []
    
    templates = [
        (["Dimana", "lokasi"], ["?"]),
        (["Jam", "buka"], ["?"]),
        (["Berapa", "tiket"], ["?"]),
        (["Info", "tentang"], []),
        (["Fasilitas"], ["apa", "aja", "?"]),
        (["Ceritakan", "tentang"], []),
        (["Mau", "ke"], []),
        (["Cara", "ke"], ["gimana", "?"]),
        (["Alamat"], ["dimana", "?"]),
    ]
    
    for abbr, abbr_tokens in ABBR_TOKENS.items():
        for prefix, suffix in templates:
            tokens = prefix + abbr_tokens + suffix
            dest_start = len(prefix)
            dest_end = dest_start + len(abbr_tokens)
            entities = [(dest_start, dest_end, "DESTINATION")]
            samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
    
    return samples


def generate_category_sentences():
    """Generate kalimat dengan entity CATEGORY"""
    samples = []
    
    templates_prefix = [
        ["Ada"],
        ["Rekomendasi"],
        ["Wisata"],
        ["Tempat"],
        ["Mau", "cari"],
        ["Cari"],
    ]
    
    templates_suffix = [
        ["di", "Palembang", "?"],
        ["Palembang", "?"],
        ["dong"],
        ["yang", "bagus", "?"],
    ]
    
    for cat_name, cat_tokens in CATEGORIES.items():
        for prefix in templates_prefix:
            for suffix in random.sample(templates_suffix, 2):
                tokens = prefix + cat_tokens + suffix
                cat_start = len(prefix)
                cat_end = cat_start + len(cat_tokens)
                entities = [(cat_start, cat_end, "CATEGORY")]
                samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
    
    return samples


def generate_time_sentences():
    """Generate kalimat dengan entity TIME"""
    samples = []
    
    for dest_name, dest_tokens in random.sample(DESTINATIONS, 8):
        for time_tokens, _ in random.sample(TIMES, 4):
            # "{dest} buka jam {time}"
            tokens = dest_tokens + ["buka"] + time_tokens
            entities = [
                (0, len(dest_tokens), "DESTINATION"),
                (len(dest_tokens) + 1, len(dest_tokens) + 1 + len(time_tokens), "TIME"),
            ]
            samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
    
    return samples


def generate_location_sentences():
    """Generate kalimat dengan entity LOCATION"""
    samples = []
    
    locations = [
        (["Jalan", "Merdeka"], "Jalan Merdeka"),
        (["Seberang", "Ulu"], "Seberang Ulu"),
        (["Seberang", "Ilir"], "Seberang Ilir"),
        (["pusat", "kota"], "pusat kota"),
        (["dekat", "sungai", "Musi"], "dekat sungai Musi"),
        (["Kertapati"], "Kertapati"),
        (["Ilir", "Barat"], "Ilir Barat"),
    ]
    
    for dest_name, dest_tokens in random.sample(DESTINATIONS, 8):
        for loc_tokens, _ in random.sample(locations, 3):
            # "{dest} ada di {location}"
            tokens = dest_tokens + ["ada", "di"] + loc_tokens
            entities = [
                (0, len(dest_tokens), "DESTINATION"),
                (len(dest_tokens) + 2, len(dest_tokens) + 2 + len(loc_tokens), "LOCATION"),
            ]
            samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
    
    return samples


def generate_english_ner():
    """Generate NER kalimat bahasa Inggris"""
    samples = []
    
    en_dests = [
        ("Ampera Bridge", ["Ampera", "Bridge"]),
        ("Kemaro Island", ["Kemaro", "Island"]),
        ("Kambang Iwak Park", ["Kambang", "Iwak", "Park"]),
        ("Grand Mosque", ["Grand", "Mosque"]),
        ("Benteng Kuto Besak", ["Benteng", "Kuto", "Besak"]),
    ]
    
    for dest_name, dest_tokens in en_dests:
        # "Where is {dest}?"
        tokens = ["Where", "is"] + dest_tokens + ["?"]
        entities = [(2, 2 + len(dest_tokens), "DESTINATION")]
        samples.append({"tokens": tokens, "tags": make_bio_tags(tokens, entities)})
        
        # "How much is the ticket to {dest}?"
        tokens2 = ["How", "much", "is", "the", "ticket", "to"] + dest_tokens + ["?"]
        entities2 = [(6, 6 + len(dest_tokens), "DESTINATION")]
        samples.append({"tokens": tokens2, "tags": make_bio_tags(tokens2, entities2)})
        
        # "Tell me about {dest}"
        tokens3 = ["Tell", "me", "about"] + dest_tokens
        entities3 = [(3, 3 + len(dest_tokens), "DESTINATION")]
        samples.append({"tokens": tokens3, "tags": make_bio_tags(tokens3, entities3)})
        
        # "What time does {dest} open?"
        tokens4 = ["What", "time", "does"] + dest_tokens + ["open", "?"]
        entities4 = [(3, 3 + len(dest_tokens), "DESTINATION")]
        samples.append({"tokens": tokens4, "tags": make_bio_tags(tokens4, entities4)})
    
    return samples


def main():
    print("Generating NER Dataset v2...")
    
    # Load existing dataset
    existing_path = "ml/data/raw/ner_dataset.json"
    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"  Loaded existing: {len(existing)} samples")
    else:
        existing = []
        print("  No existing dataset found, starting fresh")
    
    # Generate new data
    price_data = generate_price_sentences()
    print(f"  Generated PRICE sentences: {len(price_data)}")
    
    abbr_data = generate_abbreviation_sentences()
    print(f"  Generated ABBREVIATION sentences: {len(abbr_data)}")
    
    cat_data = generate_category_sentences()
    print(f"  Generated CATEGORY sentences: {len(cat_data)}")
    
    time_data = generate_time_sentences()
    print(f"  Generated TIME sentences: {len(time_data)}")
    
    loc_data = generate_location_sentences()
    print(f"  Generated LOCATION sentences: {len(loc_data)}")
    
    en_data = generate_english_ner()
    print(f"  Generated English NER sentences: {len(en_data)}")
    
    # Combine
    all_data = existing + price_data + abbr_data + cat_data + time_data + loc_data + en_data
    
    # Deduplicate by token string
    seen = set()
    unique = []
    for item in all_data:
        key = " ".join(item["tokens"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    random.shuffle(unique)
    
    # Save
    output_path = "ml/data/raw/ner_dataset_v2.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal NER samples: {len(unique)}")
    print(f"Saved to: {output_path}")
    
    # Statistik entity
    entity_counts = {}
    for item in unique:
        for tag in item["tags"]:
            if tag.startswith("B-"):
                ent = tag[2:]
                entity_counts[ent] = entity_counts.get(ent, 0) + 1
    
    print("\nEntity distribution:")
    for ent, count in sorted(entity_counts.items()):
        print(f"  {ent}: {count} mentions")


if __name__ == "__main__":
    main()
