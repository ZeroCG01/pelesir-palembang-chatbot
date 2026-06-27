"""
augment_ner.py — Data augmentation khusus untuk NER dataset (token-level).
Script ini dibuat karena augmentasi di augment_data.py hanya untuk intent classifier
(sentence-level), BUKAN untuk NER. NER perlu augmentasi yang menjaga alignment token-tag.

Teknik yang digunakan:
1. Entity Swap       : Ganti entitas dengan sinonim/variasi dari daftar yang sama
2. Sentence Template : Buat kalimat baru dari template dengan slot entitas
3. Token Dropout     : Hapus token 'O' secara acak (context dropout)

Jalankan SEBELUM split_data.py jika ingin augmentasi masuk ke train set,
ATAU jalankan manual lalu gabungkan ke train_ner.json.
"""

import json
import random
import copy

random.seed(42)

# ============================================================
# DAFTAR SINONIM / VARIASI PER ENTITAS
# Tambahkan kata/frasa yang relevan dengan chatbot Palembang
# ============================================================

PRICE_SYNONYMS = [
    ["murah", "terjangkau", "ekonomis", "harga pas", "budget friendly"],
    ["gratis", "free", "tidak berbayar", "tanpa biaya", "bebas biaya"],
    ["mahal", "premium", "luxury"],
    ["harga", "tarif", "biaya", "tiket", "HTM"],
    ["budget", "anggaran", "modal"],
    ["promo", "diskon", "potongan harga", "hemat"],
]

LOCATION_SYNONYMS = [
    # ========== AREA / KAWASAN UMUM ==========
    ["Palembang", "kota Palembang", "kota pempek", "kota wong kito"],
    ["pusat kota", "tengah kota", "center", "downtown"],
    ["sekitar sini", "dekat sini", "area sini", "daerah sini", "nearby"],
    ["dekat", "sekitar", "di area", "di kawasan", "near", "around"],

    # ========== JALAN UTAMA ==========
    ["Jalan Sudirman", "Jln. Jend. Sudirman", "Sudirman", "Jl. Sudirman"],
    ["Jalan Demang Lebar Daun", "Demang Lebar Daun", "Demang", "Jl. Demang"],
    ["Jalan Radial", "Jl. Radial", "Radial"],
    ["Jalan R. Sukamto", "R. Sukamto", "Sukamto", "Jl. Sukamto"],
    ["Jalan Basuki Rahmat", "Basuki Rahmat", "Jl. Basuki"],
    ["Jalan Kapten A. Rivai", "Kapten Rivai", "A. Rivai", "Jl. Rivai"],
    ["Jalan Merdeka", "Jl. Merdeka", "Merdeka"],
    ["Jalan Diponegoro", "Jl. Diponegoro", "Diponegoro"],
    ["Jalan MP Mangkunegara", "Mangkunegara", "Jl. Mangkunegara"],
    ["Jalan SMB II", "Jln. SMB II", "SMB II"],
    ["Jalan Ki. Gede Ing Suro", "Ki Gede Ing Suro", "Jl. Ki Gede"],
    ["Jalan Tasik", "Jl. Tasik", "Tasik"],
    ["Jalan Letkol Iskandar", "Letkol Iskandar", "Jl. Iskandar"],

    # ========== AREA / KELURAHAN / KAWASAN ==========
    ["PTC", "Palembang Trade Center", "area PTC", "mall PTC"],
    ["Transmart", "Transmart Palembang", "area Transmart", "komplek Transmart"],
    ["Jakabaring", "kawasan Jakabaring", "Jak"],
    ["16 Ilir", "pasar 16 Ilir", "kawasan 16 Ilir"],
    ["7 Ulu", "Tujuh Ulu", "kawasan 7 Ulu"],
    ["10 Ulu", "Sepuluh Ulu", "kawasan 10 Ulu"],
    ["9 Ilir", "Sembilan Ilir", "kawasan 9 Ilir"],
    ["19 Ilir", "kawasan 19 Ilir"],
    ["32 Ilir", "kawasan 32 Ilir"],
    ["13 Ulu", "Tiga Belas Ulu", "kawasan 13 Ulu"],
    ["Bukit Kecil", "kawasan Bukit Kecil", "Talang Semut"],
    ["Gandus", "kecamatan Gandus", "kawasan Gandus"],
    ["Alang-Alang Lebar", "Alang Alang Lebar", "kawasan Alang-Alang"],
    ["Ilir Barat", "Ilbrat", "kawasan Ilir Barat"],
    ["Seberang Ulu", "kawasan Seberang Ulu", "Ulu"],
    ["Sungai Musi", "tepian Musi", "tepi Musi", "pinggir Musi"],
    ["Bukit Lama", "kawasan Bukit Lama"],
    ["KM 5", "KM 5,5", "Srijaya KM 5"],
]

PRICE_SYNONYMS = [
    ["murah", "terjangkau", "ekonomis", "harga pas", "budget friendly"],
    ["gratis", "free", "tidak berbayar", "tanpa biaya", "bebas biaya"],
    ["mahal", "premium", "luxury"],
    ["harga", "tarif", "biaya", "tiket", "HTM"],
    ["budget", "anggaran", "modal"],
    ["promo", "diskon", "potongan harga", "hemat"],
]

TIME_SYNONYMS = [
    ["pagi", "pagi hari", "pagi-pagi"],
    ["siang", "siang hari", "tengah hari"],
    ["sore", "sore hari", "petang"],
    ["malam", "malam hari", "malem"],
    ["weekend", "akhir pekan", "sabtu minggu", "sabtu-minggu"],
    ["hari ini", "sekarang", "saat ini"],
    ["besok", "besoknya", "esok hari"],
    ["libur", "hari libur", "tanggal merah"],
]

CATEGORY_SYNONYMS = [
    ["wisata alam", "destinasi alam", "tempat alam"],
    ["wisata sejarah", "destinasi sejarah", "tempat bersejarah"],
    ["kuliner", "makanan", "tempat makan", "restaurant", "restoran"],
    ["wisata keluarga", "destinasi keluarga", "tempat keluarga"],
    ["wisata budaya", "destinasi budaya", "tempat budaya"],
    ["hotel", "penginapan", "akomodasi", "tempat menginap"],
    ["cafe", "kafe", "coffee shop", "ngopi"],
]

DESTINATION_SYNONYMS = [
    # ========== HOTEL / PENGINAPAN ==========
    ["Novotel Palembang", "Novotel", "hotel Novotel", "Novotel PTC"],
    ["Hotel Aryaduta", "Aryaduta", "Aryaduta Palembang"],
    ["Hotel Excelton", "Excelton", "Excelton Palembang"],
    ["Hotel Grand Inna Daira", "Grand Inna Daira", "Inna Daira", "Daira"],
    ["Hotel Sintesa Peninsula", "Sintesa Peninsula", "Peninsula Palembang"],
    ["Hotel Aston Palembang", "Aston Palembang", "Aston"],
    ["Hotel Santika Palembang", "Santika Palembang", "Santika"],
    ["Hotel Harper Palembang", "Harper Palembang", "Harper"],
    ["Hotel Luminor Palembang", "Luminor Palembang", "Luminor"],
    ["Hotel Batiqa Palembang", "Batiqa Palembang", "Batiqa"],
    ["Hotel Ibis Palembang", "Ibis Palembang", "Ibis"],
    ["Hotel Amaris Palembang", "Amaris Palembang", "Amaris"],
    ["Hotel Fave Palembang", "Fave Palembang", "Fave Hotel"],
    ["Hotel Swarna Dwipa", "Swarna Dwipa", "Swarna"],
    ["Hotel The Zuri Palembang", "The Zuri", "Zuri Palembang", "Zuri Transmart"],

    # ========== DAYA TARIK WISATA ==========
    ["Museum Balaputra Dewa", "Balaputra Dewa", "museum Balaputra"],
    ["Bukit Siguntang", "Siguntang", "taman Bukit Siguntang"],
    ["Kawasan Benteng Kuto Besak", "Benteng Kuto Besak", "BKB", "Benteng Kuto", "kawasan BKB"],
    ["Jembatan Ampera", "Ampera", "jembatan ikonik Palembang"],
    ["Pulau Kemaro", "Kemaro", "Kemaro Island"],
    ["Taman Wisata Kerajaan Sriwijaya", "TWKS", "taman Sriwijaya", "Kerajaan Sriwijaya"],
    ["Kampung Kapitan", "Kapitan", "kampung Kapitan"],
    ["Museum Sultan Mahmud Badarudin", "Museum SMB II", "SMB II", "Museum Sultan Mahmud"],
    ["Jakabaring Sport City", "JSC", "sport city"],

    # ========== KULINER / RESTORAN / CAFE ==========
    ["Pempek Candy", "Candy", "pempek Candy"],
    ["Pempek Beringin", "Beringin", "pempek Beringin"],
    ["Pempek Nony 168", "Nony 168", "pempek Nony"],
    ["Mie Celor 26 Ilir H.M. Syafeiz", "Mie Celor Syafeiz", "mie celor 26 Ilir"],
    ["Martabak HAR", "HAR", "martabak HAR"],
    ["Model H. Dowa", "H. Dowa", "Dowa"],
    ["Musi View Resto", "Musi View", "resto Musi View"],
    ["Restoran River Side", "River Side", "riverside BKB"],
    ["Dermaga Point", "Dermaga Point BKB"],
    ["Wisata Kuliner 16 Ilir", "kuliner 16 Ilir", "pasar kuliner 16 Ilir"],

    # ========== WISATA BUDAYA ==========
    ["Zainal Songket", "Songket Zainal", "toko songket Zainal"],
    ["Pusat Kerajinan Tenun Songket 32 Ilir", "songket 32 Ilir", "tenun 32 Ilir"],
    ["Pusat Kerajinan Ukir 19 Ilir", "ukir 19 Ilir", "kerajinan ukir"],
    ["Kampung Arab Al-Munawar", "Al-Munawar", "kampung Arab", "Al Munawar"],
    ["Kampung Rumah Limas 10 Ulu", "rumah limas", "kampung limas"],
    ["Museum Tekstil", "museum tekstil Palembang"],
    ["Kuto Besak Teater", "teater Kuto Besak", "teater BKB"],
    ["Museum A.K. Gani", "museum AK Gani", "AK Gani"],
    ["Rumah Adat Dekranasda", "Dekranasda", "rumah adat Jakabaring"],
    ["Sanggar Tari Rumah Elok", "Rumah Elok", "sanggar tari Palembang"],
]

def find_entity_spans(tags):
    """Ekstrak semua span entitas dari list tag BIO."""
    spans = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == f"I-{etype}":
                i += 1
            spans.append((start, i, etype))  # (start, end_exclusive, type)
        else:
            i += 1
    return spans


def replace_entity_span(tokens, tags, span_start, span_end, new_tokens):
    """Ganti span entitas dengan token baru, update tags sesuai."""
    etype = tags[span_start][2:]
    new_tags = [f"B-{etype}"] + [f"I-{etype}"] * (len(new_tokens) - 1)

    new_token_list = tokens[:span_start] + new_tokens + tokens[span_end:]
    new_tag_list   = tags[:span_start]   + new_tags   + tags[span_end:]
    return new_token_list, new_tag_list


def entity_swap_augment(sample, n_augment=2):
    """Ganti entitas dalam kalimat dengan sinonim dari daftar di atas."""
    results = []
    tokens = sample["tokens"]
    tags   = sample["tags"]
    spans  = find_entity_spans(tags)

    if not spans:
        return results

    synonym_map = {
        "PRICE":       PRICE_SYNONYMS,
        "LOCATION":    LOCATION_SYNONYMS,
        "TIME":        TIME_SYNONYMS,
        "CATEGORY":    CATEGORY_SYNONYMS,
        "DESTINATION": DESTINATION_SYNONYMS,
    }

    for _ in range(n_augment):
        new_tokens = tokens[:]
        new_tags   = tags[:]
        offset = 0  # Track pergeseran index akibat penggantian token

        for (start, end, etype) in spans:
            adj_start = start + offset
            adj_end   = end   + offset

            candidates = synonym_map.get(etype, [])
            if not candidates:
                continue

            # Pilih grup sinonim yang relevan (bisa random)
            group = random.choice(candidates)
            replacement = random.choice(group).split()  # Tokenize sederhana by space

            old_len = adj_end - adj_start
            new_tokens, new_tags = replace_entity_span(
                new_tokens, new_tags, adj_start, adj_end, replacement
            )
            offset += len(replacement) - old_len

        # Hanya tambahkan jika berbeda dari asli
        if new_tokens != tokens:
            results.append({"tokens": new_tokens, "tags": new_tags})

    return results


def token_dropout_augment(sample, dropout_rate=0.15):
    """
    Hapus token non-entitas (tag='O') secara acak.
    Ini membantu model fokus pada konteks minimal.
    Hanya dijalankan jika kalimat cukup panjang.
    """
    tokens = sample["tokens"]
    tags   = sample["tags"]

    if len(tokens) < 5:
        return []

    new_tokens, new_tags = [], []
    for token, tag in zip(tokens, tags):
        if tag == "O" and random.random() < dropout_rate:
            continue  # Drop token O ini
        new_tokens.append(token)
        new_tags.append(tag)

    # Jangan simpan jika hasilnya identik atau terlalu pendek
    if new_tokens == tokens or len(new_tokens) < 3:
        return []
    return [{"tokens": new_tokens, "tags": new_tags}]


def augment_dataset(input_path, output_path, multiplier=3):
    """
    Load dataset, augmentasi, simpan ke file baru.
    multiplier = berapa kali lipat ukuran dataset yang diinginkan.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Data asli: {len(data)} samples")

    augmented = []
    for sample in data:
        augmented.append(sample)  # Selalu simpan asli

        # Entity swap
        swapped = entity_swap_augment(sample, n_augment=2)
        augmented.extend(swapped)

        # Token dropout
        dropped = token_dropout_augment(sample, dropout_rate=0.15)
        augmented.extend(dropped)

    # Deduplicate berdasarkan token sequence
    seen = set()
    unique = []
    for s in augmented:
        key = tuple(s["tokens"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"Setelah augmentasi: {len(unique)} samples (dari {len(data)})")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"Tersimpan ke: {output_path}")
    return unique


if __name__ == "__main__":
    import os

    train_path = "ml/data/processed/train_ner.json"
    aug_path   = "ml/data/processed/train_ner_augmented.json"

    if not os.path.exists(train_path):
        print(f"ERROR: File tidak ditemukan: {train_path}")
        print("Pastikan script dijalankan dari root folder proyek (bukan dari folder training/ner/)")
    else:
        augmented_data = augment_dataset(train_path, aug_path)
        print("\nSelesai! Sekarang update TRAIN_DATA_PATH di config.py:")
        print(f'  TRAIN_DATA_PATH = "{aug_path}"')