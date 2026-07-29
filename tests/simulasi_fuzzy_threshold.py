"""
Simulasi Pengujian Fuzzy Threshold
Membandingkan threshold 0.40, 0.55, dan 0.70 terhadap berbagai variasi input pengguna.
Output: tabel perbandingan siap untuk skripsi.
"""
import difflib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === DAFTAR NAMA DESTINASI (Source of Truth dari database Supabase) ===
DB_NAMES = [
    "Kawasan Benteng Kuto Besak (BKB)",
    "Jembatan Ampera",
    "Pulau Kemaro",
    "Monumen Perjuangan Rakyat (Monpera)",
    "Museum Sultan Mahmud Badaruddin II",
    "Taman Kambang Iwak Besak",
    "Hutan Wisata Punti Kayu",
    "Masjid Agung Palembang",
    "Kampung Kapitan",
    "Kampung Arab Al-Munawar",
    "Museum Balaputra Dewa",
    "Al Quran Al Akbar",
    "Bukit Siguntang",
    "Masjid Cheng Ho",
    "Fantasy Island",
    "Amanzi Waterpark",
    "Taman Purbakala Kerajaan Sriwijaya",
    "Museum AK. Gani",
    "Makam Ki Gede Ing Suro",
    "Jakabaring Sport City",
    "Toko Kopi Har",
    "Lorong Basah Night Culinary Market",
    "Pempek Candy",
    "Kawasan Pasar 16 Ilir",
    "Pempek Lala 26 Ilir",
    "Restoran River Side",
    "RM Sri Melayu",
    "Martabak HAR",
]

# === NOISE WORDS (sinkron dengan nlp_service.py) ===
NOISE_WORDS = [
    "berapa", "harga", "tiket", "masuk", "dari", "ke", "di", "untuk",
    "jam", "buka", "tutup", "operasional", "alamat", "lokasi", "dimana",
    "fasilitas", "apa", "saja", "ada", "yang", "nya", "dong", "ya",
    "kasih", "tau", "info", "tentang", "gimana", "bagaimana", "museum",
    "wisata", "tempat", "taman", "masjid", "kampung", "kawasan", "pulau",
    "jembatan", "hutan", "sungai", "kolam", "renang", "wahana", "kuliner",
    "sejarah", "kategori", "disana", "sini", "sana", "buat",
    "apakah", "ga", "gak", "nggak"
]

def find_destination_fuzzy(text: str, threshold: float):
    """Replika logika fuzzy matching dari nlp_service.py"""
    text_clean = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").strip()
    words = text_clean.split()
    
    candidates = []
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            chunk = " ".join(words[start:start+length])
            chunk_words = chunk.split()
            
            if all(w in NOISE_WORDS for w in chunk_words):
                continue
            
            clean_chunk_words = [w for w in chunk_words if w not in NOISE_WORDS]
            clean_chunk = " ".join(clean_chunk_words).strip()
            if len(clean_chunk) < 4:
                continue
                
            candidates.append(chunk)
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        for db_name in DB_NAMES:
            score = difflib.SequenceMatcher(None, candidate, db_name.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = db_name
    
    if best_match:
        return best_match, round(best_score, 4)
    return None, 0.0


# === SKENARIO UJI ===
TEST_CASES = [
    # --- Kasus Normal (harus cocok) ---
    ("jam buka museum balaputradewa", "Museum Balaputra Dewa", "Normal", "Typo ringan pada nama destinasi"),
    ("fasilitas di punti kayu", "Hutan Wisata Punti Kayu", "Normal", "Nama pendek dari nama resmi panjang"),
    ("harga tiket benteng kuto besak", "Kawasan Benteng Kuto Besak (BKB)", "Normal", "Nama destinasi tanpa awalan 'Kawasan'"),
    ("info tentang kampung kapitan", "Kampung Kapitan", "Normal", "Nama destinasi lengkap dan tepat"),
    ("lokasi bukit siguntang dimana", "Bukit Siguntang", "Normal", "Nama destinasi tepat dengan noise words"),
    ("jam operasional kambang iwak", "Taman Kambang Iwak Besak", "Normal", "Nama pendek, DB punya nama panjang"),
    ("tiket masuk amanzi waterpark", "Amanzi Waterpark", "Normal", "Nama destinasi tepat"),
    ("fasilitas jakabaring", "Jakabaring Sport City", "Normal", "Nama pendek dari nama resmi"),
    ("harga masuk fantasy island", "Fantasy Island", "Normal", "Nama destinasi tepat (bahasa Inggris)"),
    ("info al quran al akbar", "Al Quran Al Akbar", "Normal", "Nama destinasi lengkap"),
    
    # --- Kasus Jebakan / Trap (TIDAK BOLEH cocok) ---
    ("ada fasilitas kolam renang ga disana?", None, "Trap", "Noise murni tanpa nama destinasi"),
    ("harganya brapa?", None, "Trap", "Follow-up tanpa menyebut tempat"),
    ("buka 24 jam", None, "Trap", "Pertanyaan umum tanpa destinasi"),
    ("rekomendasi wisata sejarah", None, "Trap", "Permintaan rekomendasi, bukan destinasi spesifik"),
    ("ada tempat wisata dekat sini gak", None, "Trap", "Kata generik tanpa nama tempat"),
    
    # --- Kasus Kritis (rentan false positive) ---
    ("wisata kuliner palembang", None, "Kritis", "Kata 'kuliner' mirip nama destinasi?"),
    ("tempat wisata yang bagus", None, "Kritis", "Kalimat generik, bisa false-match"),
    ("cara ke sana naik apa", None, "Kritis", "Follow-up umum tanpa destinasi"),
]


def run_simulation():
    thresholds = [0.40, 0.55, 0.70]
    
    print("=" * 130)
    print("SIMULASI PENGUJIAN FUZZY MATCHING THRESHOLD")
    print("Metode: difflib.SequenceMatcher dengan Noise Filter + Min-Length Guard")
    print("=" * 130)
    
    # Track metrics per threshold
    stats = {t: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for t in thresholds}
    
    # Print detailed results per case
    for i, (query, expected, category, desc) in enumerate(TEST_CASES, 1):
        print(f"\n--- Kasus {i} [{category}] ---")
        print(f"  Input   : \"{query}\"")
        print(f"  Expected: {expected if expected else '(None — harus minta klarifikasi)'}")
        print(f"  Deskripsi: {desc}")
        
        for t in thresholds:
            match, score = find_destination_fuzzy(query, t)
            
            # Tentukan status
            if expected is not None:  # Seharusnya cocok
                if match and (expected.lower() in match.lower() or match.lower() in expected.lower()):
                    status = "TP (True Positive)"
                    stats[t]["TP"] += 1
                elif match:
                    status = "FP (False Positive — salah tempat!)"
                    stats[t]["FP"] += 1
                else:
                    status = "FN (False Negative — gagal mencocokkan)"
                    stats[t]["FN"] += 1
            else:  # Seharusnya TIDAK cocok
                if match:
                    status = "FP (False Positive — harusnya None!)"
                    stats[t]["FP"] += 1
                else:
                    status = "TN (True Negative)"
                    stats[t]["TN"] += 1
            
            icon = "✅" if "True" in status else "❌"
            match_str = f"\"{match}\"" if match else "(None)"
            print(f"    T={t:.2f}: {match_str:<45} skor={score:.4f}  {icon} {status}")
    
    # === RINGKASAN METRIK ===
    print("\n\n" + "=" * 80)
    print("RINGKASAN METRIK PER THRESHOLD")
    print("=" * 80)
    print(f"{'Metrik':<25} | {'T=0.40':>12} | {'T=0.55':>12} | {'T=0.70':>12}")
    print("-" * 70)
    
    for t in thresholds:
        s = stats[t]
        s["Precision"] = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        s["Recall"] = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0
        s["F1"] = 2 * s["Precision"] * s["Recall"] / (s["Precision"] + s["Recall"]) if (s["Precision"] + s["Recall"]) > 0 else 0
    
    for metric_name in ["TP", "TN", "FP", "FN"]:
        row = f"{metric_name:<25} |"
        for t in thresholds:
            row += f" {stats[t][metric_name]:>12} |"
        print(row)
    
    print("-" * 70)
    
    for metric_name in ["Precision", "Recall", "F1"]:
        row = f"{metric_name:<25} |"
        for t in thresholds:
            row += f" {stats[t][metric_name]:>12.2%} |"
        print(row)
    
    print("-" * 70)
    
    # Tentukan threshold terbaik berdasarkan F1
    best_t = max(thresholds, key=lambda t: stats[t]["F1"])
    print(f"\n🏆 Threshold optimal berdasarkan F1-Score tertinggi: T = {best_t:.2f} (F1 = {stats[best_t]['F1']:.2%})")
    print(f"   Precision = {stats[best_t]['Precision']:.2%}, Recall = {stats[best_t]['Recall']:.2%}")


if __name__ == "__main__":
    run_simulation()
