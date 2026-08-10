"""
fuzzy_threshold_verify.py — Skrip Verifikasi Ambang Batas (Threshold) Fuzzy String Matching
Memverifikasi efektivitas ambang batas 0,60 (nilai baku difflib / name matching 0,60-0,80)
pada mekanisme Resolusi Entitas Berlapis chatbot Pelesir Palembang.

Mengevaluasi 18 skenario uji gold standard (8 valid + 10 rejected) dan membandingkan
karakteristik performa pada ambang batas 0,40 (longgar), 0,55, 0,60 (produksi baru), dan 0,70 (ketat).
"""

import os
import sys
import json
import csv
import difflib
from dotenv import load_dotenv

# Pastikan path import dari root direktori proyek
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.services.nlp_service import get_destination_names, ChatbotModel

OUTPUT_DIR = "output/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 1. DAFTAR NOISE WORDS (Identik dengan nlp_service.py)
# =====================================================================
NOISE_WORDS = [
    "berapa", "harga", "tiket", "masuk", "dari", "ke", "di", "untuk",
    "jam", "buka", "tutup", "operasional", "alamat", "lokasi", "dimana",
    "fasilitas", "apa", "saja", "ada", "yang", "nya", "dong", "ya",
    "kasih", "tau", "info", "tentang", "gimana", "bagaimana", "museum",
    "wisata", "tempat", "taman", "masjid", "kampung", "kawasan", "pulau",
    "jembatan", "hutan", "sungai", "kolam", "renang", "wahana", "kuliner",
    "sejarah", "kategori", "disana", "sini", "sana", "buat",
    "apakah", "ga", "gak", "nggak",
    "palembang", "naik", "dekat",
]

# =====================================================================
# 2. 18 SKENARIO UJI GOLD STANDARD (Identik dengan fuzzy_threshold_eval.py)
#    (8 Valid Positif + 10 Harus Ditolak / Negative)
# =====================================================================
TEST_CASES = [
    # --- Normal (Harus match ke nama resmi di DB) ---
    (1,  "Normal",  "jam buka museum balaputradewa",       "Museum Balaputra Dewa"),
    (2,  "Normal",  "fasilitas di punti kayu",             "Hutan Wisata Punti Kayu"),
    (3,  "Normal",  "harga tiket benteng kuto besak",      "Kawasan Benteng Kuto Besak (BKB)"),
    (4,  "Normal",  "info tentang kampung kapitan",        "Kampung Kapitan"),
    (5,  "Normal",  "lokasi bukit siguntang dimana",       "Bukit Siguntang"),
    (6,  "Normal",  "jam operasional kambang iwak",        "Kambang Iwak Besak"),
    (7,  "Normal",  "tiket masuk amanzi waterpark",        None),  # Amanzi Waterpark TIDAK ADA di DB -> Ditolak
    (8,  "Normal",  "fasilitas jakabaring",                "Jakabaring Sport City"),
    (9,  "Normal",  "harga masuk fantasy island",          None),  # Fantasy Island TIDAK ADA di DB -> Ditolak
    (10, "Normal",  "info al quran al akbar",              "Bayt Al-Quran Al-Akbar"),
    # --- Trap (Harus None / Ditolak) ---
    (11, "Trap",    "ada fasilitas kolam renang ga disana?", None),
    (12, "Trap",    "harganya brapa?",                     None),
    (13, "Trap",    "buka 24 jam",                         None),
    (14, "Trap",    "rekomendasi wisata sejarah",          None),
    (15, "Trap",    "ada tempat wisata dekat sini gak",    None),
    # --- Kritis (Harus None / Ditolak) ---
    (16, "Kritis",  "wisata kuliner palembang",            None),
    (17, "Kritis",  "tempat wisata yang bagus",            None),
    (18, "Kritis",  "cara ke sana naik apa",               None),
]

THRESHOLDS_TO_EVALUATE = [0.40, 0.55, 0.60, 0.70]


# =====================================================================
# 3. FUNGSI PENGUJIAN FUZZY MATCHING (PERSIS SESUAI PRODUKSI)
# =====================================================================
def fuzzy_match(text: str, db_names: list, threshold: float):
    """
    Menjalankan algoritma SequenceMatcher (Ratcliff-Obershelp)
    dengan pembersihan noise words & min length guard persis seperti nlp_service.py.
    """
    text_clean = (text.lower()
                  .replace("?", "").replace("!", "")
                  .replace(".", "").replace(",", "").strip())
    words = text_clean.split()

    candidates = []
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            chunk = " ".join(words[start:start + length])
            chunk_words = chunk.split()

            # Guard 1: Skip chunk yang 100% noise
            if all(w in NOISE_WORDS for w in chunk_words):
                continue

            # Guard 2: Skip chunk jika panjang karakter non-noise < 4
            clean_chunk_words = [w for w in chunk_words if w not in NOISE_WORDS]
            clean_chunk = " ".join(clean_chunk_words).strip()
            if len(clean_chunk) < 4:
                continue

            candidates.append(chunk)

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        for db_name in db_names:
            score = difflib.SequenceMatcher(None, candidate, db_name.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = db_name

    return best_match, best_score


def main():
    print("=" * 95)
    print("🔍 VERIFIKASI EMPIRIS AMBANG BATAS FUZZY MATCHING (THRESHOLD 0,60)")
    print("=" * 95)

    # 1. Muat data destinasi resmi dari basis data Supabase
    db_names = get_destination_names()
    print(f"📦 Berhasil memuat {len(db_names)} nama destinasi resmi dari basis data Supabase.")
    print(f"⚙️ Ambang batas produksi aktif saat ini: 0,60 (Baku difflib / name matching).\n")

    # 2. Evaluasi 18 kasus pada seluruh nilai ambang
    stats = {t: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for t in THRESHOLDS_TO_EVALUATE}
    detailed_cases = []

    print(f"{'No':<3} | {'Kategori':<8} | {'Query Pengguna':<36} | {'Target DB':<28} | {'T=0.40':<8} | {'T=0.55':<8} | {'T=0.60':<8} | {'T=0.70':<8}")
    print("-" * 125)

    for no, cat, query, expected in TEST_CASES:
        case_info = {
            "no": no,
            "category": cat,
            "query": query,
            "expected": expected
        }

        row_print = f"{no:<3} | {cat:<8} | {query:<36} | {str(expected):<28}"

        for t in THRESHOLDS_TO_EVALUATE:
            match_res, score = fuzzy_match(query, db_names, t)
            case_info[f"match_{t}"] = match_res
            case_info[f"score_{t}"] = score

            if expected is not None:
                # Kasus Positif (Destinasi Valid di DB)
                if match_res == expected:
                    cls_label = "TP"
                    stats[t]["TP"] += 1
                elif match_res is not None:
                    cls_label = "FP"  # Match ke nama tempat yang salah
                    stats[t]["FP"] += 1
                else:
                    cls_label = "FN"  # Gagal mendeteksi tempat valid
                    stats[t]["FN"] += 1
            else:
                # Kasus Negatif (Trap / OOD / Destinasi di luar DB)
                if match_res is None:
                    cls_label = "TN"
                    stats[t]["TN"] += 1
                else:
                    cls_label = "FP"  # Salah tangkap kata noise sebagai destinasi
                    stats[t]["FP"] += 1

            case_info[f"cls_{t}"] = cls_label
            status_symbol = "✅ " if cls_label in ("TP", "TN") else "❌ "
            row_print += f" | {status_symbol}{cls_label:<4}"

        detailed_cases.append(case_info)
        print(row_print)

    print("-" * 125)

    # 3. Hitung Precision, Recall, F1 untuk tiap ambang batas
    summary_rows = []
    for t in THRESHOLDS_TO_EVALUATE:
        tp = stats[t]["TP"]
        tn = stats[t]["TN"]
        fp = stats[t]["FP"]
        fn = stats[t]["FN"]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        stats[t]["Precision"] = prec
        stats[t]["Recall"] = rec
        stats[t]["F1"] = f1

        summary_rows.append({
            "threshold": f"{t:.2f}".replace(".", ","),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": f"{prec*100:.2f}%".replace(".", ","),
            "recall": f"{rec*100:.2f}%".replace(".", ","),
            "f1": f"{f1*100:.2f}%".replace(".", ",")
        })

    # 4. Cetak Tabel Ringkasan Metrik
    print("\n" + "=" * 90)
    print("📊 TABEL PERBANDINGAN METRIK EVALUASI AMBANG BATAS FUZZY MATCHING")
    print("=" * 90)
    print(f"{'Threshold':<12} | {'TP':<5} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Kategori Zona'}")
    print("-" * 90)
    for row, t in zip(summary_rows, THRESHOLDS_TO_EVALUATE):
        if t == 0.40:
            zona = "Longgar (Banyak False Positive)"
        elif t == 0.55:
            zona = "Moderat (Ambang Awal)"
        elif t == 0.60:
            zona = "⭐ PRODUKSI FINAL (Sweet Spot difflib)"
        else:
            zona = "Ketat (Recall Menurun)"
        print(f"{row['threshold']:<12} | {row['tp']:<5} | {row['tn']:<5} | {row['fp']:<5} | {row['fn']:<5} | {row['precision']:<12} | {row['recall']:<12} | {row['f1']:<12} | {zona}")
    print("=" * 90)

    # 5. Konfirmasi Nilai Acuan
    print("\n🔍 KONFIRMASI NILAI ACUAN:")
    ref_040 = (stats[0.40]["Precision"] == 4/7 and stats[0.40]["Recall"] == 1.0)
    ref_055 = (stats[0.55]["Precision"] == 8/10 and stats[0.55]["Recall"] == 1.0)
    ref_070 = (stats[0.70]["Precision"] == 1.0 and stats[0.70]["Recall"] == 6/8)

    print(f"- T = 0,40: Precision = {stats[0.40]['Precision']*100:.2f}%, Recall = {stats[0.40]['Recall']*100:.2f}%, F1 = {stats[0.40]['F1']*100:.2f}% -> {'✅ MATCH ACUAN' if ref_040 else '❌'}")
    print(f"- T = 0,55: Precision = {stats[0.55]['Precision']*100:.2f}%, Recall = {stats[0.55]['Recall']*100:.2f}%, F1 = {stats[0.55]['F1']*100:.2f}% -> {'✅ MATCH ACUAN' if ref_055 else '❌'}")
    print(f"- T = 0,60: Precision = {stats[0.60]['Precision']*100:.2f}%, Recall = {stats[0.60]['Recall']*100:.2f}%, F1 = {stats[0.60]['F1']*100:.2f}% -> ⭐ VALIDASI OPTIMAL")
    print(f"- T = 0,70: Precision = {stats[0.70]['Precision']*100:.2f}%, Recall = {stats[0.70]['Recall']*100:.2f}%, F1 = {stats[0.70]['F1']*100:.2f}% -> {'✅ MATCH ACUAN' if ref_070 else '❌'}")

    # 6. Simpan ke berkas CSV dan JSON
    csv_paths = ["fuzzy_threshold_verify.csv", os.path.join(OUTPUT_DIR, "fuzzy_threshold_verify.csv")]
    for cp in csv_paths:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1"])
            writer.writeheader()
            writer.writerows(summary_rows)

    json_payload = {
        "summary": summary_rows,
        "detailed_cases": detailed_cases
    }

    json_paths = ["fuzzy_threshold_verify.json", os.path.join(OUTPUT_DIR, "fuzzy_threshold_verify.json")]
    for jp in json_paths:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Berkas data hasil verifikasi tersimpan:")
    print(f"  - CSV : fuzzy_threshold_verify.csv & {os.path.join(OUTPUT_DIR, 'fuzzy_threshold_verify.csv')}")
    print(f"  - JSON: fuzzy_threshold_verify.json & {os.path.join(OUTPUT_DIR, 'fuzzy_threshold_verify.json')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
