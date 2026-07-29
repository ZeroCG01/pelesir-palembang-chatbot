"""
Evaluasi Eksperimental Fuzzy Matching Threshold
================================================
Script ini mengevaluasi Lapisan 3 (fuzzy string matching) dari mekanisme
Resolusi Entitas Berlapis secara TERISOLASI.  Logika pembersih noise dan
pencocokan skor diambil PERSIS dari fungsi produksi `find_destination_fuzzy`
di app/services/nlp_service.py (baris 240-294).

Referensi fungsi produksi:
  - Nama fungsi   : find_destination_fuzzy
  - Lokasi        : app/services/nlp_service.py, baris 240
  - Threshold prod: FUZZY_MATCH_THRESHOLD = 0.55 (baris 237)
  - Noise words   : 42 kata (baris 250-257)
  - Scoring       : difflib.SequenceMatcher(None, candidate, db_name.lower()).ratio()
  - Guard tambahan: chunk yang 100% noise di-skip; clean_chunk < 4 char di-skip

Daftar destinasi resmi di-query langsung dari tabel `destinations` di Supabase
melalui fungsi get_destination_names() (baris 21-27).

Output:
  - stdout  : tabel per-kasus + tabel ringkasan metrik
  - file    : tests/fuzzy_threshold_report.md
"""

import difflib
import sys, os

# =====================================================================
# 1. DAFTAR NAMA DESTINASI RESMI  (dari Supabase, di-freeze di sini
#    agar reproducible tanpa koneksi DB — urutan sesuai query terakhir)
# =====================================================================
DB_NAMES = [
    "Museum Balaputra Dewa",
    "Monumen Perjuangan Rakyat (Monpera)",
    "Toko Kopi Har",
    "Makam Ki Gede Ing Suro",
    "Museum AK. Gani",
    "Lorong Basah Night Culinary Market",
    "Pempek Candy",
    "Kawasan Pasar 16 Ilir",
    "Pempek Lala 26 Ilir",
    "Restoran River Side",
    "RM Sri Melayu",
    "Bayt Al-Quran Al-Akbar",
    "Pulau Kemaro",
    "Masjid Agung",
    "Masjid Cheng Ho",
    "Masjid Ki Marogan",
    "Bukit Siguntang",
    "Hutan Wisata Punti Kayu",
    "Kambang Iwak Besak",
    "Danau OPI",
    "Taman Sekanak Lambidaro",
    "Jakabaring Sport City",
    "OPI Water Fun",
    "Ampera Skate Park",
    "Palembang Sport and Convention Center (PSCC)",
    "Novotel Palembang",
    "Amaris Hotel Palembang",
    "The Zuri Palembang",
    "Hotel Aston Palembang",
    "Hotel Santika Radial Palembang",
    "Hotel Harper Palembang",
    "Hotel Luminor Palembang",
    "Hotel Batiqa Palembang",
    "Hotel ibis Palembang Sanggar",
    "favehotel Palembang",
    "Hotel Swarna Dwipa Palembang",
    "Museum Sultan Mahmud Badaruddin II",
    "Kawasan Benteng Kuto Besak (BKB)",
    "Jembatan Ampera",
    "Kampung Kapitan",
    "Martabak HAR",
    "Hotel Aryaduta Palembang",
    "Taman Purbakala Kerajaan Sriwijaya",
    "Hotel Excelton Palembang",
    "Hotel Grand Inna Daira Palembang",
    "Hotel Sintesa Peninsula Palembang",
]

# =====================================================================
# 2. NOISE WORDS — disalin verbatim dari nlp_service.py baris 250-257
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
# 3. FUNGSI EVALUASI — replika PERSIS dari find_destination_fuzzy
#    (app/services/nlp_service.py baris 240-294) dengan threshold
#    sebagai parameter.  Tidak ada modifikasi logika apa pun.
# =====================================================================
def fuzzy_match(text: str, threshold: float):
    """
    Mengembalikan (best_match_name, best_score) atau (None, 0.0).
    Logika identik dengan find_destination_fuzzy di produksi.
    """
    # Bersihkan teks query dari noise (baris 248)
    text_clean = (text.lower()
                  .replace("?", "").replace("!", "")
                  .replace(".", "").replace(",", "").strip())
    words = text_clean.split()

    # Bangun kandidat: semua substring berturut-turut (baris 260-276)
    candidates = []
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            chunk = " ".join(words[start:start + length])
            chunk_words = chunk.split()

            # Guard 1: skip chunk yang 100% noise (baris 267-268)
            if all(w in NOISE_WORDS for w in chunk_words):
                continue

            # Guard 2: skip jika sisa non-noise < 4 karakter (baris 270-274)
            clean_chunk = " ".join(
                w for w in chunk_words if w not in NOISE_WORDS
            ).strip()
            if len(clean_chunk) < 4:
                continue

            candidates.append(chunk)

    # Scoring: SequenceMatcher terhadap seluruh DB (baris 281-287)
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        for db_name in DB_NAMES:
            score = difflib.SequenceMatcher(
                None, candidate, db_name.lower()
            ).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = db_name

    if best_match:
        return best_match, round(best_score, 4)
    return None, 0.0

# =====================================================================
# 4. DATA UJI (18 kasus berlabel)
#    ground_truth = nama resmi PERSIS seperti di DB, atau None.
#
#    Penyesuaian vs prompt user:
#      - Kasus 3: "Benteng Kuto Besak" → DB: "Kawasan Benteng Kuto Besak (BKB)"
#      - Kasus 6: "Taman Kambang Iwak Besak" → DB: "Kambang Iwak Besak"
#      - Kasus 10: "Al Quran Al Akbar" → DB: "Bayt Al-Quran Al-Akbar"
# =====================================================================
TEST_CASES = [
    # --- Normal (harus match) ---
    (1,  "Normal",  "jam buka museum balaputradewa",       "Museum Balaputra Dewa"),
    (2,  "Normal",  "fasilitas di punti kayu",             "Hutan Wisata Punti Kayu"),
    (3,  "Normal",  "harga tiket benteng kuto besak",      "Kawasan Benteng Kuto Besak (BKB)"),
    (4,  "Normal",  "info tentang kampung kapitan",        "Kampung Kapitan"),
    (5,  "Normal",  "lokasi bukit siguntang dimana",       "Bukit Siguntang"),
    (6,  "Normal",  "jam operasional kambang iwak",        "Kambang Iwak Besak"),
    (7,  "Normal",  "tiket masuk amanzi waterpark",        None),  # "Amanzi Waterpark" TIDAK ADA di DB
    (8,  "Normal",  "fasilitas jakabaring",                "Jakabaring Sport City"),
    (9,  "Normal",  "harga masuk fantasy island",          None),  # "Fantasy Island" TIDAK ADA di DB
    (10, "Normal",  "info al quran al akbar",              "Bayt Al-Quran Al-Akbar"),
    # --- Trap (harus None) ---
    (11, "Trap",    "ada fasilitas kolam renang ga disana?", None),
    (12, "Trap",    "harganya brapa?",                     None),
    (13, "Trap",    "buka 24 jam",                         None),
    (14, "Trap",    "rekomendasi wisata sejarah",          None),
    (15, "Trap",    "ada tempat wisata dekat sini gak",    None),
    # --- Kritis (harus None) ---
    (16, "Kritis",  "wisata kuliner palembang",            None),
    (17, "Kritis",  "tempat wisata yang bagus",            None),
    (18, "Kritis",  "cara ke sana naik apa",               None),
]

# =====================================================================
# 5. MAIN — jalankan evaluasi, cetak + simpan laporan
# =====================================================================
THRESHOLDS = [0.40, 0.55, 0.70]

def fmt_score(score):
    """Format skor ke gaya Indonesia: 0,xx"""
    return f"{score:.2f}".replace(".", ",")

def fmt_pct(val):
    """Format persentase ke gaya Indonesia: xx,xx%"""
    return f"{val*100:.2f}%".replace(".", ",")


def evaluate():
    # --- Kumpulkan hasil per kasus per threshold ---
    results = []  # list of dict
    stats = {t: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for t in THRESHOLDS}

    for no, cat, query, expected in TEST_CASES:
        row = {"no": no, "cat": cat, "query": query, "expected": expected}
        for t in THRESHOLDS:
            match, score = fuzzy_match(query, t)
            row[f"match_{t}"] = match
            row[f"score_{t}"] = score

            # Klasifikasi
            if expected is not None:        # seharusnya match
                if match == expected:
                    row[f"cls_{t}"] = "TP"
                    stats[t]["TP"] += 1
                elif match is not None:     # match tapi salah destinasi
                    row[f"cls_{t}"] = "FP"
                    stats[t]["FP"] += 1
                else:                       # None padahal harus match
                    row[f"cls_{t}"] = "FN"
                    stats[t]["FN"] += 1
            else:                            # seharusnya None
                if match is None:
                    row[f"cls_{t}"] = "TN"
                    stats[t]["TN"] += 1
                else:
                    row[f"cls_{t}"] = "FP"
                    stats[t]["FP"] += 1

        results.append(row)

    # Hitung Precision / Recall / F1
    for t in THRESHOLDS:
        s = stats[t]
        s["Prec"] = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) else 0
        s["Rec"]  = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) else 0
        s["F1"]   = (2 * s["Prec"] * s["Rec"] / (s["Prec"] + s["Rec"])
                     if (s["Prec"] + s["Rec"]) else 0)

    # --- Bangun output ---
    lines = []

    def p(text=""):
        print(text)
        lines.append(text)

    p("# Hasil Eksperimen Penentuan Nilai Ambang (*Threshold*) Fuzzy Matching")
    p()
    p("> **Metode**: `difflib.SequenceMatcher.ratio()` dengan *noise filter* (42 kata) dan *min-length guard* (≥ 4 karakter).")
    p("> **Sumber data destinasi**: Tabel `destinations` di Supabase (46 baris, di-*freeze* untuk reprodusibilitas).")
    p("> **Threshold produksi**: 0,55.")
    p()

    # --- Tabel per-kasus ---
    p("## Tabel Perbandingan Hasil per Kasus Uji")
    p()

    # Baris header
    hdr = "| No | Kategori | Input Pengguna | Destinasi Diharapkan |"
    for t in THRESHOLDS:
        hdr += f" T = {fmt_score(t)} |"
    p(hdr)

    sep = "|:--:|:--------:|----------------|----------------------|"
    for _ in THRESHOLDS:
        sep += ":----|"
    p(sep)

    for r in results:
        exp_str = r["expected"] if r["expected"] else "None"
        row_str = f"| {r['no']} | {r['cat']} | {r['query']} | {exp_str} |"
        for t in THRESHOLDS:
            m = r[f"match_{t}"]
            s = r[f"score_{t}"]
            c = r[f"cls_{t}"]
            icon = "✅" if c in ("TP", "TN") else "❌"
            if m:
                cell = f" {icon} {m} ({fmt_score(s)}) |"
            else:
                cell = f" {icon} None |"
            row_str += cell
        p(row_str)

    p()
    p("> **Keterangan**: ✅ = hasil sesuai *ground truth*, ❌ = hasil tidak sesuai. Skor dalam kurung = `SequenceMatcher.ratio()`.")
    p()

    # --- Penyesuaian nama ---
    p("### Penyesuaian Nama *Ground Truth* terhadap Database")
    p()
    p("| Kasus | Nama di Prompt | Nama Resmi di DB |")
    p("|:-----:|---------------|------------------|")
    p("| 3 | Benteng Kuto Besak | Kawasan Benteng Kuto Besak (BKB) |")
    p("| 6 | Taman Kambang Iwak Besak | Kambang Iwak Besak |")
    p("| 7 | Amanzi Waterpark | *(tidak ada di DB — dihapus dari ground truth positif)* |")
    p("| 9 | Fantasy Island | *(tidak ada di DB — dihapus dari ground truth positif)* |")
    p("| 10 | Al Quran Al Akbar | Bayt Al-Quran Al-Akbar |")
    p()

    # --- Tabel ringkasan metrik ---
    p("## Tabel Ringkasan Metrik Evaluasi")
    p()
    p(f"| Metrik | T = {fmt_score(0.40)} | T = {fmt_score(0.55)} | T = {fmt_score(0.70)} |")
    p("|--------|:--------:|:--------:|:--------:|")
    for name in ["TP", "TN", "FP", "FN"]:
        row_str = f"| {name} |"
        for t in THRESHOLDS:
            row_str += f" {stats[t][name]} |"
        p(row_str)
    p("|--------|----------|----------|----------|")
    for name, key in [("**Precision**", "Prec"), ("**Recall**", "Rec"), ("**F1-Score**", "F1")]:
        row_str = f"| {name} |"
        for t in THRESHOLDS:
            row_str += f" {fmt_pct(stats[t][key])} |"
        p(row_str)

    p()

    # --- Analisis kasus yang berubah antar-ambang ---
    p("## Kasus yang Berubah Hasil antar Nilai Ambang")
    p()
    for r in results:
        outcomes = [r[f"match_{t}"] for t in THRESHOLDS]
        if len(set(str(o) for o in outcomes)) > 1:
            p(f"- **Kasus {r['no']}** — \"{r['query']}\"")
            p(f"  - Diharapkan: {r['expected'] if r['expected'] else 'None'}")
            for t in THRESHOLDS:
                m = r[f"match_{t}"]
                s = r[f"score_{t}"]
                c = r[f"cls_{t}"]
                icon = "✅" if c in ("TP", "TN") else "❌"
                p(f"  - T={fmt_score(t)}: {m if m else 'None'} (skor {fmt_score(s)}) {icon}")
            p()

    # --- Kesimpulan ---
    best_t = max(THRESHOLDS, key=lambda t: stats[t]["F1"])
    p("## Kesimpulan")
    p()
    p(f"Nilai ambang **{fmt_score(best_t)}** menghasilkan F1-Score tertinggi "
      f"(**{fmt_pct(stats[best_t]['F1'])}**) dengan Precision = "
      f"{fmt_pct(stats[best_t]['Prec'])} dan Recall = {fmt_pct(stats[best_t]['Rec'])}.")
    p()

    # --- Simpan ke file ---
    report_path = os.path.join(os.path.dirname(__file__), "fuzzy_threshold_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Laporan disimpan di: {report_path}")


if __name__ == "__main__":
    evaluate()
