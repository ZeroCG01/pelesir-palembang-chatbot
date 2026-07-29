# Hasil Eksperimen Penentuan Nilai Ambang (*Threshold*) Fuzzy Matching

> **Metode**: `difflib.SequenceMatcher.ratio()` dengan *noise filter* (42 kata) dan *min-length guard* (≥ 4 karakter).
> **Sumber data destinasi**: Tabel `destinations` di Supabase (46 baris, di-*freeze* untuk reprodusibilitas).
> **Threshold produksi**: 0,55.

## Tabel Perbandingan Hasil per Kasus Uji

| No | Kategori | Input Pengguna | Destinasi Diharapkan | T = 0,40 | T = 0,55 | T = 0,70 |
|:--:|:--------:|----------------|----------------------|:----|:----|:----|
| 1 | Normal | jam buka museum balaputradewa | Museum Balaputra Dewa | ✅ Museum Balaputra Dewa (0,98) | ✅ Museum Balaputra Dewa (0,98) | ✅ Museum Balaputra Dewa (0,98) |
| 2 | Normal | fasilitas di punti kayu | Hutan Wisata Punti Kayu | ✅ Hutan Wisata Punti Kayu (0,67) | ✅ Hutan Wisata Punti Kayu (0,67) | ❌ None |
| 3 | Normal | harga tiket benteng kuto besak | Kawasan Benteng Kuto Besak (BKB) | ✅ Kawasan Benteng Kuto Besak (BKB) (0,72) | ✅ Kawasan Benteng Kuto Besak (BKB) (0,72) | ✅ Kawasan Benteng Kuto Besak (BKB) (0,72) |
| 4 | Normal | info tentang kampung kapitan | Kampung Kapitan | ✅ Kampung Kapitan (1,00) | ✅ Kampung Kapitan (1,00) | ✅ Kampung Kapitan (1,00) |
| 5 | Normal | lokasi bukit siguntang dimana | Bukit Siguntang | ✅ Bukit Siguntang (1,00) | ✅ Bukit Siguntang (1,00) | ✅ Bukit Siguntang (1,00) |
| 6 | Normal | jam operasional kambang iwak | Kambang Iwak Besak | ✅ Kambang Iwak Besak (0,80) | ✅ Kambang Iwak Besak (0,80) | ✅ Kambang Iwak Besak (0,80) |
| 7 | Normal | tiket masuk amanzi waterpark | None | ❌ Ampera Skate Park (0,67) | ❌ Ampera Skate Park (0,67) | ✅ None |
| 8 | Normal | fasilitas jakabaring | Jakabaring Sport City | ✅ Jakabaring Sport City (0,65) | ✅ Jakabaring Sport City (0,65) | ❌ None |
| 9 | Normal | harga masuk fantasy island | None | ❌ Pempek Candy (0,48) | ✅ None | ✅ None |
| 10 | Normal | info al quran al akbar | Bayt Al-Quran Al-Akbar | ✅ Bayt Al-Quran Al-Akbar (0,77) | ✅ Bayt Al-Quran Al-Akbar (0,77) | ✅ Bayt Al-Quran Al-Akbar (0,77) |
| 11 | Trap | ada fasilitas kolam renang ga disana? | None | ✅ None | ✅ None | ✅ None |
| 12 | Trap | harganya brapa? | None | ❌ Kambang Iwak Besak (0,44) | ✅ None | ✅ None |
| 13 | Trap | buka 24 jam | None | ✅ None | ✅ None | ✅ None |
| 14 | Trap | rekomendasi wisata sejarah | None | ❌ Jembatan Ampera (0,44) | ✅ None | ✅ None |
| 15 | Trap | ada tempat wisata dekat sini gak | None | ✅ None | ✅ None | ✅ None |
| 16 | Kritis | wisata kuliner palembang | None | ✅ None | ✅ None | ✅ None |
| 17 | Kritis | tempat wisata yang bagus | None | ❌ Hutan Wisata Punti Kayu (0,51) | ✅ None | ✅ None |
| 18 | Kritis | cara ke sana naik apa | None | ❌ Ampera Skate Park (0,55) | ❌ Ampera Skate Park (0,55) | ✅ None |

> **Keterangan**: ✅ = hasil sesuai *ground truth*, ❌ = hasil tidak sesuai. Skor dalam kurung = `SequenceMatcher.ratio()`.

### Penyesuaian Nama *Ground Truth* terhadap Database

| Kasus | Nama di Prompt | Nama Resmi di DB |
|:-----:|---------------|------------------|
| 3 | Benteng Kuto Besak | Kawasan Benteng Kuto Besak (BKB) |
| 6 | Taman Kambang Iwak Besak | Kambang Iwak Besak |
| 7 | Amanzi Waterpark | *(tidak ada di DB — dihapus dari ground truth positif)* |
| 9 | Fantasy Island | *(tidak ada di DB — dihapus dari ground truth positif)* |
| 10 | Al Quran Al Akbar | Bayt Al-Quran Al-Akbar |

## Tabel Ringkasan Metrik Evaluasi

| Metrik | T = 0,40 | T = 0,55 | T = 0,70 |
|--------|:--------:|:--------:|:--------:|
| TP | 8 | 8 | 6 |
| TN | 4 | 8 | 10 |
| FP | 6 | 2 | 0 |
| FN | 0 | 0 | 2 |
|--------|----------|----------|----------|
| **Precision** | 57,14% | 80,00% | 100,00% |
| **Recall** | 100,00% | 100,00% | 75,00% |
| **F1-Score** | 72,73% | 88,89% | 85,71% |

## Kasus yang Berubah Hasil antar Nilai Ambang

- **Kasus 2** — "fasilitas di punti kayu"
  - Diharapkan: Hutan Wisata Punti Kayu
  - T=0,40: Hutan Wisata Punti Kayu (skor 0,67) ✅
  - T=0,55: Hutan Wisata Punti Kayu (skor 0,67) ✅
  - T=0,70: None (skor 0,00) ❌

- **Kasus 7** — "tiket masuk amanzi waterpark"
  - Diharapkan: None
  - T=0,40: Ampera Skate Park (skor 0,67) ❌
  - T=0,55: Ampera Skate Park (skor 0,67) ❌
  - T=0,70: None (skor 0,00) ✅

- **Kasus 8** — "fasilitas jakabaring"
  - Diharapkan: Jakabaring Sport City
  - T=0,40: Jakabaring Sport City (skor 0,65) ✅
  - T=0,55: Jakabaring Sport City (skor 0,65) ✅
  - T=0,70: None (skor 0,00) ❌

- **Kasus 9** — "harga masuk fantasy island"
  - Diharapkan: None
  - T=0,40: Pempek Candy (skor 0,48) ❌
  - T=0,55: None (skor 0,00) ✅
  - T=0,70: None (skor 0,00) ✅

- **Kasus 12** — "harganya brapa?"
  - Diharapkan: None
  - T=0,40: Kambang Iwak Besak (skor 0,44) ❌
  - T=0,55: None (skor 0,00) ✅
  - T=0,70: None (skor 0,00) ✅

- **Kasus 14** — "rekomendasi wisata sejarah"
  - Diharapkan: None
  - T=0,40: Jembatan Ampera (skor 0,44) ❌
  - T=0,55: None (skor 0,00) ✅
  - T=0,70: None (skor 0,00) ✅

- **Kasus 17** — "tempat wisata yang bagus"
  - Diharapkan: None
  - T=0,40: Hutan Wisata Punti Kayu (skor 0,51) ❌
  - T=0,55: None (skor 0,00) ✅
  - T=0,70: None (skor 0,00) ✅

- **Kasus 18** — "cara ke sana naik apa"
  - Diharapkan: None
  - T=0,40: Ampera Skate Park (skor 0,55) ❌
  - T=0,55: Ampera Skate Park (skor 0,55) ❌
  - T=0,70: None (skor 0,00) ✅

## Kesimpulan

Nilai ambang **0,55** menghasilkan F1-Score tertinggi (**88,89%**) dengan Precision = 80,00% dan Recall = 100,00%.
