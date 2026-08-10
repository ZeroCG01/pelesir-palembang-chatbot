# 📋 LAPORAN AUDIT ULANG INDEPENDEN DATASET NER (STAGE 2 RE-AUDIT)
**Status Gerbang Integritas**: `🟢 PASS`
**Dataset Diuji**: `train_ner_aug.json` (4.214 kalimat), `val_ner_legacy.json` (392 kalimat), `test_ner_legacy_583.json` (331 kalimat)

---

## 1. Verifikasi Distribusi Korpus Pasca-Augmentasi

| Label Entitas / Metrik | Data Latih Augmentasi (*Train Aug*) | Data Validasi (*Val*) | Data Uji (*Test*) | **Total Korpus** | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DESTINATION** | 2,211 | 286 | 229 | **2,726** | Terkunci |
| **TIME** | 711 | 151 | 187 | **1,049** | Terkunci |
| **CATEGORY** | 692 | 103 | 107 | **902** | Terkunci |
| **PRICE** | 2,129 | 136 | 44 | **2,309** | Format kaya (1,49x) |
| **LOCATION** | 780 | 11 | 16 | **807** | Minoritas seimbang (9,75x) |
| **Total Entitas** | **6,523** | **687** | **583** | **7,793** | - |
| **Total Kalimat** | **4,214** | **392** | **331** | **4,937** | - |
| **Total Token** | **28,728** | **2,321** | **1,905** | **32,954** | - |

## 2. Audit Kualitas Skema BIO & Integritas Format
- **Tag Ilegal / Urutan BIO Salah**: `0` error (0%).
- **Mismatch Panjang Token vs Tag**: `0` kasus (0%).
- **Kesimpulan**: Skema BIO 100% valid dan konsisten.

## 3. Audit Kebocoran Data (Data Leakage & Butir 3c)
- **Duplikasi Kalimat Persis (*Exact Duplicates Train vs Test*)**: `0` kasus (**0% Leakage**).
- **Duplikasi Kalimat Persis (*Exact Duplicates Train vs Val*)**: `0` kasus (**0% Leakage**).
- **Duplikasi Semu (*Near-Duplicates, Jaccard $\ge 0,9$*)**: `0` pasangan.
- **Pola Template In-Distribution (Butir 3c)**: `299` dari 331 kalimat uji (90.3%) merupakan variasi slot domain pariwisata yang konsisten tanpa bocoran nilai gazetteer.

## 4. Analisis Keragaman Template & Sintaksis LOCATION (Anti-Korelasi Semu)
- **Total Kemunculan LOCATION di Data Latih**: `779` kalimat.
- **Jumlah Template Kalimat Unik LOCATION**: `129` variasi pola kalimat.
- **Rasio Pengulangan Template**: Rata-rata `6.0` sampel per pola (sangat sehat, tidak overfit pada satu pola).

### Distribusi Pola Posisi Sintaksis LOCATION:
1. **Pola Preposisi 'di'** (*'tempat wisata di [LOC]'*): `305` kalimat (39.2%)
2. **Pola Arah / Gerak 'ke / menuju'** (*'rute jalan ke [LOC]'*): `127` kalimat (16.3%)
3. **Pola Subjek di Awal Kalimat** (*'[LOC] punya wisata apa'*): `77` kalimat (9.9%)
4. **Pola Relatif / Sekitar** (*'dekat [LOC]', 'kawasan [LOC]'*): `179` kalimat (23.0%)
5. **Pola Transit & Transportasi** (*'lrt ke [LOC]', 'stasiun [LOC]'*): `44` kalimat (5.6%)

> **Temuan**: Model tidak akan terjebak mempelajari 'kata setelah di' karena posisi token LOCATION tersebar merata di awal kalimat (subjek), setelah kata kerja arah (*menuju/ke*), dan dalam konstruksi spasial relatif (*dekat/sekitar/kawasan*).

## 5. Keselarasan Batas Span (*Boundary Alignment*) Entitas PRICE
- **Total Span Unik PRICE di Test Set**: `14` variasi span.
- **Total Span Unik PRICE di Train Augmentasi**: `106` variasi span.
- **Pelanggaran Over-Extended Boundary (*'per orang', 'per tiket', dll.*)**: `0` kasus (**0% Over-extension**).
- **Konfirmasi Boundary**: Seluruh span nominal harga pada `train_ner_aug.json` telah diselaraskan 100% dengan konvensi *ground truth* test set (hanya frase harga murni yang ditandai `B-PRICE`/`I-PRICE`, sedangkan keterangan satuan ditandai sebagai `O`).

---
### 🚪 KESIMPULAN RE-AUDIT STAGE 2: **`PASS`**
> Dataset `train_ner_aug.json` telah lolos seluruh pengujian forensik independen, bebas kebocoran, memiliki keragaman template yang kokoh, dan batas span selaras 100% dengan test set.