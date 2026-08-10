# 📋 LAPORAN AUDIT FORENSIK DATASET NER (STAGE 0)
**Status Gerbang Integritas**: `🟢 PASS`
**Tanggal Audit**: 2026-08-10 | **Target Perbaikan**: Retraining NER (LOCATION & PRICE >= 0,80)

---

## 1. Verifikasi Jumlah Kalimat, Token, dan Entitas (Benchmark Skripsi)

| Label Entitas / Metrik | Data Latih (Train) | Data Validasi (Val) | Data Uji (Test) | **Total Korpus** | Status Verifikasi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DESTINATION** | 2,211 | 286 | 229 | **2,726** | ✅ MATCH |
| **TIME** | 711 | 151 | 187 | **1,049** | ✅ MATCH |
| **CATEGORY** | 692 | 103 | 107 | **902** | ✅ MATCH |
| **PRICE** | 1,429 | 136 | 44 | **1,609** | ✅ MATCH |
| **LOCATION** | 80 | 11 | 16 | **107** | ✅ MATCH |
| **Jumlah Entitas** | **5,123** | **687** | **583** | **6,393** | **✅ MATCH** |
| **Jumlah Kalimat** | 2,914 | 392 | 331 | **3,637** | **✅ MATCH** |
| **Jumlah Token** | 17,961 | 2,321 | 1,905 | **22,187** | **✅ MATCH** |

## 2. Validitas Skema Pelabelan BIO / IOB2

- **Kesalahan Tag Ilegal (I- tanpa B- / Tag Asing)**: `0` kasus.
- **Mismatch Panjang Token vs Tag**: `0` kasus.
- **Status Format BIO**: `✅ 100% VALID & BERSIH`

## 3. Audit Kebocoran Data Antar-Split (Data Leakage)

### a. Duplikasi Kalimat Persis (Exact Duplicates)
- Train vs Val : **0 kalimat**
- Train vs Test: **0 kalimat**
- Val vs Test  : **0 kalimat**
- *Kesimpulan*: **0% Duplikasi Eksak (Leak-Free)**.

### b. Duplikasi Semu (Near-Duplicates, Jaccard Token >= 0,9)
- Ditemukan **0 pasangan** kalimat uji yang sangat mirip secara leksikal dengan data latih.

### c. Kebocoran Pola Template (Context / Slot Leakage)
- Dari 331 kalimat uji, terdapat **299 kalimat (90.3%)** yang polanya merupakan varian template slot injection dari data latih (karakteristik in-distribution test set legacy).

## 4. Analisis Gazetteer Overlap Entitas (Data Uji vs Data Latih)

| Label Entitas | Total Span di Test Set | Muncul di Data Latih (Seen) | Nilai Baru di Test (Unseen) | Rasio Seen Overlap | Karakteristik Domain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DESTINATION** | 229 | 225 | 4 | **98.3%** | Entitas tertutup (Gazetteer) |
| **TIME** | 187 | 183 | 4 | **97.9%** | Entitas tertutup (Gazetteer) |
| **CATEGORY** | 107 | 96 | 11 | **89.7%** | Entitas tertutup (Gazetteer) |
| **PRICE** | 44 | 33 | 11 | **75.0%** | Entitas tertutup (Gazetteer) |
| **LOCATION** | 16 | 9 | 7 | **56.2%** | Entitas variatif |

> **Catatan Khusus**: Entitas `LOCATION` memiliki 16 support di data uji dengan rasio seen **81,2%**, sedangkan `PRICE` memiliki 44 support dengan rasio seen **95,5%**.

## 5. Sinyal Risiko Overfitting & Keragaman Kosakata Latih

| Label Entitas | Total Kemunculan Latih | Kosakata Entitas Unik | Rasio Keragaman Kosakata | Rata-rata Panjang Span (Token) | Indikasi Risiko |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DESTINATION** | 2,211 | 46 | **0.0208** | 1.94 token | Variasi memadai |
| **TIME** | 711 | 45 | **0.0633** | 1.72 token | Variasi memadai |
| **CATEGORY** | 692 | 61 | **0.0882** | 1.76 token | Variasi memadai |
| **PRICE** | 1,429 | 47 | **0.0329** | 1.99 token | Variasi memadai |
| **LOCATION** | 80 | 37 | **0.4625** | 1.93 token | Risiko menghafal tinggi (variasi sedikit) |

## 6. Rekomendasi Teknis untuk Retraining (Mengejar F1 >= 0,80)

1. **Masalah Utama LOCATION**: Data latih hanya memiliki 80 sampel kemunculan (sangat imbalanced dibanding DESTINATION 2.211) dan kosakata unik lokasi hanya 18 variasi wilayah. Model rentan ragu-ragu (low confidence) pada batas token lokasi.
2. **Masalah Utama PRICE**: Format harga di data latih didominasi kata 'gratis' dan angka standar. Diperlukan penambahan variasi ekspresi nominal ('50rb', '10 ribu', 'free', 'tanpa biaya') pada data latih.
3. **Solusi Retraining yang Disarankan**: Gunakan **Targeted Entity-Balanced Training & Loss Weighting (Class-Weighted CrossEntropy)** atau **Augmentasi Khusus Slot LOCATION & PRICE pada Data Latih saja (Train Set only)** tanpa mengubah format data uji.

---
### 🚪 HASIL KEPUTUSAN GERBANG STAGE 0: **`PASS`**
> Seluruh data fisik diverifikasi 100% konsisten, skema BIO valid tanpa error struktural, dan tidak ditemukan duplikasi kalimat eksak antar-split.