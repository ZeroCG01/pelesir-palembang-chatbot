"""
plot_intent_stacked.py — Script visualisasi distribusi dataset INTENT per kelas (Stacked Bar Horizontal)
Menghasilkan grafik profesional berstandar skripsi: gambar_intent_stacked.png (300 DPI)
"""

import os
import sys
import csv
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# 1. KONFIGURASI PATH & BENCHMARK ACUAN SKRIPSI (TABEL 4.17)
# ============================================================
TRAIN_PATH = "ml/data/processed/train_intents_v2.csv"
VAL_PATH   = "ml/data/processed/val_intents_v2.csv"
TEST_PATH  = "ml/data/processed/test_intents_v2.csv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Benchmark Acuan Resmi Tabel 4.17 Skripsi (Train / Val / Test / Total)
BENCHMARK_INTENT = {
    "ask_ticket_price":     (500, 43, 41, 584),
    "ask_destination_info": (500, 45, 29, 574),
    "ask_operating_hours":  (500, 35, 34, 569),
    "ask_location_access":  (500, 34, 28, 562),
    "ask_facilities":       (500, 31, 20, 551),
    "ask_category":         (500, 19, 18, 537),
    "ask_unrelated":        (500, 14, 14, 528),
    "greet":                (500, 18,  8, 526),
    "goodbye":              (500, 12, 11, 523),
    "provide_feedback":     (500, 10, 10, 520),
    "ask_recommendation":   (500,  9,  7, 516),
    "ask_lrt_destinations": (500,  8,  6, 514),
    "ask_hidden_gems":      (500,  6,  5, 511),
}
BENCHMARK_TOTAL = (6500, 284, 231, 7015)


# ============================================================
# 2. FUNGSI PARSING & PENGHITUNGAN DATA CSV
# ============================================================
def extract_intent_counts(file_path):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File {file_path} tidak ditemukan!")
        sys.exit(1)

    counts = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_lower = {fn.lower().strip(): fn for fn in reader.fieldnames}
        
        # Deteksi kolom label secara otomatis
        label_col = None
        for candidate in ["label", "intent", "kelas", "category"]:
            if candidate in fieldnames_lower:
                label_col = fieldnames_lower[candidate]
                break

        if not label_col:
            print(f"❌ ERROR: Gagal mendeteksi kolom label pada {file_path}. Kolom: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            lbl = row[label_col].strip()
            if lbl:
                counts[lbl] += 1

    return counts


# ============================================================
# 3. VERIFIKASI DATA TERHADAP TABEL 4.17
# ============================================================
train_counts = extract_intent_counts(TRAIN_PATH)
val_counts   = extract_intent_counts(VAL_PATH)
test_counts  = extract_intent_counts(TEST_PATH)

all_intents = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

print("=" * 80)
print("🔍 TABEL VERIFIKASI DISTRIBUSI INTENT DATASET V2 (TABEL 4.17)")
print("=" * 80)
print(f"{'Nama Kelas Intent':<25} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7} | {'Status':<10}")
print("-" * 80)

all_match = True
summary_data = []

for intent in all_intents:
    tr = train_counts.get(intent, 0)
    va = val_counts.get(intent, 0)
    te = test_counts.get(intent, 0)
    tot = tr + va + te

    if intent in BENCHMARK_INTENT:
        b_tr, b_va, b_te, b_tot = BENCHMARK_INTENT[intent]
        is_match = (tr == b_tr and va == b_va and te == b_te and tot == b_tot)
    else:
        is_match = False

    if not is_match:
        all_match = False

    status = "✅ MATCH" if is_match else f"❌ MISMATCH (Acuan: {BENCHMARK_INTENT.get(intent, 'N/A')})"
    print(f"{intent:<25} | {tr:<7} | {va:<7} | {te:<7} | {tot:<7} | {status}")
    summary_data.append({
        "Intent": intent,
        "Train": tr,
        "Val": va,
        "Test": te,
        "Total": tot
    })

print("-" * 80)
tot_tr = sum(train_counts.values())
tot_va = sum(val_counts.values())
tot_te = sum(test_counts.values())
tot_all = tot_tr + tot_va + tot_te

b_tot_tr, b_tot_va, b_tot_te, b_tot_all = BENCHMARK_TOTAL
tot_match = (tot_tr == b_tot_tr and tot_va == b_tot_va and tot_te == b_tot_te and tot_all == b_tot_all)
if not tot_match:
    all_match = False

status_tot = "✅ MATCH" if tot_match else "❌ MISMATCH"
print(f"{'TOTAL (13 KELAS)':<25} | {tot_tr:<7} | {tot_va:<7} | {tot_te:<7} | {tot_all:<7} | {status_tot}")
print("=" * 80)

if not all_match:
    print("❌ PERINGATAN: Angka tidak cocok dengan benchmark acuan Tabel 4.17! Proses dihentikan.")
    sys.exit(1)

print("🎉 Seluruh data 100% COCOK dengan angka acuan Tabel 4.17. Memulai pembuatan gambar...\n")


# ============================================================
# 4. PENGATURAN TEMA & GAYA PLOT
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.8

df = pd.DataFrame(summary_data)
# Urutkan berdasarkan total: ascending=True agar di barh nilai terbesar berada di paling atas
df_sorted = df.sort_values(by="Total", ascending=True).reset_index(drop=True)


# ============================================================
# 5. PEMBUATAN GRAFIK: STACKED BAR CHART HORIZONTAL
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

y_pos = np.arange(len(df_sorted))
height = 0.62

# Palet warna profesional netral & selaras dengan grafik NER
color_train = "#2b5c8f"  # Deep Steel Blue
color_val   = "#e08214"  # Warm Amber / Orange
color_test  = "#31a354"  # Soft Forest Green

# Plot segmen bertumpuk (Train, Val, Test)
ax.barh(
    y_pos, df_sorted["Train"], 
    height=height, 
    label="Data Latih (Train)", 
    color=color_train, 
    edgecolor="#ffffff", 
    linewidth=0.6
)
ax.barh(
    y_pos, df_sorted["Val"], 
    left=df_sorted["Train"], 
    height=height, 
    label="Data Validasi (Val)", 
    color=color_val, 
    edgecolor="#ffffff", 
    linewidth=0.6
)
ax.barh(
    y_pos, df_sorted["Test"], 
    left=df_sorted["Train"] + df_sorted["Val"], 
    height=height, 
    label="Data Uji (Test)", 
    color=color_test, 
    edgecolor="#ffffff", 
    linewidth=0.6
)

# Grid tipis sumbu-x
ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cccccc")
ax.set_axisbelow(True)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted["Intent"], fontsize=9.5, fontweight="bold")

# Label angka total di ujung setiap bar (dengan pemisah ribuan titik bila ada)
for i, total_val in enumerate(df_sorted["Total"]):
    formatted_num = f"{int(total_val):,}".replace(",", ".")
    ax.text(
        total_val + 8,
        i,
        formatted_num,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#222222"
    )

# Judul dan Label Sumbu
ax.set_title("Distribusi Jumlah Data per Kelas Intent (Latih, Validasi, dan Uji)", fontsize=12, fontweight="bold", pad=15, color="#111111")
ax.set_xlabel("Jumlah Data (Kalimat)", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax.set_ylabel("Kelas Intent", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax.set_xlim(0, 680)

# Format ticks ribuan pada sumbu x
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", ".")))

# Legenda di posisi optimal bawah kanan
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#dddddd", fontsize=9)
plt.tight_layout()

# Simpan ke folder output/ dan ml/data/
img_output_path = os.path.join(OUTPUT_DIR, "gambar_intent_stacked.png")
fig.savefig(img_output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

shutil.copy(img_output_path, os.path.join("ml/data", "gambar_intent_stacked.png"))

print(f"✅ Grafik Stacked Intent berhasil dibuat: {img_output_path}")
print(f"✅ Salinan disimpan di: ml/data/gambar_intent_stacked.png")
print("\n🚀 Selesai 100%!")
