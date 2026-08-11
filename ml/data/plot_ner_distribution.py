"""
plot_ner_distribution.py — Script visualisasi distribusi entitas dataset NER (Stage 1 / Augmentasi Terbaru)
Menghasilkan dua grafik profesional berstandar skripsi:
1. gambar_ner_total.png   : Bar chart horizontal total entitas per label (terurut dari terbesar)
2. gambar_ner_stacked.png : Stacked horizontal bar chart (Train / Val / Test)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# 1. KONFIGURASI PATH & TARGET ENTITAS
# ============================================================
TRAIN_PATH = "ml/data/processed/train_ner_aug.json"
VAL_PATH   = "ml/data/processed/val_ner_legacy.json"
TEST_PATH  = "ml/data/processed/test_ner_legacy_583.json"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_ENTITIES = ["DESTINATION", "PRICE", "TIME", "CATEGORY", "LOCATION"]

# Benchmark Acuan Dataset NER Terbaru (Train Aug / Val / Test / Total)
BENCHMARK = {
    "DESTINATION": (2211, 286, 229, 2726),
    "PRICE":       (2129, 136, 44,  2309),
    "TIME":        (711,  151, 187, 1049),
    "CATEGORY":    (692,  103, 107, 902),
    "LOCATION":    (780,  11,  16,  807),
}
BENCHMARK_TOTAL = (6523, 687, 583, 7793)


# ============================================================
# 2. FUNGSI PARSING & PENGHITUNGAN ENTITAS (B- TAGS)
# ============================================================
def extract_entity_counts(file_path):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File {file_path} tidak ditemukan!")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts = Counter()
    total_tokens = 0

    for item in data:
        tags = None
        for key in ["tags", "ner_tags", "labels"]:
            if key in item:
                tags = item[key]
                break

        if tags is None:
            continue

        tokens = None
        for key in ["tokens", "words", "text"]:
            if key in item:
                tokens = item[key]
                break
        if tokens:
            total_tokens += len(tokens)

        for tag in tags:
            tag_str = str(tag)
            if tag_str.startswith("B-"):
                ent_name = tag_str[2:].upper()
                counts[ent_name] += 1

    return counts, len(data), total_tokens


# ============================================================
# 3. VERIFIKASI DATA
# ============================================================
train_counts, train_sents, train_tokens = extract_entity_counts(TRAIN_PATH)
val_counts, val_sents, val_tokens       = extract_entity_counts(VAL_PATH)
test_counts, test_sents, test_tokens    = extract_entity_counts(TEST_PATH)

print("=" * 70)
print("🔍 TABEL VERIFIKASI DISTRIBUSI ENTITAS DATASET NER (TERBARU / AUG)")
print("=" * 70)
print(f"{'Label Entitas':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7} | {'Status':<10}")
print("-" * 70)

all_match = True
summary_data = []

for ent in ["DESTINATION", "PRICE", "TIME", "CATEGORY", "LOCATION"]:
    tr = train_counts.get(ent, 0)
    va = val_counts.get(ent, 0)
    te = test_counts.get(ent, 0)
    tot = tr + va + te

    b_tr, b_va, b_te, b_tot = BENCHMARK[ent]
    is_match = (tr == b_tr and va == b_va and te == b_te and tot == b_tot)
    if not is_match:
        all_match = False

    status = "✅ MATCH" if is_match else f"❌ MISMATCH (Acuan: {b_tot})"
    print(f"{ent:<15} | {tr:<7} | {va:<7} | {te:<7} | {tot:<7} | {status}")
    summary_data.append({
        "Label": ent,
        "Train": tr,
        "Val": va,
        "Test": te,
        "Total": tot
    })

print("-" * 70)
tot_tr = sum(train_counts.values())
tot_va = sum(val_counts.values())
tot_te = sum(test_counts.values())
tot_all = tot_tr + tot_va + tot_te

b_tot_tr, b_tot_va, b_tot_te, b_tot_all = BENCHMARK_TOTAL
tot_match = (tot_tr == b_tot_tr and tot_va == b_tot_va and tot_te == b_tot_te and tot_all == b_tot_all)
if not tot_match:
    all_match = False

status_tot = "✅ MATCH" if tot_match else "❌ MISMATCH"
print(f"{'TOTAL ENTITAS':<15} | {tot_tr:<7} | {tot_va:<7} | {tot_te:<7} | {tot_all:<7} | {status_tot}")
print(f"{'Total Kalimat':<15} | {train_sents:<7} | {val_sents:<7} | {test_sents:<7} | {train_sents+val_sents+test_sents:<7} | -")
print(f"{'Total Token':<15} | {train_tokens:<7} | {val_tokens:<7} | {test_tokens:<7} | {train_tokens+val_tokens+test_tokens:<7} | -")
print("=" * 70)

if not all_match:
    print("❌ PERINGATAN: Angka tidak cocok dengan benchmark acuan skripsi! Proses dihentikan.")
    sys.exit(1)

print("🎉 Seluruh data 100% COCOK dengan angka acuan skripsi. Memulai pembuatan gambar...\n")


# ============================================================
# 4. PENGATURAN TEMA & GAYA PLOT
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.8

df = pd.DataFrame(summary_data)
# Urutkan dari terbesar ke terkecil untuk Total
df_sorted = df.sort_values(by="Total", ascending=True)


# ============================================================
# 5. GAMBAR 1: BAR CHART HORIZONTAL TOTAL ENTITAS
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

bar_color = "#2b5c8f"
edge_color = "#1d3e60"

bars1 = ax1.barh(
    df_sorted["Label"],
    df_sorted["Total"],
    color=bar_color,
    edgecolor=edge_color,
    height=0.6,
    linewidth=0.8
)

ax1.grid(axis="x", linestyle="--", alpha=0.5, color="#cccccc")
ax1.set_axisbelow(True)

for bar in bars1:
    width = bar.get_width()
    formatted_num = f"{int(width):,}".replace(",", ".")
    ax1.text(
        width + 50,
        bar.get_y() + bar.get_height() / 2,
        formatted_num,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#222222"
    )

ax1.set_title("Distribusi Jumlah Entitas per Label NER (Dataset Final)", fontsize=13, fontweight="bold", pad=16, color="#111111")
ax1.set_xlabel("Jumlah Entitas", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax1.set_ylabel("Label Entitas NER", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax1.set_xlim(0, 3150)

ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", ".")))
plt.tight_layout()

img1_path = os.path.join(OUTPUT_DIR, "gambar_ner_total.png")
fig1.savefig(img1_path, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"✅ Gambar 1 berhasil dibuat: {img1_path}")


# ============================================================
# 6. GAMBAR 2: STACKED BAR CHART HORIZONTAL (TRAIN / VAL / TEST)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)

y_pos = np.arange(len(df_sorted))
height = 0.58

color_train = "#2b5c8f"
color_val   = "#e08214"
color_test  = "#31a354"

b_train = ax2.barh(y_pos, df_sorted["Train"], height=height, label="Data Latih (Train)", color=color_train, edgecolor="#ffffff", linewidth=0.5)
b_val   = ax2.barh(y_pos, df_sorted["Val"], left=df_sorted["Train"], height=height, label="Data Validasi (Val)", color=color_val, edgecolor="#ffffff", linewidth=0.5)
b_test  = ax2.barh(y_pos, df_sorted["Test"], left=df_sorted["Train"] + df_sorted["Val"], height=height, label="Data Uji (Test)", color=color_test, edgecolor="#ffffff", linewidth=0.5)

ax2.grid(axis="x", linestyle="--", alpha=0.5, color="#cccccc")
ax2.set_axisbelow(True)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(df_sorted["Label"], fontsize=10, fontweight="bold")

for i, total_val in enumerate(df_sorted["Total"]):
    formatted_num = f"{int(total_val):,}".replace(",", ".")
    ax2.text(
        total_val + 50,
        i,
        formatted_num,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#222222"
    )

ax2.set_title("Distribusi Entitas NER Berdasarkan Subset Data (Train / Val / Test)", fontsize=12.5, fontweight="bold", pad=16, color="#111111")
ax2.set_xlabel("Jumlah Entitas", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax2.set_ylabel("Label Entitas NER", fontsize=10.5, fontweight="bold", labelpad=8, color="#222222")
ax2.set_xlim(0, 3150)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", ".")))

ax2.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#dddddd", fontsize=9.5)
plt.tight_layout()

img2_path = os.path.join(OUTPUT_DIR, "gambar_ner_stacked.png")
fig2.savefig(img2_path, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"✅ Gambar 2 berhasil dibuat: {img2_path}")

print("\n🚀 Selesai 100%! Seluruh grafik tersimpan di folder 'output/'")
