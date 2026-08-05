import pandas as pd
import matplotlib.pyplot as plt

# Load dataset intent
df_train = pd.read_csv("ml/data/processed/train_intents_v2.csv")
df_val = pd.read_csv("ml/data/processed/val_intents_v2.csv")
df_test = pd.read_csv("ml/data/processed/test_intents_v2.csv")
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

# Set global style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------------
# 1. GRAFIK DISTRIBUSI DATA LATIH (TRAIN - EXACT 500 PER KELAS)
# -------------------------------------------------------------------
intent_counts_train = df_train['label'].value_counts().reset_index()
intent_counts_train.columns = ['Kelas Intent', 'Jumlah Data']
intent_counts_train = intent_counts_train.sort_values(by='Kelas Intent', ascending=True)

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

bars1 = ax1.barh(
    intent_counts_train['Kelas Intent'], 
    intent_counts_train['Jumlah Data'], 
    color='#2b5c8f', 
    edgecolor='#1d3e60',
    height=0.65
)

# Grid & Spacing
ax1.grid(axis='x', linestyle='--', alpha=0.6)
ax1.set_axisbelow(True)

# Label angka di ujung bar
for bar in bars1:
    width = bar.get_width()
    ax1.text(
        width + 8, 
        bar.get_y() + bar.get_height() / 2,
        f'{int(width)}', 
        ha='left', 
        va='center',
        fontsize=10, 
        fontweight='bold',
        color='#222222'
    )

ax1.set_title('Distribusi Jumlah Data per Kelas Intent pada Data Latih (Train Set)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Jumlah Data (Kalimat)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Kelas Intent', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 600)
plt.tight_layout()

train_img_path = "ml/data/distribusi_intent_train_v2.png"
plt.savefig(train_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Gambar 1 (Train 500) berhasil dibuat: {train_img_path}")


# -------------------------------------------------------------------
# 2. GRAFIK DISTRIBUSI TOTAL DATASET (TRAIN + VAL + TEST)
# -------------------------------------------------------------------
intent_counts_all = df_all['label'].value_counts().reset_index()
intent_counts_all.columns = ['Kelas Intent', 'Jumlah Data']
intent_counts_all = intent_counts_all.sort_values(by='Kelas Intent', ascending=True)

fig, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

bars2 = ax2.barh(
    intent_counts_all['Kelas Intent'], 
    intent_counts_all['Jumlah Data'], 
    color='#388e3c', 
    edgecolor='#2e7d32',
    height=0.65
)

# Grid & Spacing
ax2.grid(axis='x', linestyle='--', alpha=0.6)
ax2.set_axisbelow(True)

# Label angka di ujung bar
for bar in bars2:
    width = bar.get_width()
    ax2.text(
        width + 10, 
        bar.get_y() + bar.get_height() / 2,
        f'{int(width)}', 
        ha='left', 
        va='center',
        fontsize=10, 
        fontweight='bold',
        color='#222222'
    )

ax2.set_title('Distribusi Total Data per Kelas Intent (Train + Val + Test)', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Jumlah Data (Kalimat)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Kelas Intent', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 700)
plt.tight_layout()

all_img_path = "ml/data/distribusi_intent_total_v2.png"
plt.savefig(all_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Gambar 2 (Total Data) berhasil dibuat: {all_img_path}")
