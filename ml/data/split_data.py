import pandas as pd
import json
from sklearn.model_selection import train_test_split

def split_intents():
    input_file = 'ml/data/processed/intents_augmented.csv'
    print(f"Membaca data intent dari {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: File intents_augmented.csv tidak ditemukan.")
        return

    # Memisahkan 80% Train dan 20% Sisa (Val + Test)
    # stratify=df['label'] memastikan distribusi 12 intent tetap seimbang
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Memisahkan 20% Sisa menjadi 10% Val dan 10% Test (setengah-setengah dari temp_df)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Simpan hasilnya
    train_df.to_csv('ml/data/processed/train_intents.csv', index=False)
    val_df.to_csv('ml/data/processed/val_intents.csv', index=False)
    test_df.to_csv('ml/data/processed/test_intents.csv', index=False)

    print("\n[INTENT CLASSIFICATION SPLIT SUCESS]")
    print(f"Train: {len(train_df)} sampel")
    print(f"Val  : {len(val_df)} sampel")
    print(f"Test : {len(test_df)} sampel")

def split_ner():
    input_file = 'ml/data/raw/ner_dataset.json'
    print(f"\nMembaca data NER dari {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File ner_dataset.json tidak ditemukan.")
        return

    # Split NER data (80% Train, 10% Val, 10% Test)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Simpan hasilnya
    with open('ml/data/processed/train_ner.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    with open('ml/data/processed/val_ner.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    with open('ml/data/processed/test_ner.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    print("\n[NER DATASET SPLIT SUCCESS]")
    print(f"Train: {len(train_data)} sampel")
    print(f"Val  : {len(val_data)} sampel")
    print(f"Test : {len(test_data)} sampel")

if __name__ == "__main__":
    split_intents()
    split_ner()
    print("\nSemua data berhasil diproses dan siap untuk tahap Training!")