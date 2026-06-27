import os
import pandas as pd
import nlpaug.augmenter.word as naw
import nltk

# Pastikan corpus WordNet terunduh untuk fitur sinonim
nltk.download('wordnet')
nltk.download('omw-1.4')

def main():
    # Definisi jalur file
    input_file = 'ml/data/raw/intents_bilingual.csv'
    output_file = 'ml/data/processed/intents_augmented.csv'

    # Buat folder processed jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Membaca dataset dari {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: File intents_bilingual.csv tidak ditemukan. Pastikan file ada di ml/data/raw/")
        return

    # Inisialisasi teknik augmentasi
    # 1. Synonym Replacement (Mengganti kata dengan sinonimnya)
    aug_syn = naw.SynonymAug(aug_src='wordnet')
    # 2. Random Swap (Menukar posisi 2 kata secara acak)
    aug_swap = naw.RandomWordAug(action="swap")
    # 3. Random Delete (Menghapus kata secara acak)
    aug_del = naw.RandomWordAug(action="delete")

    augmented_data = []

    print("Memulai proses augmentasi data (ini mungkin memakan waktu sebentar)...")
    
    for index, row in df.iterrows():
        text = str(row['text'])
        label = row['label']

        # 1. Simpan teks asli
        augmented_data.append({'text': text, 'label': label})

        # 2. Terapkan Augmentasi Synonym
        try:
            syn_text = aug_syn.augment(text)
            if isinstance(syn_text, list): syn_text = syn_text[0]
            if syn_text != text:
                augmented_data.append({'text': syn_text, 'label': label})
        except:
            pass

        # 3. Terapkan Augmentasi Swap
        try:
            swap_text = aug_swap.augment(text)
            if isinstance(swap_text, list): swap_text = swap_text[0]
            if swap_text != text:
                augmented_data.append({'text': swap_text, 'label': label})
        except:
            pass

        # 4. Terapkan Augmentasi Delete
        try:
            del_text = aug_del.augment(text)
            if isinstance(del_text, list): del_text = del_text[0]
            if del_text != text:
                augmented_data.append({'text': del_text, 'label': label})
        except:
            pass

    # Ubah list dictionary kembali menjadi DataFrame Pandas
    aug_df = pd.DataFrame(augmented_data)
    
    # Hapus data duplikat jika hasil augmentasi kebetulan sama dengan kalimat asli
    aug_df = aug_df.drop_duplicates(subset=['text'])

    # Simpan ke CSV baru
    aug_df.to_csv(output_file, index=False)
    
    print("-" * 50)
    print("PROSES SELESAI!")
    print(f"Data awal: {len(df)} baris")
    print(f"Data setelah augmentasi: {len(aug_df)} baris")
    print(f"Tersimpan di: {output_file}")

if __name__ == "__main__":
    main()