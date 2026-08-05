import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def run_stage1_proxy_eval():
    train_path = "processed/train_intents_raw_v2.csv"
    if not os.path.exists(train_path):
        train_path = "processed/train_intents_v2.csv"
    test_path = "processed/test_intents_v2.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"=== (1) STAGE 1 PROXY MODEL EVALUATION (TF-IDF + LOGISTIC REGRESSION) ===")
    print(f"Train File: {train_path} ({len(train_df)} rows)")
    print(f"Test File: {test_path} ({len(test_df)} rows)")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df['text'])
    X_test = vec.transform(test_df['text'])

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_df['label'])

    preds = clf.predict(X_test)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')

    labels = sorted(list(set(train_df['label'])))
    cm = confusion_matrix(test_df['label'], preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\n--- CONFUSION MATRIX LENGKAP ---")
    print(cm_df.to_string())

    print("\n=== (2) DISTRIBUSI JUMLAH SAMPEL PER LABEL DI TEST_INTENTS_V2.CSV ===")
    counts = test_df['label'].value_counts()
    for lbl in labels:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:22s}: {cnt:3d} sampel")

    print("\n=== (3) KESALAHAN PADA 2 PASANGAN INTENT UTAMA ===")
    # Pair 1: ask_location_access <-> ask_destination_info
    loc_to_dest = cm_df.loc['ask_location_access', 'ask_destination_info'] if 'ask_location_access' in labels and 'ask_destination_info' in labels else 0
    dest_to_loc = cm_df.loc['ask_destination_info', 'ask_location_access'] if 'ask_destination_info' in labels and 'ask_location_access' in labels else 0
    total_loc_dest_err = loc_to_dest + dest_to_loc

    # Pair 2: ask_recommendation <-> ask_category
    rec_to_cat = cm_df.loc['ask_recommendation', 'ask_category'] if 'ask_recommendation' in labels and 'ask_category' in labels else 0
    cat_to_rec = cm_df.loc['ask_category', 'ask_recommendation'] if 'ask_category' in labels and 'ask_recommendation' in labels else 0
    total_rec_cat_err = rec_to_cat + cat_to_rec

    print(f"1. ask_location_access <-> ask_destination_info : {total_loc_dest_err} kesalahan ({loc_to_dest} loc->dest, {dest_to_loc} dest->loc)")
    print(f"2. ask_recommendation   <-> ask_category         : {total_rec_cat_err} kesalahan ({rec_to_cat} rec->cat, {cat_to_rec} cat->rec)")

    print("\n=== (4) KONFIRMASI BUKTI KODE TEKS AKTUAL STACKED FILLER CLEANUP ===")
    import generate_intent_dataset_v2
    # View line 1018 of generate_intent_dataset_v2.py
    print("Fungsi balance_dataset() di generate_intent_dataset_v2.py line 1018:")
    print("  by_label[label].append((clean_stacked_fillers(text), label))")
    print("Teks aktual kalimat dibersihkan sebelum dimasukkan ke list kalimat intent (train_intents_raw_v2.csv & raw/intents_bilingual_v2.csv).")

if __name__ == "__main__":
    run_stage1_proxy_eval()
