import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
import json
import numpy as np

def main():
    print("Memuat Model NLP (Intent Classification)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    intent_path = "ZeroCG/pelesir-intent"
    tokenizer = AutoTokenizer.from_pretrained(intent_path)
    model = AutoModelForSequenceClassification.from_pretrained(intent_path).to(device)
    model.eval()

    # Ambil id2label dan balikkan jadi label2id
    intent_label_file = hf_hub_download(repo_id=intent_path, filename="id2label.json")
    with open(intent_label_file, 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    label2id = {v: k for k, v in id2label.items()}
    
    num_classes = len(id2label)

    test_file = "ml/data/processed/test_intents_v2.csv"
    if not os.path.exists(test_file):
        print(f"File {test_file} tidak ditemukan!")
        return

    print(f"\nMengeksekusi {test_file}...")
    
    y_true = []
    y_probs = []
    correct_count = 0
    total_count = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        for i, row in enumerate(rows):
            text = row['text']
            true_label = row['label']
            
            if true_label not in label2id:
                continue
                
            true_idx = label2id[true_label]
            y_true.append(true_idx)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                
            y_probs.append(probs)
            
            pred_idx = np.argmax(probs)
            if pred_idx == true_idx:
                correct_count += 1
            total_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Memproses {i + 1}/{len(rows)} data...")

    print("\nMenghitung Metrik Evaluasi...")
    
    # 1. Error Rate
    error_count = total_count - correct_count
    error_rate = (error_count / total_count) * 100
    accuracy = (correct_count / total_count) * 100
    
    # 2. ROC AUC Score (One-vs-Rest, Macro average)
    try:
        roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception as e:
        roc_auc = "N/A (Error perhitungan AUC)"
        print(f"Error ROC AUC: {e}")

    print("="*50)
    print("HASIL VALIDASI NLP (INTENT CLASSIFICATION)")
    print("="*50)
    print(f"Total Data Uji  : {total_count} kalimat")
    print(f"Benar (TP+TN)   : {correct_count} kalimat")
    print(f"Salah (FP+FN)   : {error_count} kalimat")
    print("-" * 50)
    print(f"Akurasi         : {accuracy:.2f}%")
    print(f"ERROR RATE      : {error_rate:.2f}%")
    print(f"ROC AUC SCORE   : {roc_auc:.4f} (Mendekati 1.0 = Sangat Baik)")
    print("="*50)
    print("\n* Catatan untuk Dosen:")
    print("1. Nilai AUC ini menghitung probabilitas semua kelas (One-vs-Rest).")
    print("2. Angka akurasi tinggi (Error rendah) di sini adalah metrik lokal NLP.")
    print("3. Menggunakan data uji test_intents_v2.csv.")

if __name__ == "__main__":
    main()
