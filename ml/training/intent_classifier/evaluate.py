import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
from dataset import IntentDataset

def main():
    print("Memulai evaluasi model pada Test Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load label mapping
    try:
        with open(os.path.join(config.SAVE_MODEL_PATH, 'label2id.json'), 'r') as f:
            label2id = json.load(f)
        with open(os.path.join(config.SAVE_MODEL_PATH, 'id2label.json'), 'r') as f:
            id2label_raw = json.load(f)
            id2label = {int(k): v for k, v in id2label_raw.items()}
    except FileNotFoundError:
        print("Error: Model belum dilatih! Jalankan train.py terlebih dahulu.")
        return

    # Load Model & Tokenizer yang sudah dilatih
    tokenizer = BertTokenizerFast.from_pretrained(config.SAVE_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(config.SAVE_MODEL_PATH)
    model = model.to(device)
    model.eval()

    # Load Test Data
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    test_dataset = IntentDataset(test_df['text'].to_numpy(), test_df['label'].to_numpy(), tokenizer, config.MAX_LEN, label2id)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    true_labels = []
    pred_labels = []

    print("Melakukan inferensi pada data test...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Hitung Metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f"\n[HASIL EVALUASI AKHIR]")
    print(f"Accuracy : {acc:.4f} (Target: >= 0.85)")
    print(f"F1-Score : {f1:.4f} (Target: >= 0.82)")
    
    print("\nClassification Report per Intent:")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    # Plot Confusion Matrix
    print("\nMenggambar Confusion Matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Intent', fontsize=12)
    plt.ylabel('Actual Intent', fontsize=12)
    plt.title('Confusion Matrix - Intent Classification', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Simpan plot
    os.makedirs('ml/reports', exist_ok=True)
    report_path = 'ml/reports/intent_confusion_matrix.png'
    plt.savefig(report_path, dpi=300)
    print(f"Grafik Confusion Matrix berhasil disimpan di: {report_path}")

if __name__ == "__main__":
    main()