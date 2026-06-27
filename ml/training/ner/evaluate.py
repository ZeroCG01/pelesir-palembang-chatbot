import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score, accuracy_score
import config
from dataset import NERDataset

def main():
    print("Memulai evaluasi model NER pada Test Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mapping
    try:
        with open(os.path.join(config.SAVE_MODEL_PATH, 'tag2id.json'), 'r') as f:
            tag2id = json.load(f)
        with open(os.path.join(config.SAVE_MODEL_PATH, 'id2tag.json'), 'r') as f:
            id2tag_raw = json.load(f)
            id2tag = {int(k): v for k, v in id2tag_raw.items()}
    except FileNotFoundError:
        print("Error: Model NER belum dilatih!")
        return

    # Load model pakai AutoTokenizer/AutoModel agar kompatibel dengan model apapun
    tokenizer = AutoTokenizer.from_pretrained(config.SAVE_MODEL_PATH)
    model     = AutoModelForTokenClassification.from_pretrained(config.SAVE_MODEL_PATH)
    model     = model.to(device)
    model.eval()

    test_dataset = NERDataset(config.TEST_DATA_PATH, tokenizer, config.MAX_LEN, tag2id)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    true_labels = []
    pred_labels = []

    print("Melakukan inferensi ekstraksi entitas pada data test...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)

            labels      = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()

            for i in range(labels.shape[0]):
                true_seq = []
                pred_seq = []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        true_seq.append(id2tag[labels[i, j]])
                        pred_seq.append(id2tag[predictions[i, j]])
                true_labels.append(true_seq)
                pred_labels.append(pred_seq)

    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels)

    print(f"\n[HASIL EVALUASI AKHIR NER]")
    print(f"Accuracy : {acc:.4f} (Target: >= 0.85)")
    print(f"F1-Score : {f1:.4f} (Target: >= 0.82)")
    print("\nClassification Report per Entitas:")
    print(classification_report(true_labels, pred_labels))

if __name__ == "__main__":
    main()