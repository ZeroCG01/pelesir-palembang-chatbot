import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import config
from dataset import IntentDataset

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.shape[0]

        loss.backward()
        # Gradient Clipping untuk mencegah exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

    return correct_predictions / total_predictions, total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.shape[0]

    return correct_predictions / total_predictions, total_loss / len(data_loader)

def main():
    print("Mempersiapkan data dan model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # Load Data
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    # Buat Mapping Label ke ID
    unique_labels = train_df['label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Simpan Mapping Label
    with open(os.path.join(config.SAVE_MODEL_PATH, 'label2id.json'), 'w') as f:
        json.dump(label2id, f)
    with open(os.path.join(config.SAVE_MODEL_PATH, 'id2label.json'), 'w') as f:
        json.dump(id2label, f)

    # Inisialisasi Tokenizer & Model mBERT
    tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, 
        num_labels=len(unique_labels)
    )
    model = model.to(device)

    # Setup DataLoader
    train_dataset = IntentDataset(train_df['text'].to_numpy(), train_df['label'].to_numpy(), tokenizer, config.MAX_LEN, label2id)
    val_dataset = IntentDataset(val_df['text'].to_numpy(), val_df['label'].to_numpy(), tokenizer, config.MAX_LEN, label2id)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Setup Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=total_steps)

    # Training Loop dengan Early Stopping
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    history_train_loss = []
    history_val_loss = []

    print("Memulai proses training...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 15)

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, val_loader, device)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Simpan model terbaik
            model.save_pretrained(config.SAVE_MODEL_PATH)
            tokenizer.save_pretrained(config.SAVE_MODEL_PATH)
            print(f"--> Model terbaik disimpan!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping terpicu! Menghentikan training untuk mencegah overfitting.")
                break
    # === TAMBAHKAN KODE PLOTTING INI DI AKHIR FUNGSI MAIN ===
    print("\nMenyimpan grafik Loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label='Training Loss', marker='o')
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label='Validation Loss', marker='o')
    plt.title('Kurva Training dan Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Simpan sebagai file gambar (PNG)
    plt.savefig(os.path.join(config.SAVE_MODEL_PATH, 'loss_curve.png'))
    plt.close()
    print("Grafik berhasil disimpan sebagai 'loss_curve.png'!")
    # Print ringkasan
    print("\n===== RINGKASAN TRAINING ====")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Epoch terakhir: {epoch + 1}")
    print(f"Model disimpan di: {config.SAVE_MODEL_PATH}")
    
if __name__ == "__main__":
    main()