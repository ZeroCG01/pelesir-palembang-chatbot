import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import config
from dataset import NERDataset

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / len(data_loader)

def main():
    print("Mempersiapkan data dan model NER...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # Simpan mapping label
    with open(os.path.join(config.SAVE_MODEL_PATH, 'tag2id.json'), 'w') as f:
        json.dump(config.TAG2ID, f)
    with open(os.path.join(config.SAVE_MODEL_PATH, 'id2tag.json'), 'w') as f:
        json.dump(config.ID2TAG, f)

    # --- PERBAIKAN 1: Gunakan AutoTokenizer & AutoModel ---
    # Agar bisa ganti model (IndoBERT, mBERT, dsb.) hanya dengan ubah MODEL_NAME di config
    print(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.TAGS),
        ignore_mismatched_sizes=True  # Perlu karena num_labels berbeda dari checkpoint
    )
    model = model.to(device)

    # DataLoaders
    train_dataset = NERDataset(config.TRAIN_DATA_PATH, tokenizer, config.MAX_LEN, config.TAG2ID)
    val_dataset   = NERDataset(config.VAL_DATA_PATH,   tokenizer, config.MAX_LEN, config.TAG2ID)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)

    # --- PERBAIKAN 2: Differential learning rates ---
    # Layer BERT backbone pakai LR kecil; classifier head pakai LR lebih besar
    # Ini bantu bagian NER head belajar lebih cepat tanpa merusak representasi BERT
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and not any(nd in n for nd in no_decay)],
            "lr": config.LEARNING_RATE,
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" not in n and any(nd in n for nd in no_decay)],
            "lr": config.LEARNING_RATE,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": config.LEARNING_RATE * 5,  # Classifier head: LR 5x lebih tinggi
            "weight_decay": 0.01,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # --- PERBAIKAN 3: Warmup ratio (bukan hardcoded steps) ---
    total_steps   = len(train_loader) * config.EPOCHS
    warmup_steps  = int(total_steps * config.WARMUP_RATIO)
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # --- PERBAIKAN 4: Early stopping dengan patience lebih longgar ---
    best_val_loss     = float('inf')
    patience_counter  = 0
    history_train_loss = []
    history_val_loss = []

    print("Memulai proses training NER...\n")
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print("-" * 20)

        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss   = eval_model(model, val_loader, device)

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            model.save_pretrained(config.SAVE_MODEL_PATH)
            tokenizer.save_pretrained(config.SAVE_MODEL_PATH)
            print(f"--> Model NER terbaik disimpan! (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{config.PATIENCE})")
            if patience_counter >= config.PATIENCE:
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