import os

# === MODEL & TRAINING CONFIG ===
# Diganti ke model Indonesia/multilingual yang lebih kuat untuk konteks Palembang
MODEL_NAME = "cahya/xlm-roberta-base-indonesian-NER"  # Opsi 1: BERT khusus NER Indonesia
# MODEL_NAME = "indolem/indobert-base-uncased"  # Opsi 2: IndoBERT (uncomment jika pakai ini)
# MODEL_NAME = "bert-base-multilingual-cased"   # Opsi 3: mBERT (fallback jika 2 opsi di atas error)

MAX_LEN = 128
BATCH_SIZE = 16

# Epoch lebih tinggi + patience lebih longgar agar model punya cukup waktu konvergen
EPOCHS = 30
LEARNING_RATE = 2e-5   # Sedikit lebih kecil untuk fine-tuning yang lebih stabil

# Warmup lebih panjang untuk stabilitas di awal training
WARMUP_RATIO = 0.1     # 10% dari total steps sebagai warmup (menggantikan WARMUP_STEPS hardcoded)

# Early stopping patience lebih longgar (dari 2 -> 4)
PATIENCE = 4

# === LABEL DEFINITIONS ===
TAGS = [
    "O",
    "B-DESTINATION", "I-DESTINATION",
    "B-CATEGORY",    "I-CATEGORY",
    "B-LOCATION",    "I-LOCATION",
    "B-TIME",        "I-TIME",
    "B-PRICE",       "I-PRICE"
]
TAG2ID = {tag: id for id, tag in enumerate(TAGS)}
ID2TAG = {id: tag for tag, id in TAG2ID.items()}

# === DATA PATHS ===
TRAIN_DATA_PATH = "ml/data/processed/train_ner_augmented.json"
VAL_DATA_PATH   = "ml/data/processed/val_ner.json"
TEST_DATA_PATH  = "ml/data/processed/test_ner.json"
SAVE_MODEL_PATH = "ml/saved_models/ner/"

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)