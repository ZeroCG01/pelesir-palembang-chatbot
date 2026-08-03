import os

# Konfigurasi Model & Training
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100

# Path Direktori (Relatif dari root folder 'chatbot')
TRAIN_DATA_PATH = "ml/data/processed/train_intents_v2.csv"
VAL_DATA_PATH = "ml/data/processed/val_intents_v2.csv"
TEST_DATA_PATH = "ml/data/processed/test_intents_v2.csv"
SAVE_MODEL_PATH = "ml/saved_models/intent_classifier/"

# Buat folder penyimpanan jika belum ada
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)