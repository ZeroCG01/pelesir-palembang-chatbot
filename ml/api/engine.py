import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from huggingface_hub import hf_hub_download

class ChatbotEngine:
    def __init__(self):
        print("Memuat model Intent & NER...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Intent Model dari Hugging Face Hub (XLM-RoBERTa)
        intent_path = "ZeroCG/pelesir-intent"
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_path).to(self.device)
        self.intent_model.eval()
        # Ambil custom id2label.json menggunakan HF Hub downloader
        intent_label_file = hf_hub_download(repo_id=intent_path, filename="id2label.json")
        with open(intent_label_file, 'r') as f:
            self.intent_id2label = {int(k): v for k, v in json.load(f).items()}

        # Load Temperature Scaling (Kalibrasi Confidence)
        self.temperature = 1.0  # Default: tanpa kalibrasi
        try:
            calib_file = hf_hub_download(repo_id=intent_path, filename="calibration.json")
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
                self.temperature = calib_data.get("temperature", 1.0)
            print(f"Temperature Scaling loaded: T = {self.temperature:.4f}")
        except Exception as e:
            print(f"calibration.json tidak ditemukan, menggunakan T=1.0 (tanpa kalibrasi): {e}")

        # Load NER Model dari Hugging Face Hub
        ner_path = "ZeroCG/pelesir-ner"
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_path).to(self.device)
        self.ner_model.eval()
        # Ambil custom id2tag.json menggunakan HF Hub downloader
        ner_tag_file = hf_hub_download(repo_id=ner_path, filename="id2tag.json")
        with open(ner_tag_file, 'r') as f:
            self.ner_id2tag = {int(k): v for k, v in json.load(f).items()}

    def get_intent(self, text):
        inputs = self.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            # Bagi logits dengan Temperature sebelum softmax (Kalibrasi Confidence)
            calibrated_logits = outputs.logits / self.temperature
            probs = torch.nn.functional.softmax(calibrated_logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)
            
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            
        return self.intent_id2label[pred_idx], confidence

    def get_entities(self, text):
        # Tokenisasi khusus NER
        inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        word_ids = inputs.word_ids()

        entities = {}
        current_entity = ""
        current_label = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            
            label = self.ner_id2tag[predictions[idx]]
            token = tokens[idx].replace(" ", "") # Bersihkan karakter spesial IndoBERT/RoBERTa
            
            if label == "O":
                if current_entity:
                    entities[current_label] = current_entity.strip()
                    current_entity = ""
                    current_label = None
            elif label.startswith("B-"):
                if current_entity:
                    entities[current_label] = current_entity.strip()
                current_label = label[2:]
                current_entity = token
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity += " " + token
        
        # Simpan entitas terakhir di ujung kalimat
        if current_entity:
            entities[current_label] = current_entity.strip()
        # Membersihkan artefak Tokenizer (SentencePiece) dan menggabungkan subwords
        for k, v in entities.items():
            # 1. Hapus spasi biasa buatan perulangan sebelumnya
            # 2. Ubah unicode \u2581 (karakter spasi bawaan RoBERTa) menjadi spasi asli
            # 3. Ubah underscore biasa (jaga-jaga) menjadi spasi asli
            clean_text = v.replace(" ", "").replace("\u2581", " ").replace("_", " ").strip()
            entities[k] = clean_text
        return entities

    def process_message(self, text):
        # Eksekusi Intent dan NER secara paralel
        intent, confidence = self.get_intent(text)
        entities = self.get_entities(text)
        
        return {
            "query": text,
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        }