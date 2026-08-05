"""
Evaluasi model pada TEST SET ORGANIK (bukan sintetis).
Output: ANGKA MENTAH saja (accuracy, F1, classification_report). Tanpa interpretasi.
Jalankan setelah intent_model & ner_model tersedia (di Drive atau lokal).
"""
import os, csv, json
import torch
from torch.utils.data import Dataset, DataLoader

# ===== KONFIG PATH (sesuaikan bila perlu) =====
INTENT_MODEL_DIR = "ZeroCG/pelesir-intent" if not os.path.exists("output/intent_model") else "output/intent_model"
NER_MODEL_DIR    = "ZeroCG/pelesir-ner" if not os.path.exists("output/ner_model") else "output/ner_model"
TRAIN_CSV        = "ml/data/processed/train_intents_v2.csv"      # untuk merekonstruksi label mapping identik training
ORGANIK_INTENT   = "ml/data/processed/test_intents_organik.csv"
ORGANIK_NER      = "ml/data/processed/test_ner_organik.json"
MAX_LEN = 128
BATCH   = 16
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NER_TAGS = ["O","B-DESTINATION","I-DESTINATION","B-CATEGORY","I-CATEGORY",
            "B-LOCATION","I-LOCATION","B-TIME","I-TIME","B-PRICE","I-PRICE"]
NER_TAG2ID = {t: i for i, t in enumerate(NER_TAGS)}
NER_ID2TAG = {i: t for t, i in NER_TAG2ID.items()}


def eval_intent():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    labels_set = set()
    with open(TRAIN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels_set.add(row["label"])
    label_list = sorted(labels_set)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR).to(DEVICE)
    model.eval()

    texts, gold = [], []
    with open(ORGANIK_INTENT, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"]); gold.append(label2id[row["label"]])

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i+BATCH], add_special_tokens=True, max_length=MAX_LEN,
                      padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    names = [id2label[i] for i in range(len(label_list))]
    print("=" * 60); print("INTENT - ORGANIK (RAW)"); print("=" * 60)
    print(f"N          : {len(gold)}")
    print(f"Accuracy   : {accuracy_score(gold, preds):.4f}")
    print(f"F1 Macro   : {f1_score(gold, preds, average='macro', zero_division=0):.4f}")
    print(f"F1 Weighted: {f1_score(gold, preds, average='weighted', zero_division=0):.4f}\n")
    print(classification_report(gold, preds, labels=list(range(len(label_list))),
                                target_names=names, digits=4, zero_division=0))


class NERDataset(Dataset):
    def __init__(self, path, tok, max_len, tag2id):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tok = tok; self.max_len = max_len; self.tag2id = tag2id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]; tags = self.data[idx]["tags"]
        t = self.tok(tokens, is_split_into_words=True, padding="max_length",
                     truncation=True, max_length=self.max_len, return_tensors="pt")
        wids = t.word_ids(); prev = None; lab = []
        for wi in wids:
            if wi is None: lab.append(-100)
            elif wi != prev: lab.append(self.tag2id[tags[wi]])
            else: lab.append(-100)
            prev = wi
        return {"input_ids": t["input_ids"].flatten(),
                "attention_mask": t["attention_mask"].flatten(),
                "labels": torch.tensor(lab, dtype=torch.long)}


def eval_ner():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import classification_report as seqrep, f1_score as seqf1
    tok = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR).to(DEVICE)
    model.eval()
    ds = NERDataset(ORGANIK_NER, tok, MAX_LEN, NER_TAG2ID)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for b in loader:
            ii = b["input_ids"].to(DEVICE); am = b["attention_mask"].to(DEVICE); lb = b["labels"].numpy()
            logits = model(input_ids=ii, attention_mask=am).logits
            pr = torch.argmax(logits, dim=2).cpu().numpy()
            for i in range(len(pr)):
                ts, ps = [], []
                for j in range(len(pr[i])):
                    if lb[i][j] != -100:
                        ts.append(NER_ID2TAG[lb[i][j]]); ps.append(NER_ID2TAG[pr[i][j]])
                true_tags.append(ts); pred_tags.append(ps)
    print("=" * 60); print("NER - ORGANIK (RAW)"); print("=" * 60)
    print(f"N kalimat  : {len(true_tags)}")
    print(f"F1 seqeval : {seqf1(true_tags, pred_tags):.4f}\n")
    print(seqrep(true_tags, pred_tags, digits=4))


if __name__ == "__main__":
    eval_intent()
    print()
    eval_ner()
