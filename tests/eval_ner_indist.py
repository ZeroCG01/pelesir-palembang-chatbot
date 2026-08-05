"""
Evaluation Script for In-Distribution NER (seqeval strict)
Read-only script. Outputs raw evaluation metrics to stdout and text file.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# ================================================================
# KONFIGURASI
# ================================================================
LOCAL_PATHS = ["output/ner_model", "output/ner"]
HF_FALLBACK = "ZeroCG/pelesir-ner"

MODEL_SOURCE = HF_FALLBACK
for lp in LOCAL_PATHS:
    if os.path.exists(lp):
        MODEL_SOURCE = lp
        break

TEST_PATH = "ml/data/processed/test_ner_v2.json"
SEEN_PATH = "ml/data/processed/test_ner_seen.json"
HOLDOUT_PATH = "ml/data/processed/test_ner_holdout.json"
OUTPUT_REPORT_PATH = "output/reports/ner_indist_metrics.txt"

MAX_LEN = 128
BATCH_SIZE = 16
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

NER_TAGS = ["O","B-DESTINATION","I-DESTINATION","B-CATEGORY","I-CATEGORY",
            "B-LOCATION","I-LOCATION","B-TIME","I-TIME","B-PRICE","I-PRICE"]
NER_TAG2ID = {t: i for i, t in enumerate(NER_TAGS)}
NER_ID2TAG = {i: t for t, i in NER_TAG2ID.items()}
TARGET_ENTITIES = ["DESTINATION", "TIME", "CATEGORY", "PRICE", "LOCATION"]


class NERDataset(Dataset):
    def __init__(self, path, tok, max_len, tag2id):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tok = tok
        self.max_len = max_len
        self.tag2id = tag2id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        tags = self.data[idx]["tags"]
        t = self.tok(tokens, is_split_into_words=True, padding="max_length",
                     truncation=True, max_length=self.max_len, return_tensors="pt")
        wids = t.word_ids()
        prev = None
        lab = []
        for wi in wids:
            if wi is None: lab.append(-100)
            elif wi != prev: lab.append(self.tag2id[tags[wi]])
            else: lab.append(-100)
            prev = wi
        return {
            "input_ids": t["input_ids"].flatten(),
            "attention_mask": t["attention_mask"].flatten(),
            "labels": torch.tensor(lab, dtype=torch.long)
        }


def eval_ner_file(filepath, model, tokenizer):
    if not os.path.exists(filepath):
        return None
    ds = NERDataset(filepath, tokenizer, MAX_LEN, NER_TAG2ID)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for b in loader:
            ii = b["input_ids"].to(DEVICE)
            am = b["attention_mask"].to(DEVICE)
            lb = b["labels"].numpy()
            logits = model(input_ids=ii, attention_mask=am).logits
            pr = torch.argmax(logits, dim=2).cpu().numpy()
            
            for i in range(len(pr)):
                ts, ps = [], []
                for j in range(len(pr[i])):
                    if lb[i][j] != -100:
                        ts.append(NER_ID2TAG[lb[i][j]])
                        ps.append(NER_ID2TAG[pr[i][j]])
                true_tags.append(ts)
                pred_tags.append(ps)
                
    return true_tags, pred_tags


def main():
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    if not os.path.exists(TEST_PATH):
        print(f"ERROR: File test set tidak ditemukan: {TEST_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_SOURCE).to(DEVICE)
    model.eval()

    # 1. Main Combined Test Set Evaluation
    true_tags, pred_tags = eval_ner_file(TEST_PATH, model, tokenizer)
    
    # Parse seqeval dict report
    report_dict = classification_report(true_tags, pred_tags, output_dict=True, digits=4)
    
    total_entities = sum(report_dict[e]["support"] for e in TARGET_ENTITIES if e in report_dict)

    output_lines = [
        "================================================================",
        "EVAL NER IN-DISTRIBUTION (STRICT / seqeval) — MODEL TERBARU",
        "================================================================",
        f"Model source : {MODEL_SOURCE}",
        f"Test set     : {os.path.basename(TEST_PATH)}   (total entitas: {total_entities})",
        "----------------------------------------------------------------",
        f"{'Entitas':<13} {'Prec':<7} {'Rec':<7} {'F1':<7} {'Support':<7}",
    ]

    for ent in TARGET_ENTITIES:
        if ent in report_dict:
            d = report_dict[ent]
            p = d["precision"]
            r = d["recall"]
            f = d["f1-score"]
            sup = d["support"]
            output_lines.append(f"{ent:<13} {p:<7.2f} {r:<7.2f} {f:<7.2f} {sup:<7}")

    output_lines.append("----------------------------------------------------------------")

    # Add averages
    for avg_key, avg_label in [("micro avg", "Micro avg"), ("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")]:
        if avg_key in report_dict:
            d = report_dict[avg_key]
            p = d["precision"]
            r = d["recall"]
            f = d["f1-score"]
            sup = d["support"]
            output_lines.append(f"{avg_label:<13} {p:<7.2f} {r:<7.2f} {f:<7.2f} {sup:<7}")

    output_lines.append("----------------------------------------------------------------")

    # 2. Seen & Holdout Evaluations (if available)
    if os.path.exists(SEEN_PATH):
        s_true, s_pred = eval_ner_file(SEEN_PATH, model, tokenizer)
        s_f1 = f1_score(s_true, s_pred)
        s_report = classification_report(s_true, s_pred, output_dict=True)
        s_sup = sum(s_report[e]["support"] for e in TARGET_ENTITIES if e in s_report)
        output_lines.append(f"Seen micro-F1    : {s_f1:.4f}   (entitas: {s_sup})")

    if os.path.exists(HOLDOUT_PATH):
        h_true, h_pred = eval_ner_file(HOLDOUT_PATH, model, tokenizer)
        h_f1 = f1_score(h_true, h_pred)
        h_report = classification_report(h_true, h_pred, output_dict=True)
        h_sup = sum(h_report[e]["support"] for e in TARGET_ENTITIES if e in h_report)
        output_lines.append(f"Holdout micro-F1 : {h_f1:.4f}   (entitas: {h_sup})")

    output_lines.append("================================================================")

    report_content = "\n".join(output_lines)

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content + "\n")

    print(report_content)


if __name__ == "__main__":
    main()
