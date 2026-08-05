"""
Diagnostic Script for Organic Test Set Error Analysis (Intent & NER)
Read-only script. Outputs raw diagnostics to stdout.
"""
import os
import csv
import json
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader

# Model directories (HF Hub with local fallback)
INTENT_MODEL_DIR = "ZeroCG/pelesir-intent" if not os.path.exists("output/intent_model") else "output/intent_model"
NER_MODEL_DIR    = "ZeroCG/pelesir-ner" if not os.path.exists("output/ner_model") else "output/ner_model"

TRAIN_CSV        = "ml/data/processed/train_intents_v2.csv"
ORGANIK_INTENT   = "ml/data/processed/test_intents_organik.csv"
ORGANIK_NER      = "ml/data/processed/test_ner_organik.json"
TRAIN_NER        = "ml/data/processed/train_ner_v2.json"

MAX_LEN = 128
BATCH   = 16
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NER_TAGS = ["O","B-DESTINATION","I-DESTINATION","B-CATEGORY","I-CATEGORY",
            "B-LOCATION","I-LOCATION","B-TIME","I-TIME","B-PRICE","I-PRICE"]
NER_TAG2ID = {t: i for i, t in enumerate(NER_TAGS)}
NER_ID2TAG = {i: t for t, i in NER_TAG2ID.items()}


def extract_entities_from_tags(tokens, tags):
    """ Helper to extract list of (type, span_text, indices) from token and tag sequences """
    entities = []
    curr_tokens = []
    curr_indices = []
    curr_type = None
    
    for idx, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if curr_type:
                entities.append((curr_type, " ".join(curr_tokens), curr_indices))
            curr_type = tag[2:]
            curr_tokens = [tok]
            curr_indices = [idx]
        elif tag.startswith("I-") and curr_type == tag[2:]:
            curr_tokens.append(tok)
            curr_indices.append(idx)
        else:
            if curr_type:
                entities.append((curr_type, " ".join(curr_tokens), curr_indices))
            curr_type = None
            curr_tokens = []
            curr_indices = []
            
    if curr_type:
        entities.append((curr_type, " ".join(curr_tokens), curr_indices))
        
    return entities


# =================================================================
# BAGIAN A — DIAGNOSA KEGAGALAN INTENT "ask_category"
# =================================================================
def diagnose_intent():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

    rows = []
    with open(ORGANIK_INTENT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    texts = [r["text"] for r in rows]
    gold_labels = [r["label"] for r in rows]
    gold_ids = [label2id[l] for l in gold_labels]

    all_probs = []
    all_pred_ids = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i+BATCH], add_special_tokens=True, max_length=MAX_LEN,
                      padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_pred_ids.extend(preds)

    print("=" * 80)
    print("BAGIAN A — DIAGNOSA INTENT ('ask_category' & ALL MISCLASSIFICATIONS)")
    print("=" * 80)

    print("\n--- 1. DETAIL PREDIKSI HANYA UNTUK GOLD = 'ask_category' ---")
    ask_cat_confusions = Counter()

    for idx, (text, g_lbl, g_id, p_id, prob_arr) in enumerate(zip(texts, gold_labels, gold_ids, all_pred_ids, all_probs)):
        if g_lbl == "ask_category":
            p_lbl = id2label[p_id]
            ask_cat_confusions[p_lbl] += 1
            
            # Top-3 predictions
            top3_idx = prob_arr.argsort()[-3:][::-1]
            top3_str = ", ".join([f"{id2label[t_id]}: {prob_arr[t_id]:.4f}" for t_id in top3_idx])

            status = "OK" if g_lbl == p_lbl else "MISCLASSIFIED"
            print(f"\n[Sample #{idx+1}] Status: {status}")
            print(f"  Text     : \"{text}\"")
            print(f"  Gold     : {g_lbl}")
            print(f"  Pred     : {p_lbl}")
            print(f"  Top-3    : {top3_str}")

    print("\n--- 2. DAFTAR SELURUH MISKLASIFIKASI INTENT (GOLD != PRED) ---")
    misclass_count = 0
    for idx, (text, g_lbl, p_id) in enumerate(zip(texts, gold_labels, all_pred_ids)):
        p_lbl = id2label[p_id]
        if g_lbl != p_lbl:
            misclass_count += 1
            print(f"  [{g_lbl} -> {p_lbl}] \"{text}\"")

    print(f"\nTotal Misklasifikasi: {misclass_count} / {len(texts)}")

    print("\n--- 3. FREKUENSI CONFUSION KHUSUS GOLD 'ask_category' ---")
    for pred_target, cnt in ask_cat_confusions.most_common():
        print(f"  gold ask_category -> diprediksi {pred_target:22s}: {cnt}x")


# =================================================================
# BAGIAN B — VERIFIKASI ANOTASI + ERROR ANALYSIS NER
# =================================================================
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
            if wi is None:
                lab.append(-100)
            elif wi != prev:
                lab.append(self.tag2id[tags[wi]])
            else:
                lab.append(-100)
            prev = wi
        return {
            "input_ids": t["input_ids"].flatten(),
            "attention_mask": t["attention_mask"].flatten(),
            "labels": torch.tensor(lab, dtype=torch.long)
        }


def diagnose_ner():
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    print("\n" + "=" * 80)
    print("BAGIAN B — VERIFIKASI ANOTASI + ERROR ANALYSIS NER (CATEGORY & PRICE)")
    print("=" * 80)

    # 1. Referensi Konvensi Latih
    print("\n--- 1. BENTUK KANONIK ENTITAS DARI DATA LATIH (train_ner_v2.json) ---")
    tr_data = json.load(open(TRAIN_NER, encoding="utf-8"))
    
    cat_counts = Counter()
    price_counts = Counter()

    for s in tr_data:
        ents = extract_entities_from_tags(s["tokens"], s["tags"])
        for etype, span_text, _ in ents:
            if etype == "CATEGORY":
                cat_counts[span_text.lower()] += 1
            elif etype == "PRICE":
                price_counts[span_text.lower()] += 1

    print("\n[CATEGORY - Distinct Phrases in Train Data (All)]:")
    for phrase, cnt in cat_counts.most_common():
        print(f"  - \"{phrase}\": {cnt}x")

    print("\n[PRICE - Distinct Phrases in Train Data (Top 30)]:")
    for phrase, cnt in price_counts.most_common(30):
        print(f"  - \"{phrase}\": {cnt}x")

    # 2 & 3. Dump Anotasi Organik & Error Analysis Side-by-Side
    tok = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR).to(DEVICE)
    model.eval()

    org_data = json.load(open(ORGANIK_NER, encoding="utf-8"))
    ds = NERDataset(ORGANIK_NER, tok, MAX_LEN, NER_TAG2ID)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

    all_pred_seqs = []
    with torch.no_grad():
        for b in loader:
            ii = b["input_ids"].to(DEVICE)
            am = b["attention_mask"].to(DEVICE)
            lb = b["labels"].numpy()
            logits = model(input_ids=ii, attention_mask=am).logits
            pr = torch.argmax(logits, dim=2).cpu().numpy()
            
            for i in range(len(pr)):
                ps = []
                for j in range(len(pr[i])):
                    if lb[i][j] != -100:
                        ps.append(NER_ID2TAG[pr[i][j]])
                all_pred_seqs.append(ps)

    print("\n--- 2 & 3. SIDE-BY-SIDE ERROR ANALYSIS UNTUK KALIMAT MEMUAT CATEGORY/PRICE ---")
    
    recap = {
        "CATEGORY": {"gold_total": 0, "correct": 0, "fn": 0, "fp": 0},
        "PRICE":    {"gold_total": 0, "correct": 0, "fn": 0, "fp": 0}
    }

    # Tracking metrics overall for recap
    for idx, (sample, p_tags) in enumerate(zip(org_data, all_pred_seqs)):
        g_tokens = sample["tokens"]
        g_tags   = sample["tags"]
        sentence_str = " ".join(g_tokens)

        g_ents = extract_entities_from_tags(g_tokens, g_tags)
        p_ents = extract_entities_from_tags(g_tokens, p_tags)

        g_target_ents = [e for e in g_ents if e[0] in ("CATEGORY", "PRICE")]
        p_target_ents = [e for e in p_ents if e[0] in ("CATEGORY", "PRICE")]

        if g_target_ents or p_target_ents:
            print(f"\n[Kalimat #{idx+1}] \"{sentence_str}\"")
            print("  GOLD Target Ents :", [(e[0], e[1], e[2]) for e in g_target_ents])
            print("  GOLD All Ents    :", [(e[0], e[1]) for e in g_ents])
            print("  PRED All Ents    :", [(e[0], e[1]) for e in p_ents])

        # Compute recap for target types
        for etype in ["CATEGORY", "PRICE"]:
            g_spans = set((e[0], e[1]) for e in g_ents if e[0] == etype)
            p_spans = set((e[0], e[1]) for e in p_ents if e[0] == etype)

            correct = len(g_spans & p_spans)
            fn = len(g_spans - p_spans)
            fp = len(p_spans - g_spans)

            recap[etype]["gold_total"] += len(g_spans)
            recap[etype]["correct"]    += correct
            recap[etype]["fn"]         += fn
            recap[etype]["fp"]         += fp

    print("\n--- 4. REKAP ERROR ANALYSIS PER TIPE (CATEGORY & PRICE) ---")
    for etype in ["CATEGORY", "PRICE"]:
        r = recap[etype]
        print(f"\nTipe Entitas: {etype}")
        print(f"  Total Gold Mention : {r['gold_total']}")
        print(f"  Prediksi Benar (TP): {r['correct']}")
        print(f"  False Negative (FN): {r['fn']}")
        print(f"  False Positive (FP): {r['fp']}")


if __name__ == "__main__":
    diagnose_intent()
    diagnose_ner()
