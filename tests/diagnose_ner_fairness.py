"""
NER Fairness Diagnostic Script (Dual Metric, Error Taxonomy & Manual Review Flags)
Read-only script. Outputs raw diagnostic report to stdout.
"""
import os
import json
import torch
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader

# Model & Data Paths
NER_MODEL_DIR = "ZeroCG/pelesir-ner" if not os.path.exists("output/ner_model") else "output/ner_model"
ORGANIK_NER   = "ml/data/processed/test_ner_organik.json"
TRAIN_NER     = "ml/data/processed/train_ner_v2.json"

MAX_LEN = 128
BATCH   = 16
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NER_TAGS = ["O","B-DESTINATION","I-DESTINATION","B-CATEGORY","I-CATEGORY",
            "B-LOCATION","I-LOCATION","B-TIME","I-TIME","B-PRICE","I-PRICE"]
NER_TAG2ID = {t: i for i, t in enumerate(NER_TAGS)}
NER_ID2TAG = {i: t for t, i in NER_TAG2ID.items()}
ENTITY_TYPES = ["DESTINATION", "CATEGORY", "LOCATION", "TIME", "PRICE"]

PRICE_MODIFIERS = {
    "cuma", "sekitar", "kira-kira", "tarif", "tarifnyo", "keno",
    "per", "orang", "bae", "aja", "hanya", "tiket", "karcis", "bayar"
}


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


def extract_entities(tokens, tags):
    """ Extract list of dicts: {'type': etype, 'span_text': text, 'indices': set_of_idx} """
    entities = []
    curr_tokens = []
    curr_indices = []
    curr_type = None
    
    for idx, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if curr_type:
                entities.append({
                    "type": curr_type,
                    "span_text": " ".join(curr_tokens),
                    "indices": set(curr_indices)
                })
            curr_type = tag[2:]
            curr_tokens = [tok]
            curr_indices = [idx]
        elif tag.startswith("I-") and curr_type == tag[2:]:
            curr_tokens.append(tok)
            curr_indices.append(idx)
        else:
            if curr_type:
                entities.append({
                    "type": curr_type,
                    "span_text": " ".join(curr_tokens),
                    "indices": set(curr_indices)
                })
            curr_type = None
            curr_tokens = []
            curr_indices = []
            
    if curr_type:
        entities.append({
            "type": curr_type,
            "span_text": " ".join(curr_tokens),
            "indices": set(curr_indices)
        })
        
    return entities


def calc_prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    from transformers import AutoTokenizer, AutoModelForTokenClassification

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

    # Collect sentence-level extracted entities
    sentence_records = []
    for idx, (sample, p_tags) in enumerate(zip(org_data, all_pred_seqs)):
        g_tokens = sample["tokens"]
        g_tags   = sample["tags"]
        g_ents   = extract_entities(g_tokens, g_tags)
        p_ents   = extract_entities(g_tokens, p_tags)
        sentence_records.append({
            "idx": idx + 1,
            "tokens": g_tokens,
            "text": " ".join(g_tokens),
            "gold_ents": g_ents,
            "pred_ents": p_ents
        })

    # =========================================================
    # BAGIAN 1 — DUAL METRIC (STRICT vs RELAXED)
    # =========================================================
    print("=" * 80)
    print("BAGIAN 1 — DUAL METRIC EVALUATION (STRICT vs RELAXED)")
    print("=" * 80)

    strict_counts  = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    relaxed_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for rec in sentence_records:
        g_ents = rec["gold_ents"]
        p_ents = rec["pred_ents"]

        for etype in ENTITY_TYPES:
            g_typed = [e for e in g_ents if e["type"] == etype]
            p_typed = [e for e in p_ents if e["type"] == etype]

            # --- STRICT ---
            g_strict_spans = [e["indices"] for e in g_typed]
            p_strict_spans = [e["indices"] for e in p_typed]

            p_strict_matched = set()
            g_strict_matched = set()
            for gi, g_idx in enumerate(g_strict_spans):
                for pi, p_idx in enumerate(p_strict_spans):
                    if pi not in p_strict_matched and g_idx == p_idx:
                        p_strict_matched.add(pi)
                        g_strict_matched.add(gi)
                        break

            strict_tp = len(g_strict_matched)
            strict_fn = len(g_typed) - strict_tp
            strict_fp = len(p_typed) - len(p_strict_matched)

            strict_counts[etype]["tp"] += strict_tp
            strict_counts[etype]["fn"] += strict_fn
            strict_counts[etype]["fp"] += strict_fp

            # --- RELAXED ---
            g_rel_matched = set()
            p_rel_matched = set()
            for gi, ge in enumerate(g_typed):
                for pi, pe in enumerate(p_typed):
                    if pi not in p_rel_matched and (ge["indices"] & pe["indices"]):
                        g_rel_matched.add(gi)
                        p_rel_matched.add(pi)
                        break

            rel_tp = len(g_rel_matched)
            rel_fn = len(g_typed) - rel_tp
            rel_fp = len(p_typed) - len(p_rel_matched)

            relaxed_counts[etype]["tp"] += rel_tp
            relaxed_counts[etype]["fn"] += rel_fn
            relaxed_counts[etype]["fp"] += rel_fp

    # Print Table
    print(f"\n{'ENTITY TYPE':<14} | {'--- STRICT MATCH ---':<30} | {'--- RELAXED MATCH ---':<30}")
    print(f"{'':<14} | {'Prec':<7} {'Rec':<7} {'F1':<7} {'(TP/FP/FN)':<12} | {'Prec':<7} {'Rec':<7} {'F1':<7} {'(TP/FP/FN)':<12}")
    print("-" * 80)

    total_s_tp = total_s_fp = total_s_fn = 0
    total_r_tp = total_r_fp = total_r_fn = 0

    for etype in ENTITY_TYPES:
        sc = strict_counts[etype]
        rc = relaxed_counts[etype]

        sp, sr, sf = calc_prf(sc["tp"], sc["fp"], sc["fn"])
        rp, rr, rf = calc_prf(rc["tp"], rc["fp"], rc["fn"])

        total_s_tp += sc["tp"]; total_s_fp += sc["fp"]; total_s_fn += sc["fn"]
        total_r_tp += rc["tp"]; total_r_fp += rc["fp"]; total_r_fn += rc["fn"]

        s_str = f"({sc['tp']}/{sc['fp']}/{sc['fn']})"
        r_str = f"({rc['tp']}/{rc['fp']}/{rc['fn']})"

        print(f"{etype:<14} | {sp:<7.4f} {sr:<7.4f} {sf:<7.4f} {s_str:<12} | {rp:<7.4f} {rr:<7.4f} {rf:<7.4f} {r_str:<12}")

    print("-" * 80)
    msp, msr, msf = calc_prf(total_s_tp, total_s_fp, total_s_fn)
    mrp, mrr, mrf = calc_prf(total_r_tp, total_r_fp, total_r_fn)
    s_tot_str = f"({total_s_tp}/{total_s_fp}/{total_s_fn})"
    r_tot_str = f"({total_r_tp}/{total_r_fp}/{total_r_fn})"
    print(f"{'MICRO-AVG':<14} | {msp:<7.4f} {msr:<7.4f} {msf:<7.4f} {s_tot_str:<12} | {mrp:<7.4f} {mrr:<7.4f} {mrf:<7.4f} {r_tot_str:<12}")


    # =========================================================
    # BAGIAN 2 — TAKSONOMI ERROR (entity-level)
    # =========================================================
    print("\n" + "=" * 80)
    print("BAGIAN 2 — TAKSONOMI ERROR ENTITY-LEVEL")
    print("=" * 80)

    taxonomy_counts = Counter()
    taxonomy_details = []

    for rec in sentence_records:
        tokens = rec["tokens"]
        g_ents = list(rec["gold_ents"])
        p_ents = list(rec["pred_ents"])

        g_matched = set()
        p_matched = set()

        # Match Gold to best Pred
        for gi, ge in enumerate(g_ents):
            best_pi = None
            max_overlap = 0
            for pi, pe in enumerate(p_ents):
                overlap = len(ge["indices"] & pe["indices"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pi = pi

            if best_pi is not None:
                pe = p_ents[best_pi]
                g_matched.add(gi)
                p_matched.add(best_pi)

                if ge["type"] == pe["type"] and ge["indices"] == pe["indices"]:
                    cat_name = "EXACT"
                elif ge["type"] == pe["type"]:
                    cat_name = "BOUNDARY_SAMETYPE"
                else:
                    cat_name = "WRONG_TYPE"

                taxonomy_counts[cat_name] += 1
                taxonomy_details.append({
                    "sentence_idx": rec["idx"],
                    "text": rec["text"],
                    "category": cat_name,
                    "gold": (ge["type"], ge["span_text"], list(ge["indices"])),
                    "pred": (pe["type"], pe["span_text"], list(pe["indices"]))
                })
            else:
                cat_name = "MISSED (FN)"
                taxonomy_counts[cat_name] += 1
                taxonomy_details.append({
                    "sentence_idx": rec["idx"],
                    "text": rec["text"],
                    "category": cat_name,
                    "gold": (ge["type"], ge["span_text"], list(ge["indices"])),
                    "pred": None
                })

        # Spurious Predictions (FP)
        for pi, pe in enumerate(p_ents):
            if pi not in p_matched:
                cat_name = "SPURIOUS (FP)"
                taxonomy_counts[cat_name] += 1
                taxonomy_details.append({
                    "sentence_idx": rec["idx"],
                    "text": rec["text"],
                    "category": cat_name,
                    "gold": None,
                    "pred": (pe["type"], pe["span_text"], list(pe["indices"]))
                })

    print("\n--- REKAP JUMLAH TAKSONOMI ERROR ---")
    for cat_name, cnt in sorted(taxonomy_counts.items()):
        print(f"  {cat_name:20s}: {cnt:3d} entitas")

    print("\n--- DETAIL SELURUH KASUS TAKSONOMI ERROR ---")
    for item in taxonomy_details:
        print(f"\n[Sentence #{item['sentence_idx']}] Category: {item['category']}")
        print(f"  Text : \"{item['text']}\"")
        print(f"  Gold : {item['gold']}")
        print(f"  Pred : {item['pred']}")


    # =========================================================
    # BAGIAN 3 — FLAG UNTUK REVIEW MANUAL
    # =========================================================
    print("\n" + "=" * 80)
    print("BAGIAN 3 — FLAG UNTUK REVIEW MANUAL")
    print("=" * 80)

    # 3A. PRICE Modifier Boundary Flag
    print("\n--- 3A. PRICE MODIFIER-BOUNDARY CHECK ---")
    price_boundary_cases = [item for item in taxonomy_details if item["category"] == "BOUNDARY_SAMETYPE" and item["gold"][0] == "PRICE"]
    
    for item in price_boundary_cases:
        g_tokens = set(item["gold"][1].lower().split())
        p_tokens = set(item["pred"][1].lower().split())
        diff_tokens = (g_tokens ^ p_tokens)
        
        is_modifier_diff = diff_tokens.issubset(PRICE_MODIFIERS)
        flag = "PRICE_MODIFIER_BOUNDARY" if is_modifier_diff else "PRICE_OTHER_BOUNDARY"
        
        print(f"\n[{flag}] Kalimat #{item['sentence_idx']}: \"{item['text']}\"")
        print(f"  Gold PRICE: \"{item['gold'][1]}\"")
        print(f"  Pred PRICE: \"{item['pred'][1]}\"")
        print(f"  Token Pembeda: {diff_tokens}")

    # Check Training Convention for Modifier Prefixes
    tr_data = json.load(open(TRAIN_NER, encoding="utf-8"))
    cuma_price_count = 0
    cuma_all_price_phrases = []
    
    for s in tr_data:
        ents = extract_entities(s["tokens"], s["tags"])
        for e in ents:
            if e["type"] == "PRICE":
                phrase = e["span_text"].lower()
                if "cuma 20 ribu" in phrase or phrase == "cuma 20 ribu":
                    cuma_price_count += 1
                if phrase.startswith("cuma ") or phrase.startswith("sekitar "):
                    cuma_all_price_phrases.append(phrase)

    print(f"\n[Konvensi Latih Check]: Phrase 'cuma 20 ribu' / mengandung modifier prefix di train_ner_v2.json:")
    print(f"  - Frekuensi persis 'cuma 20 ribu' sebagai span PRICE: {cuma_price_count}x")
    print(f"  - Top 10 Frasa PRICE ber-prefix 'cuma/sekitar' di train: {Counter(cuma_all_price_phrases).most_common(10)}")

    # 3B. CATEGORY OOV Review
    print("\n--- 3B. CATEGORY OOV & VOCABULARY REVIEW ---")
    train_cat_phrases = set()
    train_cat_words   = set()

    for s in tr_data:
        ents = extract_entities(s["tokens"], s["tags"])
        for e in ents:
            if e["type"] == "CATEGORY":
                phrase = e["span_text"].lower()
                train_cat_phrases.add(phrase)
                for w in phrase.split():
                    train_cat_words.add(w)

    for rec in sentence_records:
        g_cats = [e for e in rec["gold_ents"] if e["type"] == "CATEGORY"]
        for gc in g_cats:
            g_phrase = gc["span_text"].lower()
            words = g_phrase.split()
            head_word = words[-1] if words else ""

            in_full_vocab = g_phrase in train_cat_phrases
            in_word_vocab = head_word in train_cat_words

            if not in_full_vocab and not in_word_vocab:
                print(f"\n[CATEGORY_OOV_REVIEW] Kalimat #{rec['idx']}: \"{rec['text']}\"")
                print(f"  Gold CATEGORY phrase: \"{g_phrase}\" (Head word: '{head_word}')")
                print(f"  Status Vocab: Full phrase in train={in_full_vocab}, Head word in train={in_word_vocab}")

    # Conflict check: Gold vs Pred for CATEGORY
    print("\n--- DAFTAR KONFLIK PREDIKSI VS GOLD KHUSUS CATEGORY ---")
    for rec in sentence_records:
        g_cats = set(e["span_text"].lower() for e in rec["gold_ents"] if e["type"] == "CATEGORY")
        p_cats = set(e["span_text"].lower() for e in rec["pred_ents"] if e["type"] == "CATEGORY")

        if g_cats != p_cats:
            print(f"\n[CATEGORY Conflict] Kalimat #{rec['idx']}: \"{rec['text']}\"")
            print(f"  Gold CATEGORY: {list(g_cats)}")
            print(f"  Pred CATEGORY: {list(p_cats)}")


if __name__ == "__main__":
    main()
