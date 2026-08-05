import json

tr_ner = json.load(open('processed/train_ner_augmented_v2.json', 'r', encoding='utf-8'))
holdouts = json.load(open('processed/ner_holdout_entities.json', 'r', encoding='utf-8'))

tr_text = ' '.join([' '.join(item['tokens']).lower() for item in tr_ner])

print(f"Total train augmented samples: {len(tr_ner)}")
for etype, hlist in holdouts.items():
    for h in hlist:
        h_low = h.lower()
        if h_low in tr_text:
            print(f"FOUND LEAK: type={etype}, val=\"{h}\"")
            for idx, item in enumerate(tr_ner):
                item_str = ' '.join(item['tokens']).lower()
                if h_low in item_str:
                    print(f"  in sample #{idx}: {item['tokens']}")
