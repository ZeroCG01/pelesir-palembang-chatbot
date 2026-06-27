import json
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, tag2id):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tokens = self.data[item]['tokens']
        tags = self.data[item]['tags']

        # Tokenisasi dengan is_split_into_words=True karena data kita berupa list of words
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Alignment logic (Word-piece alignment untuk label BIO)
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Token spesial seperti [CLS], [SEP], [PAD] diberi label -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Kata pertama dari sebuah token diberi label sesuai tags
                label_ids.append(self.tag2id[tags[word_idx]])
            else:
                # Pecahan kata (subwords) diberi label -100 agar diabaikan saat loss calculation
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': tokenized_inputs['input_ids'].flatten(),
            'attention_mask': tokenized_inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }