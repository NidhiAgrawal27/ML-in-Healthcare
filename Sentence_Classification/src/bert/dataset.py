from pyrsistent import inc
import torch
import transformers
from torch.utils.data import Dataset, DataLoader

from utils.data_preparation import PubMedDataset


class PubMedBERTDataset(Dataset):
    def __init__(self, bert_model_name, data_path, include_index=False):
        self.data = PubMedDataset(data_path)._load_raw_data(data_path)
        self.bert_model_name = bert_model_name
        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_name)
        self.include_index = include_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.include_index:
            return sample["sentence"], sample["abstract_index"], sample["label"]
        else:
            return sample["sentence"], sample["label"]

    def collate_fn(self, batch):
        if self.include_index:
            texts, indices, labels = zip(*batch)
        else:
            texts, labels = zip(*batch)

        bert_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        if self.include_index:
            bert_input["abstract_index"] = torch.tensor(
                indices, dtype=torch.float32
            ).view(-1, 1)

        return bert_input, torch.tensor(labels)
