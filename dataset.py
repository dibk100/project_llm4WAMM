import json
from datasets import Dataset
from transformers import BertTokenizer
from typing import List, Dict

class MultiLabelDatasetProcessor:
    def __init__(self, json_path: str, label_list: List[str], tokenizer_name: str = "bert-base-uncased", max_length: int = 256):
        self.json_path = json_path
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(label_list)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_data(self) -> List[Dict]:
        with open(self.json_path, "r") as f:
            data = json.load(f)
        return data

    def one_hot_encode(self, labels: List[str]) -> List[int]:
        vec = [0] * self.num_labels
        for label in labels:
            if label in self.label2id:
                vec[self.label2id[label]] = 1
        return vec

    def preprocess(self, example: Dict) -> Dict:
        encoding = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        encoding["labels"] = self.one_hot_encode(example["label"])
        return encoding

    def get_dataset(self) -> Dataset:
        raw_data = self.load_data()
        dataset = Dataset.from_list(raw_data)
        dataset = dataset.map(self.preprocess)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset
