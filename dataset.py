from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def load_data(tokenizer, train_file, test_file, batch_size, max_length):
    dataset = load_dataset("csv", data_files={"train": train_file, "test": test_file})

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized["test"], batch_size=batch_size)
    return train_loader, test_loader
