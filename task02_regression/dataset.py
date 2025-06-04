import json
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, texts, outputs, tokenizer, max_len=128):
        self.texts = texts
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        output = self.outputs[idx]
        
        encoding = self.tokenizer(text,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(output, dtype=torch.float)  # 회귀라 float
        
        return item

# 2. 데이터 로딩
def load_data(json_path):
    texts = []
    outputs = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            outputs.append(float(data['output']))
    return texts, outputs

