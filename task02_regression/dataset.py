import json
import torch
from torch.utils.data import Dataset
import os

from sklearn.preprocessing import StandardScaler
import numpy as np

class MyDataset(Dataset):
    def __init__(self, texts, outputs, tokenizer, max_len=128, mode='train', scaler=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        outputs = np.array(outputs).reshape(-1, 1)

        if mode == 'train':
            self.scaler = StandardScaler()
            self.outputs = self.scaler.fit_transform(outputs).flatten()
        else:
            assert scaler is not None, "Validation/Test 모드에서는 scaler를 제공해야 합니다."
            self.scaler = scaler
            self.outputs = self.scaler.transform(outputs).flatten()

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
        item['labels'] = torch.tensor(output, dtype=torch.float)
        
        return item

# 2. 데이터 로딩
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # 파일 전체를 한꺼번에 파싱
    texts = [item['text'] for item in data_list]
    outputs = [float(item['output']) for item in data_list]
    return texts, outputs

def get_dataset(config, tokenizer, split='train', scaler=None):
    file_map = {
        'train': config['train_file'],
        'val': config['val_file'],
        'test': config['test_file']
    }
    json_path = os.path.join(config['data_dir'], file_map[split])
    texts, outputs = load_data(json_path)

    return MyDataset(
        texts=texts,
        outputs=outputs,
        tokenizer=tokenizer,
        max_len=config.get('max_length', 512),
        mode=split,
        scaler=scaler
    )
