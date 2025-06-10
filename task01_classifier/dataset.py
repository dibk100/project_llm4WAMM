import json
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import numpy as np
import os

class MultiLabelDataset(Dataset):
    def __init__(self, json_path, labels, model_name, max_length=512):
        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # json 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label_list = item['label']

        # 멀티라벨을 바이너리 벡터로 변환 (ex: [0,1,1,0,0])
        label_vector = np.zeros(len(self.labels), dtype=int)
        for label in label_list:
            if label in self.label2id:
                label_vector[self.label2id[label]] = 1
            else:
                # 만약 라벨이 정의된 라벨셋에 없으면 무시하거나 예외 처리
                assert False, "라벨 이슈~~~"
                
        
        # 텍스트 토크나이즈
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False             # ?
        )
        
        # 딕셔너리로 필요한 텐서 반환
        return {
            'input_ids': encoded['input_ids'].squeeze(dim=0),      # (max_length,)
            'attention_mask': encoded['attention_mask'].squeeze(dim=0),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }
        
class BinaryLabelDataset(Dataset):
    def __init__(self, json_path, normal_label_name, model_name, max_length=512):
        self.normal_label_name = normal_label_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # json 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label_list = item['label']  # 원래 멀티라벨 리스트

        # "Normal"이 포함되면 1, 아니면 0
        binary_label = 1 if self.normal_label_name in label_list else 0
        
        # 토크나이즈
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(dim=0),
            'attention_mask': encoded['attention_mask'].squeeze(dim=0),
            'labels': torch.tensor(binary_label, dtype=torch.float)  # float 타입, 0.0 or 1.0
        }

def get_dataset(config, split='train'):
    file_map = {
        'train': config['train_file'],
        'val': config['val_file'],
        'test': config['test_file']
    }
    json_path = os.path.join(config['data_dir'], file_map[split])
    
    binary_bool = config['binary_bool']
    if binary_bool :
        return BinaryLabelDataset(
            json_path=json_path,
            normal_label_name="Normal",                         # 정상/비정상
            model_name=config['model_name'],
            max_length=config.get('max_length', 512)
        )
    
    else :
        return MultiLabelDataset(
        json_path=json_path,
        labels=config['labels'],
        model_name=config['model_name'],
        max_length=config.get('max_length', 512)  # optional 설정
    )