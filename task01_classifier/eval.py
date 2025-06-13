from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
from utils import partial_match_score
from transformers import AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F
from model import *

@torch.no_grad()
def evaluate_model_val(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        preds = torch.sigmoid(outputs.logits) > 0.5                     # 각 아웃풋에 대해 시그모이드 씌워서 계산..
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(data_loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=1)
    partial_score = partial_match_score(y_true, y_pred)
    
    exact_match_acc = (y_true == y_pred).all(axis=1).mean()
    label_wise_acc = (y_true == y_pred).mean()                              # 각 라벨별로 맞은 비율을 계산해서 평균. 즉, 전체 라벨 요소 중에서 얼마나 맞았는지 계산
    
    return avg_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise_acc

@torch.no_grad()
def evaluate_model(config, split='test', threshold=0.5):
    device = config['device']
    batch_size = config['batch_size']
    binary_bool = config['binary_bool']

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     config['best_model_path'],
    #     problem_type="single_label_classification" if binary_bool else "multi_label_classification",
    #     num_labels=2 if binary_bool else len(config['labels'])
    # )
    model = get_model(config)  # Hugging Face 모델 클래스 불러오기
    best_model_path = config['best_model_path']
    model.load_state_dict(torch.load(best_model_path))

    model.to(device)
    model.eval()

    dataset = get_dataset(config, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    all_logits, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')

        outputs = model(**batch, labels=labels)
        total_loss += outputs.loss.item()

        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()

    probs = torch.sigmoid(logits).numpy()
    preds = (probs > threshold).astype(int)
    exact_match_acc = (labels == preds).all(axis=1).mean()
    label_wise_acc = (labels == preds).mean()

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=1)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=1)
    partial_score = partial_match_score(labels, preds)

    print("✅ 최종 성능 평가\n")
    print(f"[{split}] Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Partial Match Score: {partial_score:.4f} | Exact Match Acc: {exact_match_acc:.4f} | Label Wise Acc: {label_wise_acc:.4f}")

    return