import datetime
import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))

def save_best_model(model, save_dir, base_name, epoch, val_loss,partial_score):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    ckpt_dir = os.path.join(save_dir, base_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_filename = f"{timestamp}_epoch{epoch}_valLoss{val_loss:.4f}_partialScore_{partial_score:.4f}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)

    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Best model saved: {ckpt_path} (partial_score: {partial_score:.4f})")
    return ckpt_path

def partial_match_score(y_true, y_pred):
    """
    커스텀 지표 : 정답 레이블(target)의 일부만 맞췄을 때 정답의 비율만큼 점수를 주는 방식(다중 레이블 문제에서 부분 정답을 인정)
    즉, "얼마나 정답 라벨을 잘 포함시켰는가?"를 보는 커스텀 정확도 지표
    """
    scores = []
    for true, pred in zip(y_true, y_pred):
        true_labels = set(np.where(true == 1)[0])
        pred_labels = set(np.where(pred == 1)[0])
        
        if not true_labels:
            scores.append(1.0 if not pred_labels else 0.0)
        else:
            intersection = len(true_labels & pred_labels)
            scores.append(intersection / len(true_labels))
    
    return np.mean(scores)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Score improved ({self.best_score:.4f} → {current_score:.4f})")
            self.best_score = current_score
            self.counter = 0

def compute_metrics_fn(eval_pred, binary_bool=False):
    logits, labels = eval_pred
    
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    if binary_bool:
        preds = logits.argmax(axis=1)
        labels = labels.astype(int)
    else:
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
    
    print(f"labels shape: {labels.shape}, preds shape: {preds.shape}")
    print(f"labels dtype: {labels.dtype}, preds dtype: {preds.dtype}")
    
    if binary_bool:
        # 이진 분류일 때는 f1_score와 accuracy만 계산
        macro_f1 = f1_score(labels, preds, average='macro')
        exact_match_acc = (labels == preds).mean()
        
        print("Validation is running (binary classification)...")
        
        return {
            'macro_f1': macro_f1,
            'exact_match_acc': exact_match_acc
        }
    else:
        # 멀티라벨 등일 때 기존대로 계산
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')
        partial_score = partial_match_score(labels, preds)
        
        if labels.ndim == 1:
            exact_match_acc = (labels == preds).mean()
        else:
            exact_match_acc = (labels == preds).all(axis=1).mean()
        
        label_wise_acc = (labels == preds).mean()
        
        print("Validation is running (multi-label classification)...")
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score': partial_score,
            'exact_match_acc': exact_match_acc,
            'label_wise_acc': label_wise_acc
        }
