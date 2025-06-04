import datetime
import os
import torch
import random
import numpy as np

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
