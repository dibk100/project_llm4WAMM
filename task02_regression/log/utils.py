import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import os
import torch
import random
import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def compute_metrics_made(eval_pred, mean, std):
    #  print("################## compute_metrics called")  # 이 출력이 뜨는지 확인
    preds, labels = eval_pred
    preds = preds * std + mean
    labels = labels * std + mean
    
    mse = ((preds - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    return {'eval_mse': mse, 'eval_rmse': rmse, 'eval_r2': r2}

def save_best_model(model, save_dir, base_name,score):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    ckpt_dir = os.path.join(save_dir, base_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_filename = f"{timestamp}_rmse_{score:.4f}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)

    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Best model saved: {ckpt_path} (rmse_score: {score:.4f})")
    return ckpt_path

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