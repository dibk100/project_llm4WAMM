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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

def save_best_model(model, save_dir, base_name, epoch, val_loss,score):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    ckpt_dir = os.path.join(save_dir, base_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_filename = f"{timestamp}_epoch{epoch}_valLoss{val_loss:.4f}_rmse_{score:.4f}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)

    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Best model saved: {ckpt_path} (rmse_score: {score:.4f})")
    return ckpt_path