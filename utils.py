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

def save_best_model(model, save_dir, base_name, epoch, val_loss):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    ckpt_dir = os.path.join(save_dir, base_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_filename = f"{base_name}_epoch{epoch}_valLoss{val_loss:.4f}_{timestamp}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)

    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Best model saved: {ckpt_path} (val_loss: {val_loss:.4f})")
