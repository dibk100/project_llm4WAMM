import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from transformers import Trainer, TrainingArguments
from model import BertRegressionModel
from dataset import *

@torch.no_grad()
def evaluate_model_val(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.MSELoss()

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        preds = outputs.squeeze()
        loss = criterion(preds, batch['labels'])
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    val_loss = total_loss / len(val_loader)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    return val_loss, mse, rmse, r2

@torch.no_grad()
def evaluate_model(config, split='test'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧪 Using device: {device}")

    model_path = config['final_path']

    # 모델 초기화 및 가중치 로드
    model_wrapper = BertRegressionModel(config['model_name'])
    model_path = os.path.join(model_path, 'pytorch_model.bin')
    state_dict = torch.load(model_path, map_location=device)
    model_wrapper.load_state_dict(state_dict)
    model_wrapper.to(device)
    
    model_wrapper.eval()

    tokenizer = model_wrapper.get_tokenizer()
    
    test_dataset = get_dataset(config, tokenizer,split="test")

    # Trainer 대신 직접 평가 루프 사용 (더 깔끔)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    all_preds, all_labels = [], []
    criterion = torch.nn.MSELoss()
    total_loss = 0

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model_wrapper(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).squeeze()
        loss = criterion(preds, batch['labels'])
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    return avg_loss, mse, rmse, r2
