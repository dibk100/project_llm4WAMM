import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from transformers import Trainer, TrainingArguments
from model import BertRegressionModel
from dataset import *

@torch.no_grad()
def evaluate_model_val(model, val_loader, device, scaler):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'], batch['attention_mask'])
            preds.extend(outputs.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    preds = np.array(preds).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)

    # 🔁 inverse transform to original scale
    preds = scaler.inverse_transform(preds).flatten()
    labels = scaler.inverse_transform(labels).flatten()

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)
    loss = mse  # 회귀에서는 보통 MSE 기준

    return loss, mse, rmse, r2

@torch.no_grad()
def evaluate_model(config, split='test'):
    import joblib
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧪 Using device: {device}")

    model_path = os.path.join(config['save_path'],config['model_name'],config['best_model_path'])
    scaler_path = os.path.join(config['save_path'], 'bert-base-uncased','scaler.pkl')
    assert os.path.exists(scaler_path), "❌ scaler.pkl not found. Make sure training saved it."

    # 모델 초기화 및 가중치 로드
    model_wrapper = BertRegressionModel(config['model_name'])
    model_wrapper.load_state_dict(torch.load(model_path, map_location=device))
    model_wrapper.to(device)
    model_wrapper.eval()

    tokenizer = model_wrapper.get_tokenizer()

    # 데이터 세팅
    scaler = joblib.load(scaler_path)
    test_dataset = get_dataset(config, tokenizer, split=split, scaler=scaler)

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

    # 🔁 정규화 해제 (inverse transform)
    scaler = joblib.load(scaler_path)
    all_preds = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_labels = scaler.inverse_transform(np.array(all_labels).reshape(-1, 1)).flatten()

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    return avg_loss, mse, rmse, r2