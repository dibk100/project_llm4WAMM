from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset
from model import get_model

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

        preds = torch.sigmoid(outputs.logits) > 0.5
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(data_loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return avg_loss, macro_f1, micro_f1

@torch.no_grad()
def evaluate_model(config, split='test', threshold=0.5):
    device = config['device']
    batch_size = config['batch_size']

    # 모델 불러오기 및 가중치 로드
    model = get_model(config)
    model.load_state_dict(torch.load(config['best_model_path'], map_location=device))
    model.to(device)
    model.eval()

    # 데이터셋과 DataLoader 준비
    dataset = get_dataset(config, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        preds = torch.sigmoid(outputs.logits) > threshold
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"[{split}] Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
    return avg_loss, macro_f1, micro_f1