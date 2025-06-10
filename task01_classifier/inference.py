import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from dataset import *
from utils import *
from transformers import AutoModelForSequenceClassification
import numpy as np

@torch.no_grad()
def inference_model(config, threshold=0.5):
    device = config['device']
    batch_size = config['batch_size']
    binary_bool = config['binary_bool']

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        config['best_model_path'],
        problem_type="single_label_classification" if binary_bool else "multi_label_classification",
        num_labels=2 if binary_bool else len(config['labels'])  # or config['num_labels']
    )
    model.to(device)
    model.eval()

    # test split (ground truth labels 없어도 됨)
    dataset = get_dataset(config, split='test')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_ids = []

    for batch in loader:
        # 추론 시에는 labels 없어도 됨
        if 'labels' in batch:
            batch.pop('labels')
        ids = batch.get('id', None)  # ID가 있다면 수집

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        all_logits.append(outputs.logits.cpu())
        if ids is not None:
            all_ids.append(ids)

    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > threshold).astype(int)

    # binary classification인 경우 (예: [0.6] → 1)
    if binary_bool:
        preds = preds.squeeze()  # (N,1) → (N,)

    # optional: ID 포함
    if all_ids:
        ids = torch.cat(all_ids).numpy()
        return ids, preds, probs
    else:
        return preds, probs
