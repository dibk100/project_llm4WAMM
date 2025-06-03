import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(model, dataloader, device, threshold=0.5):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int().cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return f1, precision, recall
