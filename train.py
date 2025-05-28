from sklearn.metrics import f1_score

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        inputs, labels = batch[0], batch[1]
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        # print("Input shape:", inputs.shape)
        # print("Input dtype:", inputs.dtype)
        
        # raise ValueError(">> 뭐가 문제일까")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')  # 또는 'weighted'
    return avg_loss, f1