import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import get_dataloaders
from train import train
from eval import evaluate
import datetime
from sklearn.metrics import f1_score
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name='resnet18', num_classes=6).to(device)                  ## 일단 resnet18으로 테스트
    train_loader, test_loader = get_dataloaders(batch_size=32,task='emotion')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_f1 = train(model, train_loader, criterion, optimizer, device)
        test_f1  = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
            f"Test F1: {test_f1 :.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            
            today = datetime.datetime.now().strftime("%Y%m%d")  # 현재 날짜: YYYYMMDD
            ckpt_dir = os.path.join("outputs", "checkpoints")
            
            best_model_path = f"{today}_best_model_f1_{best_f1:.4f}.pth"
            ckpt_path = os.path.join(ckpt_dir, best_model_path)
            torch.save(model.state_dict(), ckpt_path)
            print(f">>> Best model saved: {best_model_path} (F1: {best_f1:.4f}%)")

if __name__ == "__main__":
    main()