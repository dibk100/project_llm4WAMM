import yaml
import torch
from transformers import BertTokenizer
from model import BertClassifier
from dataset import load_data
from train import train
from eval import evaluate
from utils import *
import os
import wandb

# 자동 로그인
os.environ["WANDB_API_KEY"] = "your-wandb-api-key"  # 여기에 본인의 API 키 입력
wandb.login()

def main():
    model_name = "base.yaml"                    # 모델 변경시 활용
    with open(model_name) as f:
        config = yaml.safe_load(f)

    project = config["project"]
    wandb.init(project=project, config=config)
    config = wandb.config  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(config["model"]["name"])
    train_loader, test_loader = load_data(
        tokenizer,
        config["data"]["train_file"],
        config["data"]["test_file"],
        config["training"]["batch_size"],
        config["training"]["max_length"]
    )

    model = BertClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"]
    ).to(device)
    
    num_epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    labels = config["model"]["num_labels"]
    
    # Loss 함수: multi-label 여부에 따라 다르게 설정
    if labels > 2 :
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    # 초기화
    best_f1 = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        train_avg_loss, train_f1 = train(model, train_loader, criterion, optimizer, device)
        test_f1  = evaluate(model, test_loader, device)
        
        ## 기록
        wandb.log({
        "train/loss": train_avg_loss,
        "train/f1": train_f1,
        "test/f1": test_f1,
        "epoch": epoch + 1
        })

        ## 기록
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_avg_loss:.4f}, Train F1: {train_f1:.4f}, "
            f"Test F1: {test_f1 :.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            epochs_no_improve = 0
            save_best_model(model, save_dir="outputs", base_name=model_name, num_labels=labels, best_f1=best_f1)
        
        else :
            epochs_no_improve += 0
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"🎉 Training complete. Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()