import yaml
import torch
from transformers import BertTokenizer
from model import BertForMultiLabelClassification  
from dataset import MultiLabelDatasetProcessor     
from train import train
from eval import evaluate
from utils import save_best_model
import os
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import random_split

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
    
    train_processor = MultiLabelDatasetProcessor(
        json_path=config["data"]["train_file"],
        label_list=config["model"]["labels"],   # 문자열 라벨 리스트
        tokenizer_name=config["model"]["name"],
        max_length=config["training"]["max_length"]
    )

    train_dataset = train_processor.get_dataset()
    
    torch.manual_seed(42)

    total_len = len(train_dataset)
    val_len = int(total_len * 0.2) # 20%를 validation으로 사용
    train_len = total_len - val_len

    train_subset, val_subset = random_split(train_dataset, [train_len, val_len])
         
    train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"])

    model = BertForMultiLabelClassification(
        model_name=config["model"]["name"],
        num_labels=len(config["model"]["labels"])
    ).to(device)

    num_epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    num_labels = len(config["model"]["labels"])
    
    # Loss 함수:
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    # 초기화
    best_f1 = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        train_avg_loss, train_f1 = train(model, train_loader, optimizer, device,num_epochs)
        test_f1  = evaluate(model, val_loader, device)
        
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
            save_best_model(model, save_dir="outputs", base_name=model_name, num_label=labels, best_f1=best_f1)
        
        else :
            epochs_no_improve += 0
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"🎉 Training complete. Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()