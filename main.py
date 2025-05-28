import yaml
import torch
from transformers import BertTokenizer
from model import BertClassifier
from dataset import load_data
from train import train
from eval import evaluate
import datetime
import os

def main():
    model_name = "base.yaml"                    # 모델 변경시 활용
    with open(model_name) as f:
        config = yaml.safe_load(f)

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
    model_path = config["training"].get("model_path", "best_model.pt")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer
    ## 실행
    
    for epoch in range(num_epochs):
        train_avg_loss, train_f1 = train(model, train_loader, criterion, optimizer, device)
        test_f1  = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_avg_loss:.4f}, Train F1: {train_f1:.4f}, "
            f"Test F1: {test_f1 :.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            epochs_no_improve = 0
            
            today = datetime.datetime.now().strftime("%Y%m%d")  # 현재 날짜: YYYYMMDD
            ckpt_dir = os.path.join("outputs", "checkpoints")
            best_model_path = f"{model_name}_{today}_best_model_f1_{best_f1:.4f}.pth"
            ckpt_path = os.path.join(ckpt_dir, best_model_path)
            torch.save(model.state_dict(), ckpt_path)
            print(f">>> Best model saved: {best_model_path} (F1: {best_f1:.4f}%)")
        else :
            epochs_no_improve += 0
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"🎉 Training complete. Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()