import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import *
from dataset import *
from model import *
from eval import *
import math
import re

torch.cuda.empty_cache()

# trainner
def train_model(config):
    set_seed(config['seed'])

    # run_name 자동 생성
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)

    # 모델 초기화
    model_wrapper = BertRegressionModel(config['model_name'])
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    
    # 데이터셋 로딩
    
    train_dataset = get_dataset(config, tokenizer, split='train')
    scaler = train_dataset.scaler  # 저장해두기

    # 검증/테스트 (transform)
    val_dataset = get_dataset(config, tokenizer, split='val', scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model.to(config['device'])
    
    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_mse = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(config['device']) for k, v in batch.items()}
            
            # 모델 예측값 계산
            predictions = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            # loss 계산을 위해 shape 맞추기
            predictions = predictions.view(-1)
            labels = batch['labels'].view(-1)
            loss = torch.nn.MSELoss()(predictions, labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # 검증
        val_loss, val_mse, val_rmse, val_r2 = evaluate_model_val(model, val_loader, config['device'], scaler)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
        })

        # Best 모델 저장
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_best_model(
                model,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch + 1,
                val_loss=val_loss,
                score=val_rmse,
            )
    
    return scaler