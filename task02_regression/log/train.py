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

# # trainner
# def train_model(config):
#     set_seed(config['seed'])

#     # run_name 자동 생성
#     run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
#     wandb.init(project=config['wandb_project'], name=run_name, config=config)

#     # 모델 초기화
#     model_wrapper = BertRegressionModel(config['model_name'])
#     model = model_wrapper.get_model()
#     tokenizer = model_wrapper.get_tokenizer()
    
#     # 데이터셋 로딩
#     train_dataset = get_dataset(config, tokenizer,split='train')
#     val_dataset = get_dataset(config, tokenizer, split='val')

#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

#     model.to(config['device'])
    
#     optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
#     num_training_steps = config['epochs'] * len(train_loader)
#     lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#     best_val_mse = float('inf')
    
#     for epoch in range(config['epochs']):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             batch = {k: v.to(config['device']) for k, v in batch.items()}
#             predictions = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
#             loss = torch.nn.MSELoss()(predictions.squeeze(), batch['labels'])

#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

#         # 검증
#         val_loss, val_mse, val_rmse, val_r2 = evaluate_model_val(model, val_loader, config['device'])
#         print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")

#         wandb.log({
#             'epoch': epoch + 1,
#             'train_loss': avg_train_loss,
#             'val_loss': val_loss,
#             'val_mse': val_mse,
#             'val_rmse': val_rmse,
#             'val_r2': val_r2,
#         })

#         # Best 모델 저장
#         if val_mse < best_val_mse:
#             best_val_mse = val_mse
#             save_best_model(
#                 model,
#                 save_dir=config['save_path'],
#                 base_name=config['model_name'],
#                 epoch=epoch + 1,
#                 val_loss=val_loss,
#                 score=val_rmse,
#             )

# ##### k-fold 
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
# from torch.utils.data import Dataset
import os
from trainer import *

def train_model_kfold_and_save_best(config):
    set_seed(config['seed'])
    
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_folds = config['n_splits']
    saved_models_dir = config['save_path']
    
    tokenizer = BertRegressionModel(config['model_name']).get_tokenizer()
    
    # 전체 데이터 로딩 (원본 값)
    full_dataset = get_dataset(config, tokenizer)
    all_texts = full_dataset.texts
    all_outputs = np.array(full_dataset.outputs, dtype=float)

    best_mse = float('inf')
    best_fold = None
    best_model_path = None
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_texts)):
        print(f"\n🔁 Fold {fold+1}/{num_folds}")

        train_texts = [all_texts[i] for i in train_idx]
        val_texts = [all_texts[i] for i in val_idx]

        train_outputs = all_outputs[train_idx]
        val_outputs = all_outputs[val_idx]

        # 1) fold별 train 데이터 기준으로 정규화값 계산
        mean = train_outputs.mean()
        std = train_outputs.std()

        # 2) 정규화 적용
        train_outputs_norm = (train_outputs - mean) / std
        val_outputs_norm = (val_outputs - mean) / std

        # 3) Dataset 생성 (정규화된 값 사용)
        train_dataset = MyDataset(train_texts, train_outputs_norm.tolist(), tokenizer, max_len=full_dataset.max_len)
        val_dataset = MyDataset(val_texts, val_outputs_norm.tolist(), tokenizer, max_len=full_dataset.max_len)

        # print(f"Validation dataset type: {type(val_dataset)}")
        # print(f"Validation dataset length: {len(val_dataset)}")
        # print(f"Validation dataset sample item keys: {val_dataset[0].keys()}")
                
        model = BertRegressionModel(config['model_name']).get_model()
        model.to(device)
        
        output_dir = os.path.join(saved_models_dir, f'fold_{fold+1}')
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            eval_strategy="epoch",
            save_strategy="epoch",             # ? 자동 저장 OFF → 수동 저장만 수행
            save_steps = None,
            save_total_limit=2,
            logging_strategy="epoch",     # val 결과 로그 기록 주기 (optional)
            learning_rate=float(config['learning_rate']),
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=config['epochs'],
            # weight_decay=0.01,                          # AdamW 옵티마이저
            load_best_model_at_end=True,                   # best추척 커스텀
            metric_for_best_model='eval_mse',  # 평가 지표
            greater_is_better=False,  # 평가 지표가 클수록 좋은지 여부 (True면 큰 값이 좋음)
            seed=config['seed'],
            report_to='wandb',
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: compute_metrics_made(eval_pred, mean, std),
            callbacks=[PrintMetricsCallback(), WandbEpochCallback()]
        )
        
        # print training_args.to_dict()
        trainer.train()

        eval_result = trainer.evaluate()
        mse = eval_result.get('eval_mse', float('inf'))
        print(f"📉 Fold {fold+1} MSE: {mse:.4f}")
        
        fold_best_mse = float('inf')
        fold_best_epoch = None

        for log in trainer.state.log_history:
            if 'eval_mse' in log and 'epoch' in log:
                if log['eval_mse'] < fold_best_mse:
                    fold_best_mse = log['eval_mse']
                    fold_best_epoch = int(log['epoch'])

        if fold_best_epoch is not None:
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{fold_best_epoch}')
            print(f"💾 Fold {fold+1} best checkpoint: {checkpoint_dir} (eval_mse={fold_best_mse:.4f})")

            if fold_best_mse < best_mse:
                best_mse = fold_best_mse
                best_fold = fold + 1
                best_model_path = checkpoint_dir

    print(f"\n🏆 Best Fold: {best_fold} with MSE: {best_mse:.4f}")
    print(f"📂 Best model saved at: {best_model_path}")
    
    return best_model_path

def retrain_best_model_on_full_data(config, best_model_path):
    print("\n📢 Retraining final model on full training set...")
    print("best_model_path : ",best_model_path)
    
    match = re.search(r'epoch-(\d+)', best_model_path)
    epoch_number = int(match.group(1))
    epochs = epoch_number
    

    train_json_path = os.path.join(config['data_dir'], config['train_file'])
    texts, outputs = load_data(train_json_path)
    max_len = config['max_len']
    batch_size = config['batch_size']
    saved_models_dir = config['save_path']
    final_output_dir = os.path.join(saved_models_dir, 'final_model')

    os.makedirs(final_output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, tokenizer = load_best_model(config, best_model_path, device=device)

    full_dataset = MyDataset(texts, outputs, tokenizer, max_len)

    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='no',
        save_strategy='no',
        logging_dir=os.path.join(final_output_dir, 'logs'),
        report_to=["wandb"],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    
    torch.save(model.state_dict(), os.path.join(final_output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(final_output_dir)    # 토크나이저 저장
    print(f"\n✅ Final model saved to: {final_output_dir}")
