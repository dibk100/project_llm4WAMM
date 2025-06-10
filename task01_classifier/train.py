from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
from dataset import get_dataset
from model import get_model
from utils import *
from eval import evaluate_model_val

def train_model(config):
    set_seed(config['seed'])
    
    # 자동 run_name 생성
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)

    # 데이터셋 & DataLoader
    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 모델 초기화
    model = get_model(config)
    model.to(config['device'])

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # best_val_loss = float('inf')
    best_score = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(config['device']) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # 검증 평가
        val_loss, macro_f1, micro_f1, partial_score,exact_match_acc, label_wise_acc  = evaluate_model_val(model, val_loader, config['device'])
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}| Partial Match Score: {partial_score:.4f}| Exact Match Acc: {exact_match_acc:.4f}| Label Wise Acc: {label_wise_acc:.4f}")

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score' : partial_score,
            'exact_match_acc' : exact_match_acc,
            'label_wise_acc' : label_wise_acc
        })

        if partial_score > best_score:
            best_score = partial_score
            save_best_model(
                model,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch,
                val_loss=val_loss,
                partial_score = partial_score,
            )

# # 추가로 진행한 코드
# # kfold!
# from sklearn.model_selection import KFold
# from torch.utils.data import DataLoader, Subset

# def train_model_kfold_and_save_best(config):
#     set_seed(config['seed'])
#     full_dataset = get_dataset(config, split='train')
#     kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])

#     best_val_loss = float('inf')
#     best_model_path = None
#     best_epoch = 0

#     for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
#         print(f"\n🔁 Fold {fold+1}/{config['n_splits']}")

#         train_dataset = Subset(full_dataset, train_idx)
#         val_dataset = Subset(full_dataset, val_idx)
#         model_path, loss, epoch = train_on_fold(train_dataset, val_dataset, fold, config)

#         if loss < best_val_loss:
#             best_val_loss = loss
#             best_model_path = model_path
#             best_epoch = epoch

#     print(f"\n🏆 Best Fold Score: {best_val_loss:.4f}")
#     print(f"\n🏆 Best Model: {best_model_path}")
#     return best_model_path, best_epoch

# def train_on_fold(train_dataset, val_dataset, fold, config):
#     device = config['device']
#     model = get_model(config).to(device)

#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
#     num_training_steps = config['epochs'] * len(train_loader)
#     scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#     run_name = f"{config['model_name']}_fold{fold+1}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
#     wandb.init(project=config['wandb_project'], name=run_name, config=config, reinit=True)

#     best_loss = float('inf')
#     best_model_path = None
#     early_stopping = EarlyStopping(patience=config.get('patience', 5), verbose=True)

#     for epoch in range(config['epochs']):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         val_loss, macro_f1, micro_f1, partial_score = evaluate_model_val(model, val_loader, device)

#         print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Partial Score: {partial_score:.4f}")
#         wandb.log({
#             'epoch': epoch + 1,
#             'train_loss': avg_train_loss,
#             'val_loss': val_loss,
#             'macro_f1': macro_f1,
#             'micro_f1': micro_f1,
#             'partial_score': partial_score,
#         })

#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_epoch = epoch + 1
#             best_model_path = save_best_model(
#                 model,
#                 save_dir=config['save_path'],
#                 base_name=f"{config['model_name']}_fold{fold+1}",
#                 epoch=epoch + 1,
#                 val_loss=val_loss,
#                 partial_score=partial_score
#             )

#         early_stopping(val_loss)
#         if early_stopping.early_stop:
#             print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
#             break

#     wandb.finish()
#     return best_model_path, best_loss, best_epoch

# def retrain_best_model_on_full_data(config, best_model_path):
#     print("\n📢 Retraining final model on full dataset...")
#     full_dataset = get_dataset(config, split='train')
#     device = config['device']
#     model = get_model(config)
#     model.load_state_dict(torch.load(best_model_path))
#     model.to(device)

#     loader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
#     scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=config['epochs'] * len(loader))

#     for epoch in range(config['best_epoch']):
#         model.train()
#         total_loss = 0
#         for batch in loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1} | Full Train Loss: {total_loss / len(loader):.4f}")

#     final_save_path = os.path.join(config['save_path'], f"{config['model_name']}_final.pt")
#     torch.save(model.state_dict(), final_save_path)
#     print(f"\n✅ Final model saved to {final_save_path}")