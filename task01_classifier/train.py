from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
from dataset import get_dataset
from model import get_model
from utils import set_seed, save_best_model
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

    best_val_loss = float('inf')
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
        val_loss, macro_f1, micro_f1, partial_score  = evaluate_model_val(model, val_loader, config['device'])

        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}| Partial Score: {partial_score:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score' : partial_score,
        })

        if partial_score > best_score:
            best_score = partial_score
            save_best_model(
                model,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch + 1,
                val_loss=val_loss,
                partial_score = partial_score,
            )
