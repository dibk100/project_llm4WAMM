import os
import torch
import wandb
import numpy as np
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model import BertRegressionModel
from dataset import MyDataset, load_data
from utils import compute_metrics


def train_model_kfold_and_save_best(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📦 Using device: {device}")

    texts, outputs = load_data(config['data']['train_path'])
    max_len = config['data']['max_len']
    num_folds = config['training']['num_folds']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    saved_models_dir = config['training']['saved_models_dir']

    tokenizer = BertRegressionModel().get_tokenizer()
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_rmse = float('inf')
    best_fold = None
    best_model_path = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n🔁 Fold {fold+1}/{num_folds}")

        train_texts = [texts[i] for i in train_idx]
        train_outputs = [outputs[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_outputs = [outputs[i] for i in val_idx]

        train_dataset = MyDataset(train_texts, train_outputs, tokenizer, max_len)
        val_dataset = MyDataset(val_texts, val_outputs, tokenizer, max_len)

        model = BertRegressionModel(device=device).get_model()

        output_dir = os.path.join(saved_models_dir, f'fold_{fold+1}')
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model='eval_rmse',
            greater_is_better=False,
            report_to=["wandb"],
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        eval_result = trainer.evaluate()

        rmse = eval_result.get('eval_rmse', float('inf'))
        print(f"📉 Fold {fold+1} RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_fold = fold + 1
            best_model_path = os.path.join(output_dir, 'checkpoint-best')

    print(f"\n🏆 Best Fold: {best_fold} with RMSE: {best_rmse:.4f}")
    return best_model_path


def retrain_best_model_on_full_data(config, best_model_path):
    print("\n📢 Retraining final model on full training set...")

    texts, outputs = load_data(config['data']['train_path'])
    max_len = config['data']['max_len']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    saved_models_dir = config['training']['saved_models_dir']
    final_output_dir = os.path.join(saved_models_dir, 'final_model')

    os.makedirs(final_output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = BertRegressionModel(model_path=best_model_path, device=device)
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()
    full_dataset = MyDataset(texts, outputs, tokenizer, max_len)

    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='no',
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
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"\n✅ Final model saved to: {final_output_dir}")
