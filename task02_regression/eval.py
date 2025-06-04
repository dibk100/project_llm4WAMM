import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from transformers import Trainer, TrainingArguments
from model import BertRegressionModel
from dataset import MyDataset, load_data

def evaluate_model(config, split='test'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧪 Using device: {device}")

    # Load test data
    test_path = config['data']['test_path']
    max_len = config['data']['max_len']
    texts, outputs = load_data(test_path)

    # Load model/tokenizer
    model_path = config['model']['final_model_dir']
    model_wrapper = BertRegressionModel(model_path=model_path, device=device)
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()

    test_dataset = MyDataset(texts, outputs, tokenizer, max_len)

    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=config['training']['batch_size'],
        do_train=False,
        do_eval=True,
        logging_dir="./eval_logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    print("🔍 Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.squeeze()
    labels = predictions.label_ids.squeeze()

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    return mse, rmse, r2
