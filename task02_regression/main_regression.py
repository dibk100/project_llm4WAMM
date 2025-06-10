import argparse
from train import *
from eval import evaluate_model
import yaml
import wandb
from dotenv import load_dotenv
import os

def main():
    load_dotenv(verbose=True)               # .env심기, wandb로그인 key
    parser = argparse.ArgumentParser(description="BERT-based Heat input Regression")
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        
        # wandb 로그인
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("Warning: WANDB_API_KEY not set. wandb login skipped.")
            assert False, "WANDB_API_KEY environment variable is missing."

        train_model(config)
        wandb.finish()
        
    elif args.mode == 'eval':
        test_mse, test_rmse, test_r2 = evaluate_model(config, split='test')
        print(f"✅ Test MSE: {test_mse:.4f}")
        print(f"✅ Test RMSE: {test_rmse:.4f}")
        print(f"✅ Test R²: {test_r2:.4f}")
    
    else :
        assert False, "다시 코드 실행"
        
if __name__ == "__main__":
    main()