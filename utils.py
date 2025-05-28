import datetime
import os
import torch

def save_best_model(model, save_dir, base_name, num_labels, best_f1):
    """모델을 저장하는 함수"""
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")  # 날짜+시간: 예) 0528_1530
    
    # 디렉토리 생성
    ckpt_dir = os.path.join(save_dir, f"{base_name}_{num_labels}")
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    # 파일 이름 구성
    model_filename = f"{timestamp}_best_model_f1_{best_f1:.4f}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)
    
    # 저장
    torch.save(model.state_dict(), ckpt_path)
    print(f">>> Best model saved: {ckpt_dir}/{model_filename} (F1: {best_f1:.4f})")
