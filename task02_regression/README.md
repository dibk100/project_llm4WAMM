# 💡Info.

- **TASK:** `Heat input`을 예측하는 **Regression task**
  - **독립변수 (4개):**  `Avg Voltage`, `Avg Current`, `Travel Speed`, `Wire Feed Rate`
  - **종속변수 (1개):**  `Heat input`

- **데이터**
  - 원본 파일: `train_waam_cls_inst_p1.json`
  - `text`에서 Heat input 관련 텍스트 제거 & 마지막에 "what~" 문장 제거
  - 대신에 `"Predict the heat input."` 문장 추가
  - `output`은 Heat input(수치값)으로 설정
```
"text": "~~~. Predict the heat input.",
"label": 10.24
```

## 📝 BERT 실험 기록

### TASK02: 에너지 예측 (수치 회귀)  
- **Loss Function:** `MSELoss`  
- **Update:** 2025.06.10  
- **결과 :**   

| Model              | Val Loss | MSE     | RMSE    | R² Score |
|--------------------|----------|---------|---------|----------|
| `bert-base-uncased`| 0.0902   | 0.0848  | 0.2912  | 0.9544   |
| `roberta-base`     | 0.1076   | 0.1064  | 0.3261  | 0.9428   |

## 📁 Folder Structure
```
project_llm4WAMM/task01_classifier/
├── config/                     # 설정 파일
│   └── yaml
├── saved_models/               # 학습된 모델 저장 
│   └── pth  
├── dataset.py              
├── eval.py                     # 모델 평가(val/test)
├── main.py                 
├── main_regression.py                
├── train.py                
├── utils.py               
└── requirements.txt    
```