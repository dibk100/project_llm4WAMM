# Project : LLM for WAAM 🤖⚙️
- 2025년 1학기 품질데이터분석 수업 기말 프로젝트(초안)   
- WAAM(Wire Arc Additive Manufacturing) 공정에서의 의사결정 지원을 위한 LLM 기반 모델 개발
- TASK : 이상탐지, 에너지 예측

### 📁 Folder Structure
```
project_llm4WAMM/
├── task01_classifier/                     # 분류
├── task02_regression/                     # 수치 예측 
├── data_analysis_EDA.ipynb                # 데이터 분석 및 전처리    
└── requirements.txt    
```

# 📝 BERT 실험 기록

### 🔹 TASK01a: 멀티 라벨 분류  
- **Loss Function:** `BCEWithLogitsLoss` 
- **모델 저장 기준:** Partial Score 기준    
- **Update:** 2025.06.03   
- **결과:**   

| Model              | Val Loss | Macro F1 | Micro F1 | Partial Score |
|--------------------|----------|----------|----------|----------------|
| `bert-base-uncased`| 0.4596   | 0.5028   | 0.6267   | 0.6185         |
| `roberta-base`     | 0.4368   | 0.5190   | 0.6196   | 0.6003         |

### 🔹 TASK01b: 이진 분류 
- **Loss Function:** `CrossEntropyLoss`    
- **모델 저장 기준:** val_loss 기준
- **Update:** 2025.06.10  
- **결과:**   

| Model              | Val Loss | Macro F1 | Accuracy |
|--------------------|----------|----------|----------------------------|
| `bert-base-uncased` | 0.3767   | 0.7294   | 0.7823                     |
| `roberta-base`      | 0.3441   | 0.7609   | 0.8231                     |

--- 
### 🔹 TASK02: 에너지 예측 (수치 회귀)  
- **Loss Function:** `MSELoss`  
- **모델 저장 기준:** val_mse 기준
- **Update:** 2025.06.10  
- **결과 :**   

| Model              | Val Loss | MSE     | RMSE    | R² Score |
|--------------------|----------|---------|---------|----------|
| `bert-base-uncased`| 0.0902   | 0.0848  | 0.2912  | 0.9544   |
| `roberta-base`     | 0.1076   | 0.1064  | 0.3261  | 0.9428   |


<!--
### To-Do

- [ ] 🔄 작업 중 : 품질 예측 성능 평가 코드 개선 중
- [ ] ✅ 완료됨 : 데이터셋 병합 및 전처리 (2025-05-23)
- [ ] 📌 다음 할 일 : inference 모듈 디버깅
-->
