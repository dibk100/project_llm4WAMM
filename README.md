# Project : LLM for WAAM 🤖⚙️

- Course : 2025년 1학기 품질데이터분석 프로젝트  
- Subject : WAAM(Wire Arc Additive Manufacturing) 공정에서의 의사결정 지원을 위한 LLM(대형언어모델) 기반 모델 개발  
- Role : **BERT** 기반 모델 실험,데이터 분석 및 전처리

## ⚙️ Tasks

- **이상탐지 (Anomaly Detection)**  
  - 정상/비정상 상태 분류 (Binary Classification)  
  - 결함 유형별 라벨 분류 (Multi-label Classification)  

- **에너지 예측 (Energy Prediction)**  
  - 열, 에너지 등의 수치 예측 (Regression)  

- **LLM 활용**  
  - 비정형 데이터 해석 및 의사결정 지원  
  - 이상탐지 및 예측 결과에 대한 자연어 설명 생성  


## 📚 학습 방법
- K-Fold Cross Validation  
- HuggingFace Trainer 활용 
- 수동 학습(Scratch code) 병행 실험

## 📌 Notes & Issues 🧷
- `BCEWithLogitsLoss`는 PyTorch에서 이진 분류(Binary Classification) 문제를 처리할 때 자주 사용하는 손실함수    
    - Binary Cross Entropy (BCE) + Sigmoid 함수 =`BCEWithLogitsLoss`
- Multi-label classification도 `BCEWithLogitsLoss`를 활용함.     
    - if labels = 3, labels[0] = [0 or 1],labels[1] = [0 or 1],labels[2] = [0 or 1]   
    - 각 라벨별로 독립적인 이진 분류를 수행.
- `model.py` 내 손실 함수 결정 로직 (AutoModelForSequenceClassification 기준)
    - num_labels=1 & problem_type='single_label_classification' → `BCEWithLogitsLoss`   
    - num_labels>=2 & problem_type='single_label_classification' → `CrossEntropyLoss`    
    - num_labels>=2 & problem_type='multi_label_classification' → `BCEWithLogitsLoss`
- Trainer :
    - 수동학습 코드와 비교했을 때 결과 차이가 있었음.
    - trainer 내부의 자동 최적화 기능(ex.weight_decay=0.01(AdamW 옵티마이저)) 설정 때문인지 정확한 원인을 모르겠음.
    - [task02에서 bert 커스텀(regression)했더니 trainer로 loss값 얻을 때 이슈가 발생했었음](https://discuss.huggingface.co/t/implementing-a-trainer-with-custom-loss-produces-key-error/38171). foward() 커스텀함. `model.py` 에 상세 작성함.


## 📁 Folder Structure
```
project_llm4WAMM/
├── task01_classifier/                     # Anomaly Detection
├── task02_regression/                     # Prediction
├── data_analysis_EDA.ipynb                # 데이터 분석 및 전처리    
└── requirements.txt    
```
   
## 🧪 BERT 실험 기록

### 🔍 TASK01a: Multi-label classification(결함 유형 분류)  
- **Loss Function:** `BCEWithLogitsLoss` 
- **모델 저장 기준:** Partial Score 기준    
- **Update:** 2025.06.14   
- **결과:**   

| Model               | Val Loss | Macro F1 | Micro F1 | Partial Score | Exact Match | Label-wise Accuracy |
| ------------------- | -------- | -------- | -------- | ------------- | ----------- | ------------------- |
| `bert-base-uncased` | 0.1719   | 0.7366   | 0.9091   | 0.8929        | 0.8750      | 0.9673              |
| `roberta-base`      | 0.3370   | 0.4354   | 0.7934   | 0.8393        | 0.8036      | 0.9256              |


### 🔍 TASK01b: Bindary classification(양/불 분류)
- **Loss Function:** `BCEWithLogitsLoss`    
- **모델 저장 기준:** val_loss 기준
- **Update:** 2025.06.14  
- **결과:**   

| Model               | Val Loss | Macro F1 | Micro F1 | Partial Score | Exact Match | Label-wise Accuracy |
| ------------------- | -------- | -------- | -------- | ------------- | ----------- | ------------------- |
| `bert-base-uncased` | 0.2688   | 0.9464   | 0.9464   | 0.9464        | 0.9464      | 0.9464              |
| `roberta-base`      | 0.2593   | 0.9642   | 0.9643   | 0.9643        | 0.9643      | 0.9643              |


--- 
### 🔍 TASK02a: Heat input Regression
- **Loss Function:** `MSELoss`  
- **모델 저장 기준:** val_mse 기준
- **Update:** 2025.06.14  
- **결과 :**   

| Model              | Val Loss | MSE     | RMSE    | R² Score |
|--------------------|----------|---------|---------|----------|
| `bert-base-uncased`| 0.0902   | 0.0848  | 0.2912  | 0.9544   |
| `roberta-base`     | 0.1076   | 0.1064  | 0.3261  | 0.9428   |

### 🔍 TASK02b: Energy Regression
- **Loss Function:** `MSELoss`  
- **모델 저장 기준:** val_mse 기준
- **Update:** 2025.06.14  
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
