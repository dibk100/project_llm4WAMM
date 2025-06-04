# Project : LLM for WAAM 🤖⚙️
- 2025년 1학기 품질데이터분석 수업 기말 프로젝트(초안)   
- WAAM(Wire Arc Additive Manufacturing) 공정에서의 의사결정 지원을 위한 LLM 기반 모델 개발
- TASK : 이상탐지, 에너지 예측

### 📁 Folder Structure
```
project_llm4WAMM/
├── config/                     # 설정 파일
│   └── yaml
├── saved_models/               # 학습된 모델 저장 
│   └── pth
├── data_analysis_EDA.ipynb     # 데이터 분석 및 전처리    
├── dataset.py              
├── eval.py                     # 모델 평가(val/test)
├── main.py                 
├── task01_model.py                
├── train.py                
├── utils.py               
└── requirements.txt    
```

# 📝 BERT 실험 기록

**Update**: 2025.06.03   
**Task:** `TASK01_멀티 라벨 분류`  
**Loss Function:** `BCEWithLogitsLoss`   

| Model              | Val Loss | Macro F1 | Micro F1 | Partial Score |
|--------------------|----------|----------|----------|----------------|
| `bert-base-uncased`| 0.4596   | 0.5028   | 0.6267   | 0.6185         |
| `roberta-base`     | 0.4368   | 0.5190   | 0.6196   | 0.6003         |


<details>
<summary>📊 평가지표 설명</summary>

| 지표 이름        | 설명 |
|------------------|------|
| **Val Loss**     | 검증 데이터(validation set)에서의 평균 손실 값. 모델의 과적합 여부나 학습 안정성을 판단하는 데 사용함. <br> `train_dataset`: 446 rows <br> `val_dataset`: 112 rows |
| **Macro F1 Score** | 각 클래스(라벨)의 F1 점수를 독립적으로 계산한 후, 단순 평균한 값. <br> → 클래스 간 불균형이 있는 멀티라벨 분류 문제에서 **소수 클래스의 성능을 강조**하는 데 유용함. |
| **Micro F1 Score** | 전체 클래스에 대해 TP/FP/FN을 모두 합산한 뒤 계산한 F1 점수. <br> → **전체 예측 성능(정확도 중심)**을 평가하는 지표로, 다수 클래스의 영향을 더 크게 받음. |
| **Partial Score** | **커스텀 평가 지표**로, 정답 레이블(target)의 일부만 맞췄을 때 정답의 비율만큼 점수를 주는 방식. <br> 예: 일부 정답 라벨만 맞혀도 점수를 부여함. |

</details>


<!--
### To-Do

- [ ] 🔄 작업 중 : 품질 예측 성능 평가 코드 개선 중
- [ ] ✅ 완료됨 : 데이터셋 병합 및 전처리 (2025-05-23)
- [ ] 📌 다음 할 일 : inference 모듈 디버깅
-->
