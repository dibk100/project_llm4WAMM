# 💡Info.
- **TASK :** 정상/비정상 분류(binary), 결함 유형 분류(multi)
- **데이터**
  - 원본 파일: `train_waam_cls_inst_p1.json`
  - `input`,`output`만 활용

## 📝 BERT 실험 기록

### 🔹 TASK01: 멀티 라벨 분류  
- **Loss Function:** `BCEWithLogitsLoss` 
- **모델 저장 기준:** Partial Score 기준   
- **Update:** 2025.06.03   
- **결과:**   

| Model              | Val Loss | Macro F1 | Micro F1 | Partial Score |
|--------------------|----------|----------|----------|----------------|
| `bert-base-uncased`| 0.4596   | 0.5028   | 0.6267   | 0.6185         |
| `roberta-base`     | 0.4368   | 0.5190   | 0.6196   | 0.6003         |
<details>
<summary>📊 평가지표 설명</summary>

| 지표 이름           | 설명 |
|---------------------|------|
| **Val Loss**        | 검증 데이터(validation set)에서의 평균 손실 값. 모델의 과적합 여부나 학습 안정성을 판단하는 데 사용됨. <br> → `train_dataset`: 446 rows / `val_dataset`: 112 rows |
| **Macro F1 Score**  | 각 클래스의 F1 점수를 개별 계산 후 단순 평균. <br> → **소수 클래스의 성능을 강조**하는 데 유리함. |
| **Micro F1 Score**  | 전체 TP/FP/FN을 합산한 후 계산한 F1 점수. <br> → **전체적인 예측 정확도**를 반영하며, 다수 클래스 영향을 많이 받음. |
| **Partial Score**   | 일부 정답 라벨만 맞췄을 때도 점수를 부여하는 **커스텀 지표**. <br> 예: 일부 정답만 맞춰도 부분 점수를 인정함. |

</details>   

---
### 🔷 TASK01B: 이진 분류  
- **데이터 전처리:**  
  - 라벨에 `"Normal"`이 포함되어 있으면 → **Normal**  
  - 그렇지 않으면 → **Abnormal**
- **예측 처리:**  
  ```python
  preds = logits.argmax(axis=1)
- **Loss Function:** `CrossEntropyLoss`    
- **모델 저장 기준:** val_loss 기준
- **Update:** 2025.06.10  
- **결과:**   
 
| Model              | Val Loss | Macro F1 | Accuracy |
|--------------------|----------|----------|----------------------------|
| `bert-base-uncased` | 0.3767   | 0.7294   | 0.7823                     |
| `roberta-base`      | 0.3441   | 0.7609   | 0.8231                     |


### 📁 Folder Structure
```
project_llm4WAMM/task01_classifier/
├── config/                     # 설정 파일
│   └── yaml
├── saved_models/               # 학습된 모델 저장 
│   └── pth  
├── dataset.py              
├── eval.py                     # 모델 평가(val/test)
├── main.py                 
├── main_classifier.py                
├── train.py                
├── utils.py               
└── requirements.txt    
```