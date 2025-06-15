# 💡Info.

- **TASK:** 
    - `Heat input`/`Energy`을(수치) 예측하는 **Regression task**

- **Dataset**
  - 원본 파일에서 템플릿화 함: `train_waam_pred_hi_p5.json`, `train_waam_pred_hi_p7.json`
  - `input`(문장),`output`(수치값만 전처리함) 활용
    ```
    example.
    {
        "input": "The material used in the ~ .What is the result of the heat input with these experiment parameters?",
        "output": "656.112."
      }
    ```

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