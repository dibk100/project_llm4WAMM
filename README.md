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
│   └── pt
├── data_analysis_EDA.ipynb     # 데이터 분석 및 전처리    
├── dataset.py              
├── eval.py                     # 모델 평가(val/test)
├── main.py                 
├── model.py                
├── train.py                
├── utils.py               
└── requirements.txt    
```

### 📦 Data Description
- 데이터 설명 : 
- 데이터 전처리 과정 :

### To-Do(with DI)
- [O] ✅ 샘플 데이터셋 전달 받기(from승환) --> 전체 데이터 
- [O] ✅ BERT기반 모델로 분류 태스크 baseline 
- [ ] 🔄 멀티 라벨 태스크 확인을 위한 데이터 분석
- [ ] 🔄 멀티 라벨 태스크 구현 및 검증

<!--
- [ ] 🔄 작업 중 : 품질 예측 성능 평가 코드 개선 중
- [ ] ✅ 완료됨 : 데이터셋 병합 및 전처리 (2025-05-23)
- [ ] 📌 다음 할 일 : inference 모듈 디버깅
-->
