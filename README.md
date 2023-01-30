# DACON_2208 : 청경채 성장 예측 AI 경진대회
- Structured Data   
- Time Series Prediction  
- Regression  


## 1. 대회 결과

- 최종 성적
    - Public  :
        - **RMSE : 20.68205  ( 47 / 171 )**
            - 1위 : 15.71016
    - Private :
        - **RMSE : 21.53071  ( 42 / 171 , top 25%)**
            - 1위 : 13.70547
            
            
## 2. 대회 개요

- 주최 : KIST 강릉분원
- 주관 : 데이콘
- 대회기간 : 2022.08.17 ~ 2022.09.19  
- Task  
    - Timeseries prediction  
- 상금 : 총 300만원
    - 1위 150만원
    - 2위 100만원
    - 3위 50만원



## 3. 진행 과정

- Deep learning 모델들(LSTM, Transformer)을 직접 구현해 보는 것이 1차 목표
    - Timeseries data 를 다뤄보는 것이 LSTM, Transformer 모델에 적합하다고 판단
- 전처리
    - 초기에는 데이터의 전처리를 따로 수행하지 않았음
        - target column들만 바꾸어가며 모델을 훈련시킴
            - 총 38개 column 중 10~14개 column 사용
            - 시간, 온도, 습도, LED 조명과 관련된 column들
    - 대회 종료 일주일 전 쯤 뒤늦게 데이터 전처리 수행
        - NULL값 정제 (interpolation)
        - 시간과 관련된 변수를 cyclical continuous feature로 encoding
            - [https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/)
- 모델 선정 및 훈련
    - 공부하며 직접 만든 LSTM, Transformer 이용  
    - torch가 제공하는 모델 이용  
    - LSTM, Transformer 가 아닌 Linear based 모델 이용 -> 청경채 데이터에 적용하는 데 어려움을 겪음
    - 직접 만든 transformer model이 가장 성능이 좋았음  
        → Transformer_0907_AB.py  
    

## 4. Self-feedback?

### 의의 :

- LSTM, Transformer 구조를 숙지함
- PyTorch 사용법을 숙지함

### 개선할 점 :

- 데이터 관련 : **Garbage in, garbage out !**  
    - EDA부터 수행할 것!
        - 모델 선정 및 훈련은 그 이후
        - 아무리 좋은 모델이라도 데이터가 구리면 결과도 구릴 수밖에 없음
        - EDA는 좋은 모델을 선정하기 위한 발판이 된다
    - 데이터 형태 및 분포 파악하기
        - NULL values, outliers 처리에 대한 고민
        - 치우친 데이터에 대한 처리 (정규화, 혹은 그대로 두기)
    - 다른 사람들의 데이터 전처리 과정 공부할 것

- 모델 관련 :  
    - 언제나 Deep learning 모델이 최고인 것은 아니다
        - baseline으로 LSTM 모델이 주어졌고, 데이터의 양이 상당했음
            - 데이터가 많으니 deep learning model이 적절할 것이라는 착각
            - 그러나 **데이터의 절대적인 양이 많다고 해서, 유용한 데이터도 많은 것은 아님**
                - 실제로 주어진 케이스는 58개 케이스 (58개 청경채)뿐
                - 58개 케이스 각각에 대해 fit되는 것이, general 하게 적용된다는 보장은 없다
            - 실제로 순위권 참가자들 대부분은 ML model을 사용
                - Xgboost, LGBM 등

- 코드 구조 관련:  
    - py 파일 구조 설계 후 작업에 코드 작성에 돌입할 것
    - 일정 주기마다 반드시 refactoring 수행할 것
        - 난잡한 코드는 생산성을 떨어뜨림
        - ex1. Config.py
            - raw data 및 out 경로와 관련된 내용 분리
            - train 관련 argument 분리
            - scheduler관련 argument 분리
        - ex2.layers
            - model.py 및 부수적인 py 파일 한 두개로 정리할 것
