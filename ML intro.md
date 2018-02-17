<!-- page_number: true -->

# ML INTRO
for our team :)

---
## WHAT IS THE DATA SCIENCE?
- Data Science
- Data Mining
- Statistics
- Machine Learning
- Deep Learning
- Artificial Intelligence
...

---
## 프로그래밍 vs 머신러닝 ( 방법의 진화 )

<img src="./deepblue.jpg" width=500 align="right">

PROGRAMING
- rule based

MACHINE LEARNING
- training based & data based

`1997.5.11 Deepblue`

---
## 통계학 vs 머신러닝 ( 관점의 변화 )

<img src="./ml_diagram.png" width=400 align="right">

STATISTICS
- limited data samples
- focusing modeling & hypothesis

MACHINE LEARNING
- lots of data samples
- focusing accuracy & optimization

---
## DATA ANALYSIS
### 통계학이든 머신러닝이든 딥러닝이든 결국 *데이터분석*
- MODELING = `Finding Function`
- OPTIMIZING = `Finding Parameters`

### 데이터분석에 대한 오해
- Garbage In, Garbage Out
- 사람이 이론적으로 분석할 수 없는 분석은 어떤 알고리즘으로도 분석 X
- 머신러닝이란 '경제성'과 '효율성'을 증가시키는 것, 거대한 '규모'의 분석을 가능하게 하는 것

---
## 데이터 타입
- 실수 ( real ) : continuous, nominal value
- 카테고리 ( category )
	- class
		- binary ( ex. 0/1, 남/여 )
		- multi ( ex. 개, 고양이, 사자 .. )
	- odinal ( ex. A등급, B등급, ... )

---
## 데이터 형태
- 정형 데이터 ( Structured ) 
	- table 형태 ( row x column )
	- ex. excel, SQL( Structured Query Language )
- 비정형 데이터 ( Unstructured )
	- table 에 담기지 않는 형태
	- ex. image, voice, video, text ...
	- feature extraction 작업이 필요 (딥러닝에선 이 작업도 건너뜀)

---
## Input vs Output
<img src="./function.png" width=300 align="right">

- 입력데이터
	- 분석의 기반이 되는 데이터
	- 표기법 : $X$
	- feature, 독립변수( independent variable ), 설명변수 ( explanatory variable )
- 출력데이터
	- 추정하거나 예측하고자 하는 목적 데이터
	- 표기법 : $y$
	- target, 종속변수 ( dependent variable)

---
## 데이터분석 방법론
- Supervised learning
	- with completely labeled training data ( $X$ 에 대한 $y$ 가 주어짐 )
	- `prediction`
- Unsupervised learning
	- without any labeled training data ( $X$ 에 대한 $y$ 가 없음 )
	- `clustering, approximation`
- Semi-supervised learning
	- typically a small amount of labeled data with a large amount of unlabeled data
	- `Chess, AlphaGo`

---
## 데이터분석 종류
- 예측 ( Prediction ) : $X$ 가 주어질 때 $y$ 를 예측
	- 회귀 ( Regression ) : target $y$ 가 실수
	- 분류 ( Classification ) : target $y$ 가 카테고리
- 군집 ( Clustering ) : $y$ 값(결과)이 주어지지 않고, $X$ 의 특성만으로 분류
- 모사 ( Approximation ) : 차원축소, 압축
	- 일부데이터가 $y$ 를 잘 표현 -> data 양 줄일 수 있음

---
## Prediction model
예측 문제의 목표는 $X$ 와 $y$ 의 관계 함수 $f$ 를 찾는 것 ( 현실적으로는 정확한 $f$ 를 구할 수 없으므로 가장 유사한 재현 함수 $\hat{f}$ 을 구한다.

$$
\hat{y} = \hat{f}(X) = \arg\max_y P(y|X, D)
$$

<center><img src="./table1.png" width=700 height=300></center>

`질병발생 예측 알고리즘 개발 데이터(감기_진료_기상_환경)`

---
## Overfitting

- 모델의 복잡도가 과도하게 높은 경우
- 데이터가 학습 데이터의 디테일과 노이즈를 과도하게 학습, 새로운 데이터에 적용하기 힘든 모델

<center><img src="./overfitting.png" width=600>


---
## Cross Validation ( 1 )
- 모형의 최종 성능을 객관적으로 측정하기 위함
- training 에 사용되지 않은 새로운 데이터, 즉 검증용 혹은 테스트용 데이터를 사용해서 예측한 결과를 기반으로 성능을 개선

<center><img src="./cv1.png" width=350 > <img src="./cv2.png" width=350 ></center>

<center><img src="./cv3.png" width=350 > <img src="./cv4.png" width=350 ></center>

---
## Cross Validation ( 2 )
- 모델이 가진 파라미터 값을 조정하면서, 모델의 복잡도를 조절
`아래 그림에선 파라미터 값이 높아질 수록 모델 복잡성 (학습 정도) 가 증가한다고 가정`

<center><img src="./cv_curve.png" width=600>

---
## 데이터분석 WORK FLOW
1. 데이터 수집 ( Collection )
2. 데이터 전처리 ( Preprocessing )
	- Feature extraction
	- EDA ( Exploratory data analysis )
	- Feature engineering
	- Feature selection
3. 모델링 ( Modeling )
	- Data split & Cross validation
	- Training & Optimization ( Parameter tuning )
	- model selection
4. 예측 ( prediction )
5. 평가 ( evaluation )

---
###### workflow : Collection - Preprocessing - Modeling - Prediction - Evaluation

</br></br></br></br></br></br></br></br></br></br></br></br></br>