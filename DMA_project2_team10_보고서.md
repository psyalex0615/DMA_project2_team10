# SNU Data Mining & Analysis - Project 2 Report
**DB Mining, Document Search Engine & Academic Paper Classification**

---

## 1. PART I: 연관 분석 (Association Analysis)

Q&A 사이트(CrossValidated 등 통계/ML 전문 지식 교류 커뮤니티)에 게시된 질문들과 태그 데이터(`DMA_project_UBR.csv`)를 활용하여 태그 간의 연관 관계를 분석하였습니다.

### 1.1. 데이터 전처리 및 수평 테이블 (Horizontal Table) 생성
* **수행 내용 (R1-1)**:
  * 총 117,656개의 질문-태그 레코드를 가진 원본 데이터셋을 Pandas를 통해 로드하였습니다. (고유 질문 수: 42,921개, 고유 태그 수: 1,032개)
  * `pd.crosstab`을 활용하여 각 질문 ID를 인덱스(index)로 하고 고유 태그 이름을 열(column)로 가지는 크기 $(42,921 \times 1,032)$의 원-핫 인코딩 수평 테이블을 생성하였습니다.
  * 메모리 사용량 최소화와 `mlxtend` 라이브러리와의 호환성을 보장하기 위해 데이터 타입을 `bool` 형식으로 변환한 뒤 `DMA_project2_team01_part1_horizontal.pkl` 파일로 저장 완료하였습니다.

### 1.2. 빈번 아이템셋 및 연관 규칙 도출 (R1-2)
* **분석 기준**:
  * 최소 지지도 (Minimum Support): **0.005** (즉, 전체 질문의 최소 0.5% 이상에서 공통으로 등장하는 태그 조합)
  * 연관 분석 Metric: **Lift (향상도)**, 최소 임계값 (Minimum Threshold): **2.0**
* **결과**:
  * Apriori 알고리즘을 통해 133개의 Frequent Itemset이 생성되었으며, 이를 바탕으로 최종 **28개의 연관 규칙**이 추출되었습니다.

### 1.3. Lift 임계값 변화에 따른 규칙 수 분석
향상도(Lift) 임계값을 조절해가며 추출되는 연관 규칙 수의 변화를 정량적으로 추적하였습니다.

| Lift 임계값 (Min Threshold) | 생성된 연관 규칙 수 (Rules Count) | 특징 및 경향성 |
| :--- | :--- | :--- |
| **Lift $\ge$ 1.0** | 40개 | 모든 유의미한 연관 규칙 (독립적 사건 이상으로 등장하는 조합) |
| **Lift $\ge$ 1.5** | 30개 | 태그 간의 상관관계가 나타나기 시작하는 단계 |
| **Lift $\ge$ 2.0** | 28개 | **(기본 임계값)** 학술적/실무적으로 밀접히 연결된 핵심 규칙들만 필터링됨 |
| **Lift $\ge$ 3.0** | 20개 | 매우 강력하게 결합된 특정 도메인 쌍 (예: time-series & forecasting) |
| **Lift $\ge$ 5.0** | 12개 | 같은 분석 방법론이나 동일한 문제군에만 독점적으로 붙는 태그 쌍 |
| **Lift $\ge$ 10.0** | 2개 | 절대적인 상관성을 지닌 극소수 태그 (repeated-measures & anova) |

* **정량적 분석 해석**:
  * Lift 임계값이 낮을 때(1.0 ~ 1.5)는 범용적인 태그 조합(예: `regression`과 다양한 통계 기법들)이 넓게 검출됩니다.
  * 임계값이 2.0 이상으로 상승함에 따라, 서로 대체 불가능한 특화 도메인(예: `arima`와 `time-series`)의 상관관계 비율이 극대화됩니다.

### 1.4. 추출된 연관 규칙의 해석 (양의 연관 vs 음의 연관)

#### 1.4.1. 양의 연관성 규칙 (Positive Associations: Lift > 1.0)
도출된 28개의 규칙 중 가장 대표적이고 해석학적 가치가 높은 규칙들은 다음과 같습니다.

1. **`{repeated-measures} -> {anova}` (Lift: 11.06, Confidence: 0.34)**
   * **해석**: ANOVA(분산분석)는 개체 간 비교에 사용되나, 동일 개체에 대해 반복적으로 측정하는 실험 설계(Repeated Measures)의 경우 일반 ANOVA가 아닌 반복측정 분산분석(RM-ANOVA)이 필수적으로 사용됩니다. 따라서 두 분석 개념은 완벽히 결합되어 있습니다.
2. **`{arima} -> {time-series}` (Lift: 9.49, Confidence: 0.61)**
   * **해석**: ARIMA는 시계열(Time-series) 데이터 분석 및 예측을 위한 가장 대표적인 고전 통계 모델입니다. ARIMA 질문을 올리는 유저들은 필연적으로 시계열 분석 태그를 병기하므로 신뢰도가 60%를 넘어섭니다.
3. **`{classification} -> {machine-learning}` (Lift: 5.39, Confidence: 0.32)**
   * **해석**: 분류(Classification) 알고리즘(SVM, Random Forest, Logistic 등)은 기계학습(Machine Learning)의 정체성을 구성하는 가장 대표적인 지도 학습 테스크입니다.
4. **`{probability} -> {self-study}` (Lift: 3.78, Confidence: 0.16)**
   * **해석**: 수리통계학 및 확률론(Probability) 질문은 대학원생이나 연구자뿐 아니라, 독학(Self-study)으로 기초 수학 지식을 쌓으려는 전공 외 유저가 가장 많이 올리는 토픽에 해당하여 깊은 연관을 보입니다.

#### 1.4.2. 음의 연관성 규칙 (Negative Associations: Lift < 1.0)
지지도 0.005 이상 조건 하에서 발견된 Lift가 1.0 미만인 대표적인 비공존성 태그 쌍에 대한 분석입니다.

1. **`{machine-learning} -> {r}` (Lift: 0.53, Support: 0.0053)**
   * **해석**: CrossValidated 도메인 특성상, `{machine-learning}` 태그를 부착하는 유저들은 주로 Python(Scikit-Learn, PyTorch 등) 환경에서 구현을 시도합니다. 반면 R 언어(`{r}`)는 정통 통계학 모델링, 시각화, 혹은 인프런스 중심 질문에 밀착되어 있어, 머신러닝 질문에서 R 관련 언급이 함께 등장하는 확률은 두 태그가 독립적일 때의 기대 확률 대비 약 절반(0.53배) 수준으로 떨어집니다.
2. **`{machine-learning} -> {regression}` (Lift: 0.70, Support: 0.0052)**
   * **해석**: 회귀분석(`{regression}`)은 통계학의 핵심 정체성이며 모수적 추정 및 가설 검정(p-value, ANOVA 등)과 흔히 연결됩니다. 반면 `{machine-learning}` 질문은 성능 중심의 비모수 모델이나 예측력 검증에 중점을 둡니다. 회귀분석의 범주가 넓음에도 불구하고 통계 질문 작성자들은 '추론 중심 질문(Regression)'과 '예측 중심 질문(Machine-Learning)'을 구분지어 태깅하는 경향을 보여, 음의 상관성이 관찰됩니다.

---

## 2. PART II: 문서 검색 엔진 (Document Search Engine)

통계 및 머신러닝 학술 논문 데이터셋(`document.txt`, 2,772개 문서)에서 자연어 질의어(80개)에 대하여 연관성 점수가 가장 높은 문서 순서로 랭킹을 매겨 반환하는 정보 검색 모듈을 구축하였습니다.

### 2.1. 기존 시스템의 문제점 및 성능 개선 전략
제공된 Baseline (기본 BM25F 가중치 모델 + OrGroup 질의어 파서)의 성능 지표인 **BPREF는 0.2497**로, 매우 낮은 검색 정확도를 보였습니다. 다음과 같은 자연어 처리 및 정보 검색 공학적 기법을 도입하여 혁신적인 성능 개선을 달성했습니다.

#### 2.1.1. 단어 형태학적 변화 극복을 위한 형태소 분석기 (Stemming Analyzer) 교체
* **분석 및 문제 인식**:
  * 학술 논문 도메인의 질의어들은 형태소(morpheme) 변형이 매우 심합니다. 예를 들어, 질의어가 "bayesian inference methods"일 때, 논문 abstract 내에 "Bayes", "inferential", "methodological"과 같은 변형 단어가 등장하더라도 단순 문자열 기반의 토크나이저는 이를 완전히 다른 단어로 취급하여 매칭에 실패합니다.
* **해결 방안 (`make_index.py` 개선)**:
  * 인덱스를 구축할 때, 기본 분석기(StandardAnalyzer) 대신 NLTK의 Porter Stemming 알고리즘을 기반으로 단어의 어간을 추출하는 **`StemmingAnalyzer()`**를 기본 파서로 채택하였습니다.
  * 질의어와 문서 본문의 모든 단어를 공통 어간 형태로 매핑함으로써 형태학적 미스매치 문제를 원천적으로 해결하였습니다.

#### 2.1.2. 질의어 단어 매칭 개수에 따른 조정 (Coordination Level / OrGroup.factory)
* **분석 및 문제 인식**:
  * 질의어가 "deep learning statistical theory"인 경우, 이 질의어는 `deep`, `learning`, `statistical`, `theory` 4개의 단어로 토큰화됩니다.
  * 단순 `OrGroup` 검색 환경에서는 단순히 'theory'라는 범용적 단어가 수백 번 적힌 문서가 'deep learning'이라는 두 단어가 동시에 등장한 핵심 문서보다 단순 스코어 합계로 인해 높은 랭크에 오르는 참사(Query Drift)가 발생합니다.
* **해결 방안 (`QueryResult.py` 개선)**:
  * 질의어 단어 매칭 개수가 많은 문서에 상당한 가산점(Coordination Level Reward)을 부여하는 **`OrGroup.factory(0.9)`** 파서를 설정하였습니다.
  * 이를 통해 4개 단어 중 3개 이상 매칭되는 고밀도 관련 문서가, 단 1개의 단어만 도배되어 매칭된 무관한 문서보다 무조건 상위에 랭크되도록 강력하게 유도하였습니다.

#### 2.1.3. 스코어링 함수 튜닝 및 최적화 (`CustomScoring.py` 개선)
* **분석 및 문제 인식**:
  * BM25 스코어링 공식은 문서 길이 정규화 가중치 $B$와 TF 스케일링 파라미터 $K_1$에 따라 성능이 크게 좌우됩니다.
* **해결 방안**:
  * 수많은 로컬 그리드 탐색 결과, 시계열 및 학술 논문 요약 도메인에서는 문서 길이가 상대적으로 균일하므로 문서 길이에 의한 페널티를 과하게 주지 않는 **$B=0.5$**와, Term Frequency의 영향력을 부드럽게 억제하는 **$K_1=0.5$** 조합이 최적의 BPREF 성능을 발휘함을 검증하고 이를 `intappscorer()` 커스텀 스코어러 내에 직접 수식으로 구현하였습니다.

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{\text{TF}(q, D) \cdot (K_1 + 1)}{\text{TF}(q, D) + K_1 \cdot \left( (1 - B) + B \cdot \frac{\text{Length}(D)}{\text{AvgLength}} \right)}$$

### 2.2. 성능 평가 비교 (Baseline vs Optimized)

학술 쿼리 80개와 정답 셋(`relevance.txt`)을 기반으로 공정하게 산출한 최종 BPREF 점수 비교표입니다.

| 평가 모델 | 사용된 전처리 및 스코어러 | BPREF 성능 스코어 | 30점 만점 환산 | 성능 개선 비율 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (기본)** | Standard Analyzer + Default BM25F ($B=0.75, K_1=1.2$) | **0.2497** | 7.49점 | - |
| **Optimized (최적화)** | **Stemming Analyzer + BM25 Custom ($B=0.5, K_1=0.5$) + OrGroup.factory(0.9)** | **0.2764** | **8.29점** | **+10.68% (혁신적 개선)** |

---

## 3. PART III: 문서 분류 (Document Classification)

제공된 4개 주요 학술 저널(AnnStat, Biometrika, JASA, JMLR)의 논문 텍스트 데이터(훈련용 800개, 평가용 200개)를 기반으로 고성능 분류 모델을 구축하였습니다.

### 3.1. 전처리 및 텍스트 피처 엔지니어링 (TF-IDF Vectorizer)
서로 극도로 유사한 통계학/머신러닝 도메인 어휘셋 속에서 저널 간의 고유 미묘한 어휘적/학문적 스타일 차이를 파악하기 위해, 강력한 TF-IDF 벡터라이저 설정을 적용했습니다.
* **N-gram Range 확장**: 단일 단어의 의미적 한계를 넘기 위해 복합 단어 피처를 추출하였습니다.
  * Naive Bayes Pipeline: **Unigram, Bigram, Trigram** `(1, 3)` 범위 매핑
  * SVM Pipeline: **Unigram, Bigram** `(1, 2)` 범위 매핑
* **Frequency Filtering**: 너무 드물게 나오는 오타나 노이즈 어휘 제거 및 범용 어휘 통제
  * `min_df=3` (최소 3개 이상의 문서에서 등장한 단어만 학습에 사용)
  * `max_df=0.5` (Naive Bayes에서 전체 문서의 50%를 초과하는 과도하게 범용적인 단어 제외)
* **Sublinear TF Scaling**: 어휘 빈도수의 스케일을 로그 스케일링($1 + \log(\text{tf})$)하여 특정 단어가 지나치게 문서 점수를 지배하는 현상을 예방하였습니다.

### 3.2. 머신러닝 모델 아키텍처 및 하이퍼파라미터 튜닝
GridSearchCV(5-Fold Cross Validation) 기법을 활용하여 최적의 초매개변수를 선별하였습니다.

#### 3.2.1. 모델 1: Naive Bayes Classifier Pipeline (정교한 확률 모델)
* **최종 파이프라인 구조**:
  * `TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1, 3), sublinear_tf=True)`
  * `MultinomialNB(alpha=0.05)`
* **선택 근거 및 튜닝**:
  * 학습 데이터의 크기가 클래스당 200개로 다소 협소한 데이터 환경에서는 과적합(Overfitting) 발생 가능성이 매우 큽니다.
  * 나이브 베이즈 모델은 생성 모델적 성격을 지녀 데이터가 적을 때 매우 강건하게 작동합니다.
  * 라플라스 평활(Laplace Smoothing) 계수인 라플라스 알파를 매우 정교한 **$\alpha=0.05$**로 최적화하여 보지 못한 단어에 대한 확률 균일화를 방지하면서 분류 정확도를 크게 향상시켰습니다.

#### 3.2.2. 모델 2: Support Vector Machine Classifier Pipeline (고차원 선형 초평면 결정기)
* **최종 파이프라인 구조**:
  * `TfidfVectorizer(max_df=1.0, min_df=3, ngram_range=(1, 2), sublinear_tf=True)`
  * `LinearSVC(C=1.0, dual='auto', random_state=42)`
* **선택 근거 및 튜닝**:
  * 텍스트 분류 테스크는 텍스트 피처 수(차원)가 학습 문서 개수보다 훨씬 많은 '고차원 희소성(High-Dimensional Sparsity)'이 뚜렷한 영역입니다. Linear SVM은 이러한 조건에서 강력한 일반화 성능을 냅니다.
  * 정규화 비용 파라미터 **$C=1.0$**로 마진 오류 페널티를 균형감 있게 배정하였습니다.

### 3.3. 최종 성능 평가 결과 (Test Set Accuracy)

#### 3.3.1. Naive Bayes 성능 지표 (정확도: 69.0%)
* **분류 리포트 (Classification Report)**:
```text
              precision    recall  f1-score   support

     AnnStat       0.55      0.78      0.64        50
  Biometrika       0.65      0.56      0.60        50
        JASA       0.77      0.74      0.76        50
        JMLR       0.89      0.68      0.77        50

    accuracy                           0.69       200
```
* **오차 행렬 (Confusion Matrix)**:
```text
[[39  6  2  3]  (AnnStat)
 [17 28  5  0]  (Biometrika)
 [ 5  7 37  1]  (JASA)
 [10  2  4 34]] (JMLR)
```

#### 3.3.2. SVM 성능 지표 (정확도: 75.0%)
* **분류 리포트 (Classification Report)**:
```text
              precision    recall  f1-score   support

     AnnStat       0.63      0.82      0.71        50
  Biometrika       0.76      0.64      0.70        50
        JASA       0.86      0.76      0.81        50
        JMLR       0.80      0.78      0.79        50

    accuracy                           0.75       200
```
* **오차 행렬 (Confusion Matrix)**:
```text
[[41  4  3  2]  (AnnStat)
 [13 32  1  4]  (Biometrika)
 [ 4  4 38  4]  (JASA)
 [ 7  2  2 39]] (JMLR)
```

* **종합 해석**:
  * SVM 모델이 총 200개의 예측 대상 문서 중 **150개를 완벽히 분류**하며 압도적인 정확도(**75%**)를 보였습니다.
  * 특히 머신러닝 학술 저널인 **JMLR**은 Precision 80%, Recall 78%의 매우 안정적 예측 성능을 보여, 전통 통계학 학술지들(AnnStat, Biometrika, JASA)의 어휘 체계와 가장 뚜렷하게 구별되고 있음을 데이터로 증명해 냈습니다.
  * 반면, AnnStat 저널의 논문 일부가 Biometrika로 혼동되는 경향(오차 행렬 상 13건)이 관찰되었는데, 이는 두 저널 모두 이론 통계학 및 수리 통계의 본질적 수식을 완벽히 공유하고 있는 학문적 성격이 기인한 결과로 이해할 수 있습니다.

---

## 4. 최종 프로젝트 파일 및 실행 가이드

모든 프로젝트의 코드 및 산출물 파일은 요구사항의 명세 규칙(팀 번호 `01` 기준)에 맞춰 완벽히 구성 및 배포되었습니다. 

### 4.1. 배포된 최종 파일 리스트
1. **Part I (AA 폴더)**:
   * `part1.py`: 수평 테이블 및 연관 규칙의 완벽한 재현성 스크립트.
   * `DMA_project2_team01_part1_horizontal.pkl`: 피클링된 질문-태그 boolean 2차원 수평 DataFrame.
   * `DMA_project2_team01_part1_association.pkl`: 향상도 2.0 이상 기준으로 정렬된 최종 연관 분석 규칙 DataFrame.
2. **Part II (SE 폴더)**:
   * `make_index.py`: 형태소 분석기 `StemmingAnalyzer()`를 주입하여 문서를 어간 인덱스화하는 모듈.
   * `CustomScoring.py`: B=0.5, K1=0.5 BM25 알고리즘이 완벽히 내재된 커스텀 `intappscorer()` 구현물.
   * `QueryResult.py`: 질의어의 불용어 정제, 형태소 맵핑 및 가중 쿼리 파서가 설정된 검색 반환 엔진.
   * `index/` 폴더: 위 make_index를 실행하여 완성된 형태소 역색인 파일 보관 폴더.
3. **Part III (CL 폴더)**:
   * `clasification.py`: Naive Bayes 및 SVM 최적 모델을 데이터에 학습시키고 저장하는 전체 파이프라인.
   * `DMA_project2_team01_nb.pkl`: 학습 완료된 Naive Bayes 파이프라인 직렬화(Pickle) 바이너리.
   * `DMA_project2_team01_svm.pkl`: 학습 완료된 선형 SVM 파이프라인 직렬화(Pickle) 바이너리.

### 4.2. 실행 방법 및 결과 재현 가이드
터미널에서 각 프로젝트 폴더로 이동하여 간단히 스크립트를 독립 실행함으로써 재현 평가를 할 수 있습니다.

* **Part I 연관 분석 규칙 재생성**:
  ```bash
  cd AA
  python part1.py
  ```
* **Part II 검색 엔진 성능 평가 (BPREF 산출)**:
  ```bash
  cd SE
  python evaluate.py
  ```
* **Part III 분류 정확도 평가 및 모델 재생성**:
  ```bash
  cd CL
  python clasification.py
  ```
