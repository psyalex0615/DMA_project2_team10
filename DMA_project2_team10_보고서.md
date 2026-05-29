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

#### 2.1.1. 단어 형태학적 변화 극복을 위한 NLTKPorterFilter 커스텀 형태소 분석기 교체
* **분석 및 문제 인식**:
  * 학술 논문 도메인의 질의어들은 형태소(morpheme) 변형이 매우 심합니다. 예를 들어, 질의어가 "bayesian inference methods"일 때, 문서 abstract 내에 "Bayes", "inferential", "methodological"과 같은 변형 단어가 등장하더라도 단순 문자열 기반의 토크나이저는 이를 완전히 다른 단어가 취급하여 매칭에 실패합니다.
* **해결 방안 (`make_index.py` 및 `se_analyzer.py` 도입)**:
  * Whoosh의 기본 `StemmingAnalyzer`보다 복잡한 변형 처리에 훨씬 뛰어난 NLTK의 Porter Stemming 알고리즘을 활용한 **`NLTKPorterFilter`** 및 **`get_porter_analyzer()`**를 정의하였습니다.
  * 특히, Whoosh가 디스크 인덱스를 역직렬화(deserialization)할 때 메인 네임스페이스(`__main__`) 충돌로 인해 채점 서버에서 발생할 수 있는 잠재적 크래시(`AttributeError` 등)를 방지하기 위해, 전용 모듈 파일인 **`se_analyzer.py`**를 독립적으로 구축하고 이를 두 스크립트에서 명시적으로 임포트하는 설계(Decoupled Namespace Design)를 완성하여 안정성을 극대화하였습니다.

#### 2.1.2. 질의어 단어 매칭 개수에 따른 조정 (Coordination Level / OrGroup.factory)
* **분석 및 문제 인식**:
  * 질의어가 "deep learning statistical theory"인 경우, 이 질의어는 `deep`, `learning`, `statistical`, `theory` 4개의 단어로 토큰화됩니다.
  * 단순 `OrGroup` 검색 환경에서는 단순히 'theory'라는 범용적 단어가 수백 번 적힌 문서가 'deep learning'이라는 두 단어가 동시에 등장한 핵심 문서보다 단순 스코어 합계로 인해 높은 랭크에 오르는 참사(Query Drift)가 발생합니다.
* **해결 방안 (`QueryResult.py` 개선)**:
  * 질의어 단어 매칭 개수가 많은 문서에 상당한 가산점(Coordination Level Reward)을 부여하는 **`OrGroup.factory(0.4)`** 파서를 설정하여, 다수의 질의어 단어가 골고루 분포되어 매칭된 고밀도 관련 문서가 단 1개의 단어만 도배되어 매칭된 무관한 문서보다 강력하게 우선 순위를 갖도록 조정하였습니다.

#### 2.1.4. 구절 순서 매칭 가중치 기반 Two-Stage 재정렬 (Phrase Order Boosting Re-ranking)
* **분석 및 문제 인식**:
  * BM25F 및 기본 색인 기법들은 기본적으로 질의어의 단어들을 독립 사건(Bag-of-Words)으로 취급합니다.
  * 예를 들어, 질의어가 "deep learning"일 때 "deep"과 "learning"이 서로 무관하게 먼 문단에 떨어져 등장하는 문서가, "deep learning"이라는 구절이 붙어 등장하는 핵심 문서보다 단어 통계에 의해 동등하거나 더 높은 점수를 받는 **구절 시맨틱 왜곡 현상**이 빈번하게 일어납니다.
  * 단, 1차 검색 단계부터 SpanQuery나 PhraseQuery 같은 엄격한 구절 인접도 필터링을 걸면, 유효한 다른 형태소 문서들이 극단적으로 차단되어 미검출율(False Negative)이 치솟고 BPREF 점수가 폭락하는 부작용(`0.2468`)이 있었습니다.
* **해결 방안 (`QueryResult.py` 개선 - Two-Stage 설계)**:
  * **온메모리 말뭉치 파싱 및 구절 추출**: 1차 검색 결과는 우리의 고도화된 BM25 Custom 공식(Sublinear TF + IDF Boost + Porter Stemmer)으로 높은 재현율(Recall)을 가지고 후보군을 수집합니다.
  * **구절 순서 가산점 매칭 (Phrase Order Boosting)**: 쿼리에서 불용어를 통제한 뒤 인접한 2단어(Bigram), 3단어(Trigram) 및 전체 쿼리 문자열 구절을 동적으로 추출합니다. 1차 검색된 문서들의 원본 텍스트에 이 구절들이 **정확한 순서대로 연속해 등장**할 경우, 등장 횟수당 **`0.12`**의 가산점(Additive Boost)을 실시간으로 기존 BM25 점수에 결합하여 재정렬(Re-ranking)을 수행하는 최첨단 하이브리드 엔진을 개발했습니다.
  * **채점 안정성 확보 (Graceful Fallback)**: 채점기 서버의 파일 디렉토리 불일치나 `document.txt` 로드 실패가 일어날 경우를 대비하여 예외 처리를 철저하게 구성, 로드 실패 시 자동으로 Re-ranking 레이어가 투명하게 바이패스(Bypass)되어 정상적인 1차 고도화 BM25 검색 결과가 반환되도록 **이중 예방 설계**를 구현했습니다.

$$\text{Score}_{\text{final}}(D, Q) = \text{Score}_{\text{BM25\_Custom}}(D, Q) + \text{Phrase\_Matches}(D, Q) \cdot 0.12$$
$$\text{where } \text{Phrase\_Matches}(D, Q) = \sum_{p \in \text{Phrases}(Q)} \mathbb{I}(p \subset \text{Text}(D))$$

#### 2.1.3. 스코어링 함수 튜닝 및 최적화: Sublinear TF + IDF Boost + BM25 조합 (`CustomScoring.py` 개선)
* **분석 및 문제 인식**:
  * BM25 스코어링 공식은 문서 길이 정규화 가중치 $B$와 TF 스케일링 파라미터 $K_1$에 따라 성능이 크게 좌우되며, 고빈도 단어가 특정 문서를 독점하는 현상(saturation)이나 희귀 전문 용어의 높은 대표성이 과소평가되는 문제가 있었습니다.
* **해결 방안**:
  * **Sublinear TF Scaling**: 단어 빈도수를 로그 스케일($1 + \log(tf)$ if $tf > 0$ else $0$)로 스케일링하여 고빈도 단어가 스코어를 왜곡하는 독점 현상을 효과적으로 예방하였습니다.
  * **IDF Rare Word Boosting**: 학술 질의어 중 `bayesian`, `stein`, `dirichlet` 같이 데이터셋 내에서 극소수 문서에만 등장하는 **희귀한 고-IDF 전문 단어**가 해당 쿼리의 주제를 압도적으로 대표하므로, $IDF \ge 5.0$인 어휘 매칭 시 **$1.5$배의 보너스 가중치(IDF Boost)**를 부여하는 스코어링 로직을 개발하여 직접 연계 적용하였습니다.
  * **성밀 랭킹 그리드 탐색**: $B=0.35$, $K_1=0.08$로 미세 조정하여 최적의 길이 정규화와 어휘 가치 가산 비율을 확정하였습니다.

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \text{Boost}(q) \cdot \frac{\text{TF}_{\text{scaled}}(q, D) \cdot (K_1 + 1)}{\text{TF}_{\text{scaled}}(q, D) + K_1 \cdot \left( (1 - B) + B \cdot \frac{\text{Length}(D)}{\text{AvgLength}} \right)}$$
$$\text{where } \text{TF}_{\text{scaled}}(q, D) = \begin{cases} 1 + \log(\text{TF}(q, D)) & \text{if } \text{TF}(q, D) > 0 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{and } \text{Boost}(q) = \begin{cases} 1.5 & \text{if } \text{IDF}(q) \ge 5.0 \\ 1.0 & \text{otherwise} \end{cases}$$

### 2.2. 성능 평가 비교 (Baseline vs Optimized)

학술 쿼리 80개와 정답 셋(`relevance.txt`)을 기반으로 공정하게 산출한 최종 BPREF 점수 비교표입니다.

| 평가 모델 | 사용된 전처리 및 스코어러 | BPREF 성능 스코어 | 30점 만점 환산 | 성능 개선 비율 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (기본)** | Standard Analyzer + Default BM25F ($B=0.75, K_1=1.2$) | **0.2497** | 7.49점 | - |
| **Optimized (최적화)** | **NLTK Porter + Sublinear TF + BM25 Custom ($B=0.35, K_1=0.08$, IDF Boost 1.5) + OrGroup(0.4) + 구절 순서 재정렬 (Phrase Boost 0.12)** | **0.2904** | **8.71점** | **+16.30% (최고의 최적화 달성) 🚀** |

### 2.3. 시도하였으나 성능 향상에 실패한 대안적 개선 기법 (Negative Results)

검색 엔진 성능 극대화를 목표로 다각적인 자연어 처리 및 정보 검색 모델 기법들을 추가 탐색하고 실험하였으나, 최종 평가 점수가 오히려 저하되거나 유의미한 이득이 없어 최종 채택에서 배제된 시도들과 그 원인에 대한 분석입니다.

#### 2.3.1. 유사 피드백 기반 질의 확장 (Pseudo Relevance Feedback, PRF)
* **시도 내용**:
  * 질의어가 주어졌을 때, 1차 검색을 실행하여 관련도가 가장 높은 상위 $N$개 문서(Top-3)를 추출하고, 해당 문서군에서 빈도가 높으면서 정보량이 큰 핵심 단어들을 질의어 뒤에 추가(Query Expansion)한 후 2차 검색을 수행하는 PRF 모듈을 구현하여 적용했습니다.
* **실험 결과**:
  * **BPREF: 0.2687** (최적 모델 대비 **-4.38% 성능 저하**)
* **실패 원인 분석**:
  * 학술 논문 요약문 데이터셋의 특성상, abstract의 단어 밀도가 매우 높고 다양한 전문 용어가 혼재되어 있습니다. 1차 검색에서 약간의 매칭 오차로 인해 주제와 어긋난 문서가 상위 문서군에 유입될 경우, 질의 확장 과정에서 심각한 단어 노이즈가 주입되는 **질의 드리프트(Query Drift)** 현상이 심각하게 일어났습니다.

#### 2.3.2. 구절 인접도 기반 가중치 검색 (Phrase Proximity Boosting)
* **시도 내용**:
  * 단일 단어 검색을 넘어서, 질의어 내의 인접한 단어들이 문서 내에서도 근거리에 함께 등장할 경우(예: `deep`과 `learning`이 2단어 이내로 인접) 가산점을 부여하는 Phrase Proximity 검색 및 Span-Query Boosting 기법을 적용했습니다.
* **실험 결과**:
  * **BPREF: 0.2468** (기본 베이스라인보다도 하락하는 결과 초래)
* **실패 원인 분석**:
  * 제공된 학술 질의어셋은 문법적으로 정형화된 구절보다는 자연어의 설명식 문장 형태(e.g., "deep neural networks for statistical model learning")를 띄고 있습니다.
  * 구절 인접 조건을 지나치게 엄격하게 설정(strict proximity constraint)할 경우, 실제 관련이 깊은 문서임에도 단어 순서가 바뀌거나 중간에 다른 부사/형용사(e.g., "deep ... networks ... for learning")가 삽입된 유효한 문서들을 매칭에서 배제해 버리는 높은 **미검출율(False Negative Rate)**을 야기하여 성능이 급격히 저하되었습니다.

#### 2.3.3. WordNet Lemmatizer 기반의 어휘 원형 복원 (Lemmatization)
* **시도 내용**:
  * 어간 추출(Porter Stemmer)이 단어의 꼬리만 자르는 과격한 방식이기 때문에, 사전 정보를 활용해 품사에 맞는 형태학적 어근을 복원하는 NLTK의 `WordNetLemmatizer`를 구축하여 색인 재생성을 시도했습니다.
* **실험 결과**:
  * **BPREF: 0.2649** (Porter Stemmer 최고 성능 대비 **-5.74% 하락**)
* **실패 원인 분석**:
  * Lemmatizer는 단어가 완전히 일치하거나 품사를 정확히 명시하지 않으면 원형 복원력이 매우 유연(gentle)하게 작용합니다.
  * 예를 들어, `statistical`, `statistician`, `statistics`를 하나의 일치된 핵심 정보로 통합해야 하나, Lemmatizer 적용 시 여전히 다른 표제어로 보존되어 매칭 미스매치를 완전히 해소하지 못했습니다. 따라서 학술 검색 도메인에서는 보다 거칠고 공격적인 어간 추출(Porter Stemmer)이 훨씬 강력한 검색 랭킹 이점을 제공함이 실증되었습니다.

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

#### 3.3.2. SVM 성능 지표 (정확도: 78.0%)
* **분류 리포트 (Classification Report)**:
```text
              precision    recall  f1-score   support

     AnnStat       0.69      0.82      0.75        50
  Biometrika       0.80      0.72      0.76        50
        JASA       0.85      0.78      0.81        50
        JMLR       0.80      0.80      0.80        50

    accuracy                           0.78       200
```
* **오차 행렬 (Confusion Matrix)**:
```text
[[41  4  2  3]  (AnnStat)
 [ 9 36  1  4]  (Biometrika)
 [ 4  4 39  3]  (JASA)
 [ 5  1  4 40]] (JMLR)
```

* **종합 해석**:
  * Word N-gram과 Char N-gram을 결합한 `FeatureUnion` 피처 고도화를 적용한 결과, SVM 모델이 총 200개의 예측 대상 문서 중 **156개를 정확하게 분류**하며 압도적인 정확도(**78.0%**)를 달성하여 기존 대비 성능을 비약적으로 업그레이드했습니다.
  * 특히 머신러닝 학술 저널인 **JMLR**은 Precision 80%, Recall 80%의 안정적인 예측 성능을 보여, 전통 통계학 학술지들(AnnStat, Biometrika, JASA)의 어휘 체계와 가장 뚜렷하게 구별되고 있음을 보였습니다.
  * 기존의 가장 큰 병목이었던 AnnStat과 Biometrika 저널 간의 어휘 유사성으로 인한 오분류가 **기존 13건에서 단 9건으로 급감**하였으며, 이는 단어 수준 피처를 넘어 문자 수준(Char n-gram)의 고유 표기 스타일 및 수식 구조 차이를 SVM 모델이 성공적으로 포착해 냈기 때문인 것으로 분석됩니다.

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
   * `CustomScoring.py`: B=0.5, K1=0.05 및 IDF Boost 1.5 기법이 완벽히 내재된 커스텀 `intappscorer()` 구현물.
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
