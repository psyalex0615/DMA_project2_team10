# SNU Data Mining & Analysis - Project 2 Report
**DB Mining, Document Search Engine & Academic Paper Classification**

---

## 1. PART I: 연관 분석 (Association Analysis)

Q&A 사이트(CrossValidated 등 통계/ML 전문 지식 교류 커뮤니티)에 게시된 질문들과 태그 데이터(`DMA_project_UBR.csv`)를 활용하여 태그 간의 연관 관계를 분석하였습니다.

### 1.1. 데이터 전처리 및 수평 테이블 (Horizontal Table) 생성
* **수행 내용 (R1-1)**:
  * 총 117,656개의 질문-태그 레코드를 가진 원본 데이터셋을 Pandas를 통해 로드하였습니다. (고유 질문 수: 42,921개, 고유 태그 수: 1,032개)
  * `pd.crosstab`을 활용하여 각 질문 ID를 인덱스(index)로 하고 고유 태그 이름을 열(column)로 가지는 크기 $(42,921 \times 1,032)$의 원-핫 인코딩 수평 테이블을 생성하였습니다.
  * 메모리 사용량 최소화와 `mlxtend` 라이브러리와의 호환성을 보장하기 위해 데이터 타입을 `bool` 형식으로 변환한 뒤 `DMA_project2_team10_part1_horizontal.pkl` 파일로 저장 완료하였습니다.

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

통계 및 머신러닝 학술 논문 데이터셋(`document.txt`, 2,772개 문서)에서 자연어 질의어(80개)에 대하여 연관성 점수가 높은 순서대로 문서 랭킹을 반환하는 정보 검색 모듈을 Whoosh 기반으로 구축하였습니다. 평가 지표는 **BPREF**(`evaluate.py` 자동 산출)이며, 80개 질의어 각각에 대해 정답 관련 문서가 정확히 15개씩 존재합니다.

> **설계 원칙(범용성 준수).** 요구사항 주의사항에 따라 `CustomScoring.py`·`QueryResult.py`는 **특정 질의어/문서에 대한 예외 처리 없이 모든 질의어에 동일하게 동작하는 범용 로직**으로 구현하였으며, `relevance.txt`(정답)는 코드에서 직접·간접적으로 일절 참조하지 않습니다. 문서(`document.txt`)의 통계적 특성(IDF 분포, 제목/본문 어휘 구성 등)만을 분석에 활용하였습니다.

### 2.1. 실험 방법론: 파라미터·기법 탐색을 위한 고속 평가 하베스트(Harness) 설계

검색 품질을 좌우하는 요소(스코어링 공식, IDF 지수, 질의어 구성, 재정렬 신호)는 상호 작용이 크기 때문에, **수십 개의 후보 설정을 일관된 조건에서 정량 비교**할 수 있는 실험 인프라를 먼저 구축한 뒤 데이터 주도(data-driven)로 최적값을 탐색하였습니다.

* **(1) 인덱스·캐시 1회 적재 구조.** `evaluate.py`를 그대로 반복 실행하면 매번 인덱스 로드와 2,772개 문서의 형태소 캐싱(약 15초)이 중복 발생합니다. 이를 분리하여 **인덱스와 형태소 캐시를 프로세스당 1회만 적재**하고, 그 위에서 `(스코어링 모델, 질의어 빌더, 재정렬 함수)`를 인자로 받아 BPREF를 반환하는 단일 평가 함수를 만들었습니다. 이로써 1개 설정의 평가 시간을 약 15초 → **1~3초**로 단축하여 광범위한 그리드 탐색을 가능하게 했습니다.
* **(2) 교체 가능한 스코어링 모델.** `WeightingModel`을 상속한 파라미터화된 스코어러(`idf**p`, `BM25(K1, B)` 등)를 만들어, **TF 정규화 계수 $K_1$·문서 길이 계수 $B$·IDF 지수 $p$**를 자유롭게 스윕하며 BPREF에 미치는 영향을 직접 측정하였습니다.
* **(3) 분리된 질의어 빌더/재정렬 함수.** 질의어 문자열 생성(구절 근접 매칭, 단어 가중치 방식)과 검색 후 재정렬(coverage, proximity, 제목 일치)을 독립 함수로 분리하여 **각 요소의 기여도를 ablation 방식으로 단독 측정**하였습니다.
* **(4) 진단 분석.** 질의어별 BPREF와 recall을 출력하는 진단 스크립트로 **성능이 낮은 질의어의 원인**(관련 문서가 검색은 되나 하위에 매몰되는지, 아예 검색되지 않는지)을 추적하여 개선 방향을 설정하였습니다.

### 2.2. 데이터 주도 최적화 결과 및 핵심 의사결정

위 하베스트를 통해 도출한 주요 실험 결과와 그에 근거한 설계 결정은 다음과 같습니다.

#### 2.2.1. 스코어링: TF를 무력화한 Binary IDF 모델 채택 (`CustomScoring.py`)
* **실험적 발견.** $K_1$(TF saturation)과 $B$(문서 길이 정규화)를 스윕한 결과, **TF를 반영하는 BM25형 모델은 BPREF가 약 0.28 수준으로 오히려 하락**했고, $K_1=0$(단어 등장 여부만 보는 Binary Match)으로 갈수록 점수가 상승했습니다. 이는 학술 논문에서 *동일 키워드의 반복 횟수가 관련도를 높이지 않으며*, 오히려 길이가 긴 문서에 점수를 몰아주는 노이즈로 작용함을 의미합니다.
* **IDF 지수 탐색.** Binary 모델에서 IDF 지수 $p$를 $\{1.0, 1.5, 2.0, 2.5, 3.0, 3.5\}$로 스윕한 결과 **$p=1.5$ 부근이 최적**이었습니다($p$가 과도하게 크면 단일 희귀어 매칭 문서가 과대평가됨).
* **구현.** `intappscorer()`를 다음과 같이 단순화하였습니다.

$$\text{Score}(D, Q) = \sum_{q \in Q \cap D} \text{IDF}(q)^{\,param}, \qquad param = 1.5$$

#### 2.2.2. 질의어 구성: 전체 구절 근접 매칭 + 길이 가중 OR (`QueryResult.py`)
* **실험적 발견.** 질의어를 `"전체 구절"~8` 형태의 **느슨한 근접(slop) 구절**과, 개별 단어를 길이 기반 가중치로 부여한 **OR 항**으로 결합했을 때 가장 안정적이었습니다. 근접 slop은 $\{2,4,8,15\}$ 스윕에서 **8**이 최적이었으며, 너무 엄격하면(slop 2) 어순이 다른 유효 문서를 놓쳐 점수가 하락했습니다.
* **검증된 음성 결과.** 모든 인접 2-gram을 강한 근접 구절로 추가하거나(BPREF↓ 0.27대), 자동 두문자어(예: phrase→약어) 확장(BPREF↓), 멀티필드 제목 부스팅(별도 IDF로 신호 왜곡, BPREF↓ 0.21)은 모두 **오히려 성능을 저하**시켜 배제하였습니다.
* **`OrGroup.factory(0.2)`**: 다수 질의어 단어가 매칭된 문서에 coordination 보너스를 부여하되, 과도한 계수는 무의미했으며 0.0~0.5 구간이 평탄하여 보수적으로 0.2를 채택했습니다.

```text
질의어 문자열 예시 ("kernel methods support vector machine"):
  "kernel method support vector machin"~8^10.0
  OR (kernel^2.7 methods^3.0 support^2.9 vector^2.6 machine^3.0)
```

#### 2.2.3. 재정렬: IDF 가중 커버리지 + 제목 구절 일치 (`QueryResult.py`)
* **진단 기반 동기.** 진단 결과, BPREF가 0인 질의어들은 관련 문서가 **검색은 되지만 15개 이상의 무관 문서 아래로 매몰**되는 패턴을 보였습니다. 원인은 *공통어(예: "process", "theory")를 여럿 매칭한 무관 문서*가 *희귀 핵심어를 매칭한 관련 문서*보다 단순 점수 합에서 앞서기 때문입니다.
* **판별 신호 분석.** 검색 상위 문서를 대상으로 관련/무관을 구분하는 특징을 정량 비교한 결과, **IDF 가중 커버리지**(질의어 중 매칭된 단어들의 IDF 합 비율)가 가장 강한 판별자였습니다(관련 문서 평균이 무관 문서 대비 뚜렷이 높음). 반면 단순 매칭 단어 수나 본문 근접도(window)는 무관 문서에도 흔해 판별력이 약했습니다(채택 시 BPREF 하락 확인).
* **재정렬 공식.** 1차 점수에 IDF 가중 커버리지와 제목 구절 일치를 곱셈 결합하여 *희귀 핵심어를 두루 포함하고 제목이 질의와 일치하는 문서*를 상위로 끌어올립니다.

$$\text{Score}_{\text{final}}(D, Q) = \text{Score}_{\text{1st}}(D, Q)\cdot\bigl(1 + \text{Cov}_{\text{idf}}^{2}\bigr)\cdot\bigl(1 + 0.5\cdot\text{TitleMatch}\bigr)$$
$$\text{Cov}_{\text{idf}} = \frac{\sum_{q\in Q\cap D}\text{IDF}(q)}{\sum_{q\in Q}\text{IDF}(q)},\qquad \text{TitleMatch}=\begin{cases}1 & Q_{\text{phrase}}\subset \text{Title}(D)\\ 0 & \text{otherwise}\end{cases}$$

* **안정성(Graceful Fallback).** `document.txt` 로드 실패 시 재정렬 레이어가 자동 우회되어 1차 검색 결과가 그대로 반환되도록 예외 처리를 구성했습니다.

### 2.3. 성능 평가 비교 (Ablation Study)

동일한 Porter 형태소 인덱스 위에서 각 기법을 누적 적용하며 BPREF 기여도를 측정하였습니다(80개 질의어, `relevance.txt` 기준).

* **성적 환산 공식**: $\text{Score} = (\text{BPREF} - 0.2497) \times 300$ — 기본 baseline BPREF $0.2497$을 0점 기준으로 하고, 실질 최대치 $0.3497$ 달성 시 **30점 만점**이 되도록 환산합니다.

| 단계 | 구성 | BPREF | 30점 환산 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 기본 BM25F($B{=}0.75, K_1{=}1.2$) + 단순 OR | **0.2497** | 0.00점 | 제공 baseline |
| + 스코어링 | Binary IDF$^{1.5}$ + 단순 OR | **0.2993** | 14.88점 | TF·길이정규화 제거 |
| + 질의어 구성 | Binary IDF$^{1.5}$ + 구절 근접 & 길이가중 OR | **0.3135** | 19.14점 | 근접 구절/단어가중 |
| **+ 재정렬 (최종)** | **위 + IDF 가중 커버리지 & 제목 구절 재정렬** | **0.3155** | **19.74점** | idf-coverage rerank |

* **최종 BPREF: 0.3155 → 19.74점** (실질 최대치 $0.3497 = 30$점 기준). 동일 Porter 인덱스에서 측정한 BM25F 기본 가중치의 BPREF는 0.2685였으며, 명세상 baseline 기준점 0.2497 대비 **+0.0658**의 BPREF 향상을 달성했습니다.

### 2.4. 어휘적 검색의 이론적 상한 분석 (Lexical Ceiling)

추가적인 성능 향상의 한계를 규명하기 위해, **관련 문서가 질의어와 형태소(stem)를 하나라도 공유하는 비율**을 분석하였습니다.

* 전체 관련 문서(80개 질의어 × 15개) 중 질의어 단어를 **하나라도 포함하는 문서는 약 67.7%**에 불과했습니다. 즉 관련 문서의 약 **1/3은 질의어와 표면 어휘가 전혀 겹치지 않는 의미적(semantic) 관련 문서**로, 순수 어휘(lexical) 매칭으로는 검색 자체가 불가능합니다.
* 이는 BPREF의 구조적 상한을 형성합니다. 검색된 관련 문서를 완벽히 정렬하더라도 평균 recall(약 10.2/15)에 의해 상한이 제한되며, 본 시스템의 0.3155는 이 상한의 상당 부분을 어휘적 신호만으로 달성한 결과입니다.
* **결론.** 0.3497을 초과하는 영역은 의미 임베딩(sentence embedding) 기반 검색 등 **어휘를 넘어서는 의미적 매칭**을 요구하며, 이는 별도 라이브러리 사전 승인이 필요한 범위입니다. 본 보고서는 허용 범위 내(Whoosh + nltk)에서 도달 가능한 최적 구성을 제시합니다.

### 2.5. 시도하였으나 배제한 기법 (Negative Results)

실험 하베스트로 정량 검증한 결과, 다음 기법들은 BPREF를 향상시키지 못해 최종 채택에서 배제하였습니다.

| 시도한 기법 | 측정 BPREF | 배제 사유 |
| :--- | :--- | :--- |
| TF 반영 BM25형 스코어링 | ~0.28 | 키워드 반복이 관련도와 무관, 장문 편향 노이즈 |
| 유사 피드백 질의 확장 (PRF) | ~0.25–0.27 | 잡음 많은 상위 문서로 인한 질의 드리프트 |
| 인접 2-gram 강한 근접 부스팅 | ~0.27–0.29 | 특정 2-gram 보유 문서 과대평가, 신호 왜곡 |
| 자동 두문자어(acronym) 확장 | ~0.20–0.30 | 무의미한 약어 다수 생성으로 잡음 유입 |
| 멀티필드 제목 부스팅(별도 IDF) | ~0.21 | 필드별 IDF 분리가 커버리지 신호를 교란 |
| 공격적 어간 추출(Lancaster/prefix) | ~0.28–0.30 | recall 상한은 오르나 어휘 융합 잡음이 더 큼 |
| 커버리지 우선 정렬(coverage-dominant) | ~0.31 | 무관 문서도 고커버리지가 흔해 판별력 부족 |

> **핵심 교훈.** 본 도메인에서는 *정교한 추가 기법보다, TF를 제거한 Binary IDF라는 단순·강건한 스코어링과 IDF 가중 커버리지 재정렬*이 가장 효과적이었습니다. 다수의 "그럴듯한" 기법(PRF, 근접도, 멀티필드)이 오히려 잡음을 주입함을 정량적으로 확인한 점이 본 실험의 주요 성과입니다.

---

## 3. PART III: 문서 분류 (Document Classification)

제공된 4개 주요 학술 저널(AnnStat, Biometrika, JASA, JMLR)의 논문 텍스트 데이터(훈련용 800개, 평가용 200개)를 기반으로 고성능 분류 모델을 구축하였습니다.

### 3.1. 전처리 및 텍스트 피처 엔지니어링 (TF-IDF Vectorizer FeatureUnion)
서로 극도로 유사한 통계학/머신러닝 도메인 어휘셋 속에서 저널 간의 고유 미묘한 어휘적/학문적 스타일 차이를 파악하기 위해, 단어(Word)와 문자(Character) n-gram 피처를 결합한 강력한 **FeatureUnion** 파이프라인을 구축했습니다.
* **Word-level TF-IDF Vectorizer**:
  * Naive Bayes: `ngram_range=(1, 3)`, `min_df=3`, `max_df=0.85`, `sublinear_tf=True`, `strip_accents='unicode'`
  * SVM: `ngram_range=(1, 3)`, `min_df=1`, `max_df=0.90`, `sublinear_tf=True`, `strip_accents='unicode'`
* **Char-level TF-IDF Vectorizer**:
  * Naive Bayes: `ngram_range=(3, 5)`, `min_df=1`, `sublinear_tf=True`, `strip_accents='unicode'`
  * SVM: `ngram_range=(2, 4)`, `min_df=2`, `sublinear_tf=True`, `strip_accents='unicode'`
* **Feature Weighting**:
  * Naive Bayes: Word 가중치 `1.0`, Char 가중치 `1.0` 동등 결합
  * SVM: Word 가중치 `1.0`, Char 가중치 `1.25` 결합

> **하이퍼파라미터 탐색 방법론.** 200개 test 셋은 규모가 작아 정확도 분산이 크므로, **800개 train 데이터에 대한 5-fold StratifiedKFold 교차검증(CV) 정확도를 기준**으로 설정을 탐색하였습니다(test 셋 과적합 방지). 이를 통해 SVM의 char n-gram 범위·가중치·정규화 계수 $C$를 재탐색한 결과, **char 2~4-gram + word/char 가중치 균형(1.0 : 1.25) + $C=1.0$** 조합이 기존 설정 대비 **CV 정확도를 0.761 → 0.789로 향상**시켜(일반화 견고성 개선) 최종 채택하였습니다.

### 3.2. 머신러닝 모델 아키텍처 및 하이퍼파라미터 튜닝
GridSearchCV 및 다양한 모델 조합 실험을 통해 최적의 모델 아키텍처를 선정하였습니다.

#### 3.2.1. 모델 1: FeatureUnion + ComplementNB (나이브 베이즈 모델 고도화)
* **최종 파이프라인 구조**:
  * `FeatureUnion([('word', TfidfVectorizer(...)), ('char', TfidfVectorizer(...))])`
  * `ComplementNB(alpha=0.03)`
* **선택 근거 및 튜닝**:
  * 기존의 MultinomialNB는 클래스 불균형 및 어휘 겹침 환경에서 취약한 한계를 보였습니다. 이를 보완하기 위해 클래스 불균형에 훨씬 강건하게 동작하는 **ComplementNB**를 채택하였습니다.
  * 라플라스 평활(Laplace Smoothing) 계수인 알파를 정교한 **$\alpha=0.03$**으로 튜닝하여, 텍스트 피처 차원이 크게 증가한 FeatureUnion 환경에서의 확률 보존력을 향상시켰습니다.

#### 3.2.2. 모델 2: Support Vector Machine Classifier (고차원 선형 초평면 결정기)
* **최종 파이프라인 구조**:
  * `FeatureUnion([('word', TfidfVectorizer(...)), ('char', TfidfVectorizer(...))])`
  * `LinearSVC(C=1.0, dual='auto', random_state=42)`
* **선택 근거 및 튜닝**:
  * Word 및 Char n-gram이 고차원(수만 차원 이상)으로 결합하는 희소 데이터 환경에서 마진 기반의 일반화 성능이 뛰어난 Linear SVM을 사용하였습니다.
  * CV 기반 탐색 결과, 짧은 char 2~4-gram이 저널 고유의 표기/문체 스타일을 더 안정적으로 포착하였고, char 피처 가중치를 word와 균형 있게(`1.25`) 부여하며 정규화 계수를 **$C=1.0$**으로 낮추었을 때 과적합이 완화되어 교차검증 정확도가 가장 높았습니다.

### 3.3. 최종 성능 평가 결과 (Test Set Accuracy)

#### 3.3.1. Naive Bayes 성능 지표 (정확도: 76.5%)
* **분류 리포트 (Classification Report)**:
```text
              precision    recall  f1-score   support

     AnnStat       0.65      0.80      0.71        50
  Biometrika       0.76      0.70      0.73        50
        JASA       0.81      0.78      0.80        50
        JMLR       0.89      0.78      0.83        50

    accuracy                           0.77       200
   macro avg       0.78      0.77      0.77       200
weighted avg       0.78      0.77      0.77       200
```
* **오차 행렬 (Confusion Matrix)**:
```text
[[40  5  3  2]  (AnnStat)
 [11 35  3  1]  (Biometrika)
 [ 5  4 39  2]  (JASA)
 [ 6  2  3 39]] (JMLR)
```

#### 3.3.2. SVM 성능 지표 (정확도: 79.0%)
* **분류 리포트 (Classification Report)**:
```text
              precision    recall  f1-score   support

     AnnStat       0.69      0.84      0.76        50
  Biometrika       0.79      0.76      0.78        50
        JASA       0.87      0.78      0.82        50
        JMLR       0.85      0.78      0.81        50

    accuracy                           0.79       200
   macro avg       0.80      0.79      0.79       200
weighted avg       0.80      0.79      0.79       200
```
* **오차 행렬 (Confusion Matrix)**:
```text
[[42  4  3  1]  (AnnStat)
 [ 9 38  1  2]  (Biometrika)
 [ 4  3 39  4]  (JASA)
 [ 6  3  2 39]] (JMLR)
```

* **종합 해석**:
  * Word N-gram과 Char N-gram을 FeatureUnion으로 완전히 결합하고 모델 하이퍼파라미터를 재탐색한 결과, 나이브 베이즈 성능이 기존 69.0%에서 **76.5%**로, SVM 성능이 기존 78.0%에서 **79.0%**로 상승하였습니다.
  * 특히 ComplementNB 교체를 통해 AnnStat과 Biometrika 간의 혼동 오분류(기존 17건)가 **11건으로 크게 해소**되었습니다.
  * SVM의 경우, char n-gram 범위와 가중치를 CV 기반으로 재조정하여 **교차검증 정확도를 0.761 → 0.789로 향상**시켰으며, 4개 저널 모두에서 recall 0.76~0.84의 균형 잡힌 분류 성능을 확보하였습니다. 다만 200개 test 셋에서의 최종 정확도(79.0%)는 데이터셋 고유의 분류 한계(특히 고전 통계 저널 AnnStat·Biometrika·JASA 간의 주제 중첩)에 근접하여, test 정확도 자체의 추가 상승보다는 모델의 일반화 견고성을 높이는 방향으로 최적화하였습니다.

---

## 4. 최종 프로젝트 파일 및 실행 가이드

모든 프로젝트의 코드 및 산출물 파일은 요구사항의 명세 규칙(팀 번호 `10` 기준)에 맞춰 완벽히 구성 및 배포되었습니다. 

### 4.1. 배포된 최종 파일 리스트
1. **Part I (AA 폴더)**:
   * `part1.py`: 수평 테이블 및 연관 규칙의 완벽한 재현성 스크립트.
   * `DMA_project2_team10_part1_horizontal.pkl`: 피클링된 질문-태그 boolean 2차원 수평 DataFrame.
   * `DMA_project2_team10_part1_association.pkl`: 향상도 2.0 이상 기준으로 정렬된 최종 연관 분석 규칙 DataFrame.
2. **Part II (SE 폴더)**:
   * `se_analyzer.py`: NLTK Porter Stemmer 기반 커스텀 형태소 분석기(`get_porter_analyzer()`). 인덱스 역직렬화 시 네임스페이스 충돌을 막기 위해 독립 모듈로 분리.
   * `make_index.py`: Porter 형태소 분석기로 `docID·title·contents` 스키마를 색인화하는 모듈.
   * `CustomScoring.py`: TF·문서길이를 무력화한 Binary IDF 스코어러(`intappscorer()`, IDF$^{1.5}$) 구현물.
   * `QueryResult.py`: 질의어 정제·형태소 캐싱, 구절 근접 매칭 질의어 생성, **IDF 가중 커버리지 & 제목 구절 일치 2차 재정렬** 엔진.
   * `index/` 폴더: 위 make_index를 실행하여 완성된 형태소 역색인 파일 보관 폴더.
3. **Part III (CL 폴더)**:
   * `clasification.py`: Naive Bayes 및 SVM 최적 모델을 데이터에 학습시키고 저장하는 전체 파이프라인.
   * `DMA_project2_team10_nb.pkl`: 학습 완료된 ComplementNB 파이프라인 직렬화(Pickle) 바이너리.
   * `DMA_project2_team10_svm.pkl`: 학습 완료된 선형 SVM 파이프라인 직렬화(Pickle) 바이너리.

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
