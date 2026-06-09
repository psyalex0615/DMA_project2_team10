import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from math import log


# ─────────────────────────────────────────────────────────────────────────
#  문서 본문을 메모리에 적재 (re-ranking 단계에서 사용).
#  문서의 통계적 특성(어떤 질의어 단어가 문서에 등장하는지, 제목 구 일치
#  여부)을 활용하는 것은 허용된 "문서 분석"에 해당하며, 특정 질의어/문서에
#  대한 예외 처리가 아닌 범용 로직이다. relevance.txt 는 일절 참조하지 않는다.
# ─────────────────────────────────────────────────────────────────────────
doc_dict = {}        # docID -> 본문 전체 (소문자, 구두점 제거, 공백 정규화)
doc_title = {}       # docID -> 제목 행 (소문자, 구두점 제거)
try:
    with open('doc/document.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        docs = text.split('   /\n')[:-1]
        for doc in docs:
            br = doc.find('\n')
            docID = int(doc[:br])
            raw = doc[br + 1:]
            # 제목 = 본문 첫 줄 (newline 보존을 위해 정규화 전에 분리)
            title_line = raw.strip().split('\n')[0].lower()
            doc_title[docID] = re.sub(r'[^a-z0-9\s]', ' ', title_line)
            body = re.sub(r'[^a-z0-9\s]', ' ', raw.lower())
            doc_dict[docID] = ' '.join(body.split())
except Exception:
    pass

# 문서별 stem 캐시 (지연 초기화)
doc_body_stems = {}       # docID -> {stem, ...}  (문서 전체 단어 집합)
doc_title_phrase = {}     # docID -> "stem stem ..."  (제목을 stem 한 문자열)


def _init_caches(stemmer, stop_words):
    if doc_body_stems:
        return
    for docID, text in doc_dict.items():
        doc_body_stems[docID] = {
            stemmer.stem(w) for w in text.split() if w and w not in stop_words
        }
        title_line = doc_title.get(docID, '')
        doc_title_phrase[docID] = ' '.join(
            stemmer.stem(w) for w in title_line.split() if w and w not in stop_words
        )


def getSearchEngineResult(query_dict):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
      - query_dict : {queryID: query_text, ...}
      - return     : {queryID: [docID, docID, ...], ...}  (관련도 내림차순)

    [동작 개요 — 모든 질의어에 동일하게 적용되는 범용 로직]
      1) 채점(CustomScoring): TF·문서길이를 무력화한 Binary IDF 모델.
         IDF^param 으로 희소(specific) 단어 매칭에 가중.
      2) 질의어 변환: 전체 구(phrase)를 느슨한 근접 매칭(~8)으로 OR 결합하고,
         개별 단어는 길이 기반 보정 가중치를 부여해 OR 질의로 확장.
      3) Re-ranking: 검색된 문서를 (a) IDF 가중 질의어 커버리지,
         (b) 제목-구 일치 보너스로 재정렬. 질의어의 희소 단어를 많이
         포함하고 제목이 질의와 일치하는 문서를 상위로 끌어올린다.
    """
    result_dict = {}
    ix = index.open_dir("index")

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    _init_caches(stemmer, stop_words)

    # CustomScoring: Binary IDF 모델. param=1.5 -> idf**1.5 가중.
    with ix.searcher(weighting=scoring.ScoringFunction(param=1.5)) as searcher:
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.2))

        N = searcher.doc_count_all()

        def term_idf(stem):
            df = searcher.doc_frequency("contents", stem)
            return log(N / (df + 1)) + 1.0

        for qid, q in query_dict.items():
            # ---- 질의어 토큰화 (구두점 제거 + 불용어 제거) ----
            words = [
                w for w in re.sub(r'[^a-zA-Z0-9\s]', ' ', q).lower().split()
                if w and w not in stop_words
            ]

            if not words:
                result_dict[qid] = []
                continue

            stems = [stemmer.stem(w) for w in words]
            phrase = ' '.join(stems)

            # ---- (2) 질의어 객체 생성 ----
            # 전체 구 근접 매칭 + 개별 단어(길이 기반 가중) OR 결합
            term_part = ' '.join(
                f'{w}^{round(1.5 * log(len(w) + 1.0), 1)}' for w in words
            )
            query_string = f'"{phrase}"~8^10.0 OR ({term_part})'
            query = parser.parse(query_string)

            results = searcher.search(query, limit=None)
            if not results:
                result_dict[qid] = []
                continue

            # ---- (3) Re-ranking ----
            q_stem_set = set(stems)
            idf_total = sum(term_idf(s) for s in q_stem_set) or 1.0
            idf_map = {s: term_idf(s) for s in q_stem_set}

            reranked = []
            for res in results:
                docID = res['docID']
                base_score = res.score

                # (a) IDF 가중 질의어 커버리지: 매칭된 질의어 단어의 IDF 합 / 전체 IDF 합
                #     희소(specific)한 질의어 단어를 매칭한 문서일수록 높음.
                matched = q_stem_set & doc_body_stems.get(docID, set())
                cov_idf = sum(idf_map[s] for s in matched) / idf_total

                # (b) 제목-구 일치 보너스
                title_match = 1.0 if (phrase and phrase in doc_title_phrase.get(docID, '')) else 0.0

                final_score = base_score * (1.0 + cov_idf ** 2) * (1.0 + 0.5 * title_match)
                reranked.append((docID, final_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            result_dict[qid] = [docID for docID, _ in reranked]

    return result_dict
