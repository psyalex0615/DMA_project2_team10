import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords


def getSearchEngineResult(query_dict):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
    - query_dict: {queryID: query_text, ...}
    - 반환값: {queryID: [docID, docID, ...], ...}  (관련도 순)
    """
    result_dict = {}
    ix = index.open_dir("index")

    # Custom scoring 사용
    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # ============================================================
        # !! TODO 2 - 질의어 전처리를 개선하세요 !!
        # 아래는 기본 예시 (OrGroup + stopword 제거)입니다.
        # 쿼리 파싱 방법, 전처리 방법 등을 자유롭게 수정할 수 있습니다.
        # ============================================================
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.9))
        stopWords = set(stopwords.words('english'))

        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                if word.lower() not in stopWords:
                    new_q += word + ' '
            query = parser.parse(new_q.lower())

            # !! DO NOT MODIFY - 검색 실행 및 결과 수집 !!
            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict
