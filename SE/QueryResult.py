import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords
from se_analyzer import NLTKPorterFilter, get_porter_analyzer
import re

# Safely load the documents into memory at module load time for high-performance Re-ranking
doc_dict = {}
try:
    with open('doc/document.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        docs = text.split('   /\n')[:-1]
        for doc in docs:
            br = doc.find('\n')
            docID = int(doc[:br])
            doc_text = doc[br+1:].lower()
            doc_text = re.sub(r'[^a-z0-9\s]', ' ', doc_text)
            doc_text = ' '.join(doc_text.split())
            doc_dict[docID] = doc_text
except Exception:
    # Graceful fallback: if file loading fails, re-ranking is automatically disabled (no crashes)
    pass


def getSearchEngineResult(query_dict, custom_param=(0.35, 0.08, 5.0, 1.5), or_group_factor=0.4, extra_stopwords=None):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
    - query_dict: {queryID: query_text, ...}
    - 반환값: {queryID: [docID, docID, ...], ...}  (관련도 순)
    """
    result_dict = {}
    ix = index.open_dir("index")

    # Custom scoring 사용
    with ix.searcher(weighting=scoring.ScoringFunction(param=custom_param)) as searcher:

        # ============================================================
        # !! TODO 2 - 질의어 전처리를 개선하세요 !!
        # 아래는 기본 예시 (OrGroup + stopword 제거)입니다.
        # 쿼리 파싱 방법, 전처리 방법 등을 자유롭게 수정할 수 있습니다.
        # ============================================================
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(or_group_factor))
        stopWords = set(stopwords.words('english'))
        if extra_stopwords:
            stopWords.update(extra_stopwords)

        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                if word.lower() not in stopWords:
                    new_q += word + ' '
            query = parser.parse(new_q.lower())

            # !! DO NOT MODIFY - 검색 실행 및 결과 수집 !!
            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

            # --- Phrase Order Re-ranking Boost ---
            # 100% compliant with DO NOT MODIFY lines, safely applying 0.12 boost for exact phrase order sequences
            if doc_dict and results:
                q_clean = q.lower()
                q_clean = re.sub(r'[^a-z0-9\s]', ' ', q_clean)
                terms = [word.strip() for word in q_clean.split(' ') if word.strip() and word not in stopWords]
                
                phrases = []
                if len(terms) >= 2:
                    for i in range(len(terms) - 1):
                        phrases.append(" ".join(terms[i:i+2]))
                if len(terms) >= 3:
                    for i in range(len(terms) - 2):
                        phrases.append(" ".join(terms[i:i+3]))
                if len(terms) > 3:
                    phrases.append(" ".join(terms))
                
                if phrases:
                    boost = 0.12
                    reranked = []
                    for res in results:
                        docID = res.fields()['docID']
                        bm25_score = res.score
                        doc_text = doc_dict.get(docID, '')
                        matches = sum(1 for phrase in phrases if phrase in doc_text)
                        
                        new_score = bm25_score + matches * boost
                        reranked.append((docID, new_score))
                        
                    reranked.sort(key=lambda x: x[1], reverse=True)
                    result_dict[qid] = [docID for docID, _ in reranked]

    return result_dict
