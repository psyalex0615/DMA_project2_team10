import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords
from se_analyzer import NLTKPorterFilter, get_porter_analyzer
import re
from math import log


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
    pass

# Lazy pre-stemmed caches
doc_bodies_stemmed = {}
doc_titles_stemmed = {}
doc_titles_phrase_stemmed = {}

def init_stemmed_caches(stemmer):
    global doc_bodies_stemmed, doc_titles_stemmed, doc_titles_phrase_stemmed
    if doc_bodies_stemmed:
        return
    for docID, text in doc_dict.items():
        # Clean doc text has title on the first line
        lines = text.strip().split('\n')
        title = lines[0] if lines else ""
        doc_bodies_stemmed[docID] = {stemmer.stem(w) for w in text.split() if w}
        doc_titles_phrase_stemmed[docID] = " ".join([stemmer.stem(w) for w in title.split() if w])
        doc_titles_stemmed[docID] = {stemmer.stem(w) for w in title.split() if w}

def getSearchEngineResult(query_dict):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
    - query_dict: {queryID: query_text, ...}
    - 반환값: {queryID: [docID, docID, ...], ...}  (관련도 순)
    """
    result_dict = {}
    ix = index.open_dir("index")

    # [1] 어휘 캐싱 및 스태머 준비
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    init_stemmed_caches(stemmer)
    
    # 학술 용어 노이즈 단어 정의 (OR 매칭 단어들에서만 걸러내어 노이즈 유입 차단)
    stopWords = set(stopwords.words('english'))
    academic_stopwords = {'using', 'results', 'based', 'study', 'analysis', 'research', 'methods', 'used', 'paper', 'use', 'approach', 'show', 'method', 'provide', 'findings'}
    strict_stopWords = stopWords.union(academic_stopwords)

    # Custom scoring (param=1.5로 idf**2.5 Binary Match 적용)
    # OrGroup factor = 0.2로 단어 매칭 개수별 가중치 최적화
    with ix.searcher(weighting=scoring.ScoringFunction(param=1.5)) as searcher:
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.2))

        for qid, q in query_dict.items():
            cleaned_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', q)
            words = cleaned_q.split()
            
            # (A) 구절(Phrase)용 단어 추출 - 표준 stopword만 제거하여 구문 맥락 유지
            phrase_words = []
            for word in words:
                cleaned_word = word.lower().strip()
                if cleaned_word not in stopWords and cleaned_word != '':
                    phrase_words.append(cleaned_word)
                    
            # (B) 개별 단어(OR)용 단어 추출 - strict stopword(학술노이즈 포함) 제거 및 글자 수 비선형 부스팅
            term_queries = []
            strict_term_words = []
            for word in words:
                cleaned_word = word.lower().strip()
                if cleaned_word not in strict_stopWords and cleaned_word != '':
                    strict_term_words.append(cleaned_word)
                    auto_boost = round(log(len(cleaned_word) + 1.0) * 1.5, 1)
                    term_queries.append(f"{cleaned_word}^{auto_boost}")
            
            # strict filter로 다 지워질 경우, 일반 stopword 기준으로 복구 (Fallback)
            if not term_queries:
                for word in phrase_words:
                    auto_boost = round(log(len(word) + 1.0) * 1.5, 1)
                    term_queries.append(f"{word}^{auto_boost}")
                    strict_term_words.append(word)
            
            if not phrase_words and not term_queries:
                result_dict[qid] = []
                continue
                
            base_query = " ".join(phrase_words)
            boosted_terms = " ".join(term_queries)
            
            # (C) 최종 합성: "슬롭 구문 검색 OR (글자수로 차등 가중치 먹인 개별 단어들)"
            if base_query:
                final_query_string = f'"{base_query}"~8^10.0 OR ({boosted_terms})'
            else:
                final_query_string = boosted_terms
                
            query = parser.parse(final_query_string)
            results = searcher.search(query, limit=None) # Retrieve ALL matching documents
            
            if not results:
                result_dict[qid] = []
                continue

            # (D) Reranking using Title Phrase Boost and Body Co-occurrence
            q_phrase_stemmed = " ".join([stemmer.stem(w) for w in phrase_words])
            q_words_stemmed = {stemmer.stem(w) for w in strict_term_words}
            q_len = len(q_words_stemmed)
            
            results_list = [(res['docID'], res.score) for res in results]
            
            reranked = []
            for docID, original_score in results_list:
                title_phrase_stemmed = doc_titles_phrase_stemmed.get(docID, '')
                title_words_stemmed = doc_titles_stemmed.get(docID, set())
                
                # 1. Exact phrase match in title (using phrase_title_boost = 1.0)
                phrase_match = 0.0
                if q_phrase_stemmed and q_phrase_stemmed in title_phrase_stemmed:
                    phrase_match = 1.0
                
                boost_multiplier = 1.0 + phrase_match
                
                # 2. Co-occurrence match in body
                doc_words_stemmed = doc_bodies_stemmed.get(docID, set())
                body_matches = sum(1 for qw in q_words_stemmed if qw in doc_words_stemmed)
                body_ratio = (body_matches / q_len) if q_len > 0 else 0.0
                body_co_boost = 1.0 + 0.5 * (body_ratio ** 2)
                
                final_score = original_score * body_co_boost * boost_multiplier
                reranked.append((docID, final_score))
                
            reranked.sort(key=lambda x: x[1], reverse=True)
            result_dict[qid] = [docID for docID, _ in reranked]

    return result_dict


    return result_dict
