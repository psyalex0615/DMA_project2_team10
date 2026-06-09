import numpy as np
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
import CustomScoring as scoring
from nltk.corpus import stopwords
import sys

def readQueryFile(filename):
    query_dict = {}
    with open(filename, 'r') as f:
        text = f.read()
        queries = text.split('   /\n')
        for query in queries:
            if not query.strip():
                continue
            br = query.find('\n')
            qID = int(query[:br])
            q = query[br+1:]
            query_dict[qID] = q
    return query_dict

def getGroundtruthRelevance(query_ids):
    relevant_dict = {}
    with open('doc/relevance.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            items = line.split(' ')
            queryID = int(items[0])
            docID = int(items[1])
            if queryID in query_ids:
                if queryID in relevant_dict:
                    relevant_dict[queryID].append(docID)
                else:
                    relevant_dict[queryID] = [docID]
    return relevant_dict

def evaluate_bpref(query_dict, relevent_dict, results_dict):
    BPREF = []
    for queryID in query_dict.keys():
        if queryID not in results_dict or queryID not in relevent_dict:
            BPREF.append(0.0)
            continue
        relevantCount = 0
        nonRelevantCount = 0
        score = 0
        results = results_dict[queryID]
        relevantDocuments = relevent_dict[queryID]
        relDocCount = len(relevantDocuments)
        
        if relDocCount == 0:
            BPREF.append(0.0)
            continue

        for document in results:
            if document in relevantDocuments:
                relevantCount += 1
                if nonRelevantCount >= relDocCount:
                    score += 0
                else:
                    score += (1 - nonRelevantCount / relDocCount)
            else:
                nonRelevantCount += 1
            if relevantCount == relDocCount:
                break
        score = score / relDocCount
        BPREF.append(score)
    return np.mean(BPREF)

def run_grid_search(ix, query_dict, relevant_dict, B_candidates, K1_candidates, org_candidates, stopwords_options):
    best_score = -1
    best_params = {}
    
    for sw_name, sw in stopwords_options.items():
        # Pre-process queries for this stopword set
        stopWords = set(stopwords.words('english'))
        if sw:
            stopWords.update(sw)
        
        preprocessed_queries = {}
        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                if word.lower() not in stopWords:
                    new_q += word + ' '
            preprocessed_queries[qid] = new_q.lower()
            
        for org in org_candidates:
            parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(org))
            # Parse all queries once
            parsed_queries = {qid: parser.parse(q_text) for qid, q_text in preprocessed_queries.items()}
            
            for B in B_candidates:
                for K1 in K1_candidates:
                    results_dict = {}
                    with ix.searcher(weighting=scoring.ScoringFunction(param=(B, K1))) as searcher:
                        for qid, query in parsed_queries.items():
                            results = searcher.search(query, limit=None)
                            results_dict[qid] = [result.fields()['docID'] for result in results]
                            
                    score = evaluate_bpref(query_dict, relevant_dict, results_dict)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'B': B,
                            'K1': K1,
                            'or_group_factor': org,
                            'stopwords': sw_name
                        }
                        print(f"  -> Found better score! B={B:.3f}, K1={K1:.3f}, OrGroup={org:.3f}, Stopwords={sw_name} => BPREF: {best_score:.6f}")
                        sys.stdout.flush()
                        
    return best_score, best_params

def main():
    print("Loading index and query data...")
    sys.stdout.flush()
    ix = index.open_dir("index")
    query_dict = readQueryFile('doc/query.txt')
    relevant_dict = getGroundtruthRelevance(query_dict.keys())
    
    academic_stopwords = {'using', 'results', 'based', 'study', 'analysis', 'research', 'methods', 'used', 'paper', 'use', 'approach', 'show', 'method', 'provide', 'findings'}
    stopwords_options = {
        'Standard': None,
        'Academic': academic_stopwords
    }
    
    # ============================================================
    # PHASE 1: COARSE GRID SEARCH (넓고 성글게 탐색)
    # ============================================================
    print("\n[PHASE 1] Starting COARSE Grid Search...")
    sys.stdout.flush()
    
    B_coarse = [0.2, 0.5, 0.8]
    K1_coarse = [0.2, 0.5, 1.0, 1.5]
    org_coarse = [0.7, 0.9, 0.98]
    
    coarse_score, coarse_params = run_grid_search(
        ix, query_dict, relevant_dict, 
        B_coarse, K1_coarse, org_coarse, stopwords_options
    )
    
    print(f"\nCoarse Best Score: {coarse_score:.6f}")
    print(f"Coarse Best Params: {coarse_params}")
    sys.stdout.flush()
    
    # ============================================================
    # PHASE 2: FINE-GRAINED SEARCH (최적 지점 주변을 세밀하게 탐색)
    # ============================================================
    print("\n[PHASE 2] Starting FINE-GRAINED Grid Search...")
    sys.stdout.flush()
    
    best_B = coarse_params['B']
    best_K1 = coarse_params['K1']
    best_org = coarse_params['or_group_factor']
    best_sw_name = coarse_params['stopwords']
    
    # Set fine search bounds around best coarse params
    B_fine = [max(0.0, best_B - 0.1), best_B, min(1.0, best_B + 0.1)]
    K1_fine = [max(0.05, best_K1 - 0.1), best_K1, best_K1 + 0.1]
    org_fine = [max(0.5, best_org - 0.05), best_org, min(0.98, best_org + 0.05)]
    
    # Fix the best stopword option
    fine_stopwords_options = {
        best_sw_name: stopwords_options[best_sw_name]
    }
    
    fine_score, fine_params = run_grid_search(
        ix, query_dict, relevant_dict,
        B_fine, K1_fine, org_fine, fine_stopwords_options
    )
    
    print("\n" + "="*50)
    print("COARSE-TO-FINE GRID SEARCH COMPLETED!")
    print(f"Final Best BPREF Score: {fine_score:.6f}")
    print("Optimal Parameters:")
    for k, v in fine_params.items():
        print(f"  - {k}: {v}")
    print("="*50)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
