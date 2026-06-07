import os.path
import shutil
import time
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from se_analyzer import NLTKPorterFilter, get_porter_analyzer, get_snowball_analyzer

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT(analyzer=get_snowball_analyzer()))
index_dir = "index_temp"

if os.path.exists(index_dir):
    try:
        shutil.rmtree(index_dir)
    except Exception:
        pass
        
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)
writer = ix.writer()
 
with open('doc/document.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    docs = text.split('   /\n')[:-1]
    for doc in docs:
        br = doc.find('\n')
        docID = int(doc[:br])
        doc_text = doc[br+1:]
        writer.add_document(docID=docID, contents=doc_text)

writer.commit()

# 빌드 성공 후 index 폴더 원자적 대체
final_index_dir = "index"
if os.path.exists(final_index_dir):
    try:
        shutil.rmtree(final_index_dir)
    except Exception:
        try:
            os.rename(final_index_dir, f"index_old_{int(time.time())}")
        except Exception:
            pass
            
os.rename(index_dir, final_index_dir)
print("Index build completed successfully.")