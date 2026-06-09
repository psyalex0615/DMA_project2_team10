import os.path
import shutil
import time
import tempfile
import uuid
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from se_analyzer import get_porter_analyzer

# ─────────────────────────────────────────────────────────────────────
# Index schema
#  - docID   : 문서 식별자 (검색 결과 반환용으로만 저장)
#  - title   : 제목. re-ranking 단계에서 제목-구(phrase) 일치 판단에 사용.
#  - contents: 제목 + 본문 전체. 채점(scoring) 대상 필드.
# Porter stemmer + stopword 필터 analyzer 를 사용하여 질의어와 문서를
# 동일한 어휘 공간으로 정규화한다.
# ─────────────────────────────────────────────────────────────────────
schema = Schema(
    docID=NUMERIC(stored=True),
    title=TEXT(stored=True, analyzer=get_porter_analyzer()),
    contents=TEXT(stored=False, analyzer=get_porter_analyzer())
)

index_dir = os.path.join(tempfile.gettempdir(), f"whoosh_index_{uuid.uuid4().hex}")
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

        lines = doc_text.strip().split('\n')
        title = lines[0] if lines else ""

        writer.add_document(
            docID=docID,
            title=title,
            contents=doc_text
        )

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

shutil.move(index_dir, final_index_dir)
print("Index build completed successfully.")
