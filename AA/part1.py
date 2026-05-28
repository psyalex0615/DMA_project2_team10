import pandas as pd
import pickle

# ============================================================
# !! TODO - 팀 번호를 수정하세요 !!
# ============================================================
TEAM = 0

# ============================================================
# !! TODO - R1-1. 데이터 로드 및 horizontal table 생성 !!
# DMA_project_UBR.csv를 pandas로 로드하고,
# question을 index, tag을 column으로 하는 horizontal table을 생성하세요.
# 필요한 import를 파일 상단에 추가하세요.
# ============================================================
horizontal = None  # TODO: horizontal table (DataFrame) 생성


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-1 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
horizontal.to_pickle('DMA_project2_team%02d_part1_horizontal.pkl' % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_horizontal.pkl 저장 완료")

# ============================================================
# !! TODO - R1-2. Frequent Itemsets & Association Rules !!
# R1-1에서 만든 horizontal table을 사용하여
# frequent itemset을 만들고 연관분석을 수행하세요.
# - min_support=0.005
# - metric='lift', min_threshold=2.0
# 필요한 import를 파일 상단에 추가하세요.
# ============================================================
rules = None  # TODO: association rules (DataFrame) 생성


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-2 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
rules.to_pickle('DMA_project2_team%02d_part1_association.pkl' % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_association.pkl 저장 완료")

print("\nPart I 완료!")
