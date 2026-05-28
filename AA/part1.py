import pandas as pd
import pickle
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# !! TODO - 팀 번호를 수정하세요 !!
# ============================================================
TEAM = 10

# ============================================================
# !! TODO - R1-1. 데이터 로드 및 horizontal table 생성 !!
# DMA_project_UBR.csv를 pandas로 로드하고,
# question을 index, tag을 column으로 하는 horizontal table을 생성하세요.
# 필요한 import를 파일 상단에 추가하세요.
# ============================================================
# 데이터 로드
df = pd.read_csv("DMA_project_UBR.csv")
# question을 index, tag를 column으로 하는 horizontal table (one-hot table) 생성
horizontal = pd.crosstab(df["question"], df["tag"]).astype(bool)


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-1 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
horizontal.to_pickle("DMA_project2_team%02d_part1_horizontal.pkl" % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_horizontal.pkl 저장 완료")

# ============================================================
# !! TODO - R1-2. Frequent Itemsets & Association Rules !!
# R1-1에서 만든 horizontal table을 사용하여
# frequent itemset을 만들고 연관분석을 수행하세요.
# - min_support=0.005
# - metric='lift', min_threshold=2.0
# 필요한 import를 파일 상단에 추가하세요.
# ============================================================
# Frequent itemsets 생성 (최소 support = 0.005)
frequent_itemsets = apriori(horizontal, min_support=0.005, use_colnames=True)
# 연관분석 규칙 생성 (metric: lift, min_threshold: 2.0)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2.0)


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-2 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
rules.to_pickle("DMA_project2_team%02d_part1_association.pkl" % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_association.pkl 저장 완료")

# 결과를 화면에 전체 출력하기 위한 Pandas 설정
pd.set_option("display.max_rows", None)  # 행 생략 없이 전체 출력
pd.set_option("display.max_columns", None)  # 열 생략 없이 전체 출력
pd.set_option("display.width", 1000)  # 가로 화면 잘림 방지

print("\n" + "=" * 60)
print(f"★ 1. Frequent Itemsets (총 {len(frequent_itemsets)}개 추출 완료) ★")
print("=" * 60)
# 지지도가 높은 순서대로 출력합니다.
print(frequent_itemsets.sort_values(by="support", ascending=False))

print("\n" + "=" * 80)
print(f"★ 2. Association Rules (총 {len(rules)}개 추출 완료) ★")
print("=" * 80)
# 향상도(lift)가 높은 순서대로 중요 지표만 선택하여 출력합니다.
printed_rules = rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].sort_values(by="lift", ascending=False)
print(printed_rules)

print("\nPart I 완료!")
