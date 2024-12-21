import pandas as pd
import json
import ml  # ml.pyをインポート

df = pd.read_csv("_WITH_random_companies_AS_SELECT_id_FROM_companies_WHERE_establi_202412211737.csv")

results = []

for _, row in df.iterrows():
    # ml.predictを使ってスコアを取得
    capital_fund = row.get("capital_fund")
    capital_fund = int(capital_fund) if pd.notna(capital_fund) else 0

    num_employees_updates = row.get("num_employees_updates")

    # JSON文字列をパースしてリストに変換
    if num_employees_updates:
        try:
            updates = json.loads(num_employees_updates)
        except json.JSONDecodeError:
            updates = []
    else:
        updates = []

    # 3月と5月の社員数データを取得
    num_employees_march = max(
        (update.get('num') for update in updates if '-03-' in update.get('date', '')), 
        default=0
    )
    num_employees_may = max(
        (update.get('num') for update in updates if '-05-' in update.get('date', '')), 
        default=0
    )

    # 最新の社員数を取得
    recent_num_employees = updates[-1].get('num') if updates else 0

    # ml.predictを呼び出してスコアを取得
    basic_result = ml.predict(
        registered_address=str(row.get("registered_address") or ""),
        capital_fund=capital_fund,
        established_year_month=str(row.get("established_year_month") or ""),
        main_category_name=str(row.get("main_category_name") or ""),
        category_count=int(row.get("category_count") or 0),
        recent_num_employees=int(recent_num_employees or 0),
        num_employees_march=int(num_employees_march or 0),
        num_employees_may=int(num_employees_may or 0),
    )
    
    # 取得した結果を行に追加
    for key, value in basic_result.items():
        row[key] = value  # 各スコアをカラムとして追加
    
    results.append(row)

# basic_scoreだけを抽出してデータフレームに変換
df_basic_score = pd.DataFrame([{"basic_score": row["basic_result"]["basic_score"]} for row in results])

# 必要に応じてCSVに保存
df_basic_score.to_csv("basic_score_only.csv", index=False)