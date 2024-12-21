import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
from typing import TypedDict

from constant import basic_boundary_score
from constant import category_sales_statistics
from constant import dict_of_industry
from constant import dict_of_prefecture
from constant import prefectures_list

import joblib
import lightgbm as lgb
model_file_path = "./var/files/basic_score_model.txt"
sales_model_path = "./var/files/sales_prediction_model.joblib"

sales_model = joblib.load(sales_model_path)
loaded_model = lgb.Booster(model_file=model_file_path)
LOOP_COUNT_LIMIT = 1000


class BasicPredictResultScore(TypedDict):
    basic_score: int


def predict(
    registered_address: str,
    capital_fund: int,
    established_year_month: str,
    main_category_name: str,
    category_count: int,
    recent_num_employees: int,
    num_employees_march: int,
    num_employees_may: int,
) -> BasicPredictResultScore:
    period = _calculate_years(established_year_month)
    category_info = _get_category_sales_info(main_category_name, dict_of_industry, category_sales_statistics)
    prefecture = _extract_prefecture(registered_address)
    prefecture_info = _get_prefecture_sales_info(prefecture, dict_of_prefecture)
    diversification = category_count

    result = _get_bankrupcy_probability(capital_fund, period, diversification, category_info, prefecture_info, recent_num_employees, num_employees_march, num_employees_may)
    basic_score = _get_basic_score(result["bankruptcy_probability"])


    return BasicPredictResultScore(
        basic_score=basic_score,
    )

def _get_basic_score(bankruptcy_probability: float) -> int | None:
    for i in range(len(basic_boundary_score) - 1):
        if basic_boundary_score[i] <= bankruptcy_probability <= basic_boundary_score[i + 1]:
            return 500 - i
    return None


def _extract_prefecture(address: str) -> str | None:
    if isinstance(address, str):
        for prefecture in prefectures_list:
            if prefecture in address:
                return prefecture
    return None


def _calculate_years(established_year_month: str | None) -> int | None:
    if not established_year_month:
        return None
    date_formats = [
        "%Y/%m/%d",
        "%Y/%m",
        "%Y",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]
    est_date = None
    for date_format in date_formats:
        try:
            est_date = datetime.strptime(established_year_month, date_format).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    if est_date is None:
        msg = f"This established_year_month is invalid: {established_year_month}"
        raise ValueError(msg) from None
    now = datetime.now(timezone.utc)
    return now.year - est_date.year - 1


def _get_rating(y_pred_prob: float) -> str:
    ranges = [
        (0, 0.0014575274532251184),
        (0.0014575274532251184, 0.0017450058541299688),
        (0.0017450058541299688, 0.002362195845509499),
        (0.002362195845509499, 0.003668949185041362),
        (0.003668949185041362, 0.0063123134902336436),
        (0.0063123134902336436, 0.01228082631512945),
        (0.01228082631512945, 0.029377423887577737),
        (0.029377423887577737, 0.0711530564828329),
        (0.0711530564828329, 1),
    ]
    ratings = [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
    ]
    for i, (range_min, range_max) in enumerate(ranges):
        if range_min <= y_pred_prob < range_max:
            return ratings[i]
    return None


def _get_category_sales_info(main_category_name: str, dict_of_industry: dict, category_sales_statistics: dict) -> dict:
    if main_category_name in dict_of_industry:
        parent_of_main_name = dict_of_industry.get(main_category_name)
    else:
        return {
            "median_category_sales": None,
            "min_category_sales": None,
            "max_category_sales": None,
            "avg_category_sales": None,
        }

    category_sales_data = category_sales_statistics.get(parent_of_main_name, {})
    median_category_sales = category_sales_data.get("CATEGORY_MEDIAN_SALES")
    min_category_sales = category_sales_data.get("CATEGORY_MIN_SALES")
    max_category_sales = category_sales_data.get("CATEGORY_MAX_SALES")
    avg_category_sales = category_sales_data.get("CATEGORY_AVG_SALES")
    return {
        "median_category_sales": float(median_category_sales),
        "min_category_sales": float(min_category_sales),
        "max_category_sales": float(max_category_sales),
        "avg_category_sales": float(avg_category_sales),
    }


def _get_prefecture_sales_info(prefecture: str, dict_of_prefecture: dict) -> dict:
    if prefecture not in dict_of_prefecture:
        return {
            "median_prefecture_sales": None,
            "min_prefecture_sales": None,
            "max_prefecture_sales": None,
            "avg_prefecture_sales": None,
        }
    prefecture_data = dict_of_prefecture.get(prefecture, {})
    median_prefecture_sales = prefecture_data.get("MEDIAN_SALES_on_PREFECTURE")
    min_prefecture_sales = prefecture_data.get("MIN_PREFECTURE_SALES")
    max_prefecture_sales = prefecture_data.get("MAX_PREFECTURE_SALES")
    avg_prefecture_sales = prefecture_data.get("AVG_PREFECTURE_SALES")

    return {
        "median_prefecture_sales": float(median_prefecture_sales),
        "min_prefecture_sales": float(min_prefecture_sales),
        "max_prefecture_sales": float(max_prefecture_sales),
        "avg_prefecture_sales": float(avg_prefecture_sales),
    }


def _get_sales(capital_fund: int, avg_category_sales: float, recent_num_employees: int, period: int, median_category_sales: float, max_category_sales: float, min_category_sale: float) -> float:
    dataflame_for_sales = pd.DataFrame(
        {
            "CAPITAL_FUND": [capital_fund],
            "CATEGORY_AVG_SALES": [avg_category_sales],
            "NUM_EMPLOYEES": [recent_num_employees],
            "BUSINESS_YEARS": [period],
            "CATEGORY_MEDIAN_SALES": [median_category_sales],
            "CATEGORY_MAX_SALES": [max_category_sales],
            "CATEGORY_MIN_SALES": [min_category_sale],
        },
    )

    return sales_model.predict(dataflame_for_sales)


def _get_bankrupcy_probability(
    capital_fund: int | None,
    period: int,
    diversification: int,
    category_info: dict,
    prefecture_info: dict,
    recent_num_employees: int,
    num_employees_march: int,
    num_employees_may: int,
) -> dict:
    x = pd.DataFrame(
        {
            "CAPITAL_FUND": [int(capital_fund) if capital_fund is not None else np.nan],
            "PERIOD": [int(period) if period is not None else np.nan],
            "CATEGORY_COUNT": [int(diversification) if diversification is not None else np.nan],
            "MEDIAN_CATEGORY_SALES": [float(category_info.get("median_category_sales")) if category_info.get("median_category_sales") is not None else np.nan],
            "MIN_CATEGORY_SALES": [float(category_info.get("min_category_sales")) if category_info.get("min_category_sales") is not None else np.nan],
            "MAX_CATEGORY_SALES": [float(category_info.get("max_category_sales")) if category_info.get("max_category_sales") is not None else np.nan],
            "AVG_CATEGORY_SALES": [float(category_info.get("avg_category_sales")) if category_info.get("avg_category_sales") is not None else np.nan],
            "MEDIAN_PREFECTURE_SALES": [float(prefecture_info.get("median_prefecture_sales")) if prefecture_info.get("median_prefecture_sales") is not None else np.nan],
            "MIN_PREFECTURE_SALES": [float(prefecture_info.get("min_prefecture_sales")) if prefecture_info.get("min_prefecture_sales") is not None else np.nan],
            "MAX_PREFECTURE_SALES": [float(prefecture_info.get("max_prefecture_sales")) if prefecture_info.get("max_prefecture_sales") is not None else np.nan],
            "AVG_PREFECTURE_SALES": [float(prefecture_info.get("avg_prefecture_sales")) if prefecture_info.get("avg_prefecture_sales") is not None else np.nan],
            "NUM_EMPLOYEES_OCTOBER": [int(recent_num_employees) if recent_num_employees is not None else np.nan],
            "NUM_EMPLOYEES_MARCH": [int(num_employees_march) if num_employees_march is not None else np.nan],
            "NUM_EMPLOYEES_MAY": [int(num_employees_may) if num_employees_may is not None else np.nan],
            "NEW_EMPLOYEES_NUM": [int(num_employees_may - num_employees_march) if num_employees_may is not None and num_employees_march is not None else np.nan],
        },
    )

    x["SALES"] = _get_sales(
        x["CAPITAL_FUND"],
        x["AVG_CATEGORY_SALES"],
        x["NUM_EMPLOYEES_OCTOBER"],
        x["PERIOD"],
        x["MEDIAN_CATEGORY_SALES"],
        x["MAX_CATEGORY_SALES"],
        x["MIN_CATEGORY_SALES"],
    )

    x["NEW_EMPLOYEES_RATE"] = x.apply(
        lambda row: np.nan if row["NUM_EMPLOYEES_MARCH"] == 0 else row["NUM_EMPLOYEES_MAY"] / row["NUM_EMPLOYEES_MARCH"],
        axis=1,
    )

    x = x.astype(np.float32)
    t_pred_prob = loaded_model.predict(x)
    rating = _get_rating(t_pred_prob)

    return {
        "bankruptcy_probability": float(t_pred_prob.iloc[0] if hasattr(t_pred_prob, "iloc") else t_pred_prob[0]),
        "rank": rating,
        "column_value": x,
    }
