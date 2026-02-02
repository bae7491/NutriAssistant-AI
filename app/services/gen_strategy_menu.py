import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# [수정됨] langchain.prompts -> langchain_core.prompts
from langchain_core.prompts import ChatPromptTemplate

from app.models.analysis_trends import TrendAnalysisResult
from app.models.strategies import (
    NutritionCompliance,
    QualityScorecard,
    RiskForecast,
    MenuStrategyItem,
    MenuStrategyResponse,
)


NUTRI_STANDARDS = {
    "초등(4~6)": {"energy_kcal": 670, "protein_g": 16.7},
    "중학생": {"energy_kcal": 840, "protein_g": 20.0},
    "고등학생": {"energy_kcal": 900, "protein_g": 21.7},
}

ENERGY_RATIO_STANDARDS = {
    "protein": {"min": 0, "max": 0.20},
    "carb": {"min": 0.55, "max": 0.65},
    "fat": {"min": 0.15, "max": 0.30},
}


def _generate_plan(
    llm, point, nutrition_compliance, quality_scorecard, risk_forecast
) -> dict[str, str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
(Response Language: {response_lang})

당신은 급식 운영 데이터를 해석하는 전문가입니다.
""",
            ),
            (
                "human",
                """
다음은 기간 내 제기된 급식 메뉴 운영 상의 문제 혹은 약점과 그에 대한 정보입니다.
 
(지적사항)
{point}
 
(Nutrition Compliance)
{nutrition_compliance}

(Quality Scorecard - in Nutritional Aspects)
{quality_scorecard}
 
(Risk Forecast)
{risk_forecast}
 
당신은 이에 대응하는 전략을 수립합니다.

(다음 형태의 JSON으로 답합니다)
{{
    "trigger": 전략에 대한 '언제',
    "adjustment": 전략에 따른 조정 사항,
    "howToApply": [
        전략에 따른 조정 사항
    ]
}}

<주의>
* 각 사항은 간결하게 한 문장으로 작성합니다.
""",
            ),
        ]
    )
    message = prompt.format_messages(
        response_lang="Korean",
        point=point,
        nutrition_compliance=nutrition_compliance,
        quality_scorecard=quality_scorecard,
        risk_forecast=risk_forecast,
    )
    try:
        response = llm.invoke(message).content.strip()
        return json.loads(response)
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return {"trigger": "", "adjustment": "", "howToApply": []}


def convert_meal_plan_to_df(meal_plan: List[dict]) -> pd.DataFrame:
    """mealPlan JSON -> DataFrame."""
    df = pd.DataFrame(meal_plan)
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["meal_type"] = df["mealType"]

    menu_cols = ["rice", "soup", "main1", "main2", "side", "kimchi", "dessert"]

    def combine_menus(row):
        menus = []
        for col in menu_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                menus.append(str(val).strip())
        return menus

    df["menus"] = df.apply(combine_menus, axis=1)
    df["menu_key"] = df["menus"].apply(lambda x: " | ".join(x))

    df["Kcal"] = pd.to_numeric(df.get("kcal"), errors="coerce")
    df["Prot"] = pd.to_numeric(df.get("protein"), errors="coerce")
    df["Carb"] = pd.to_numeric(df.get("carb"), errors="coerce")
    df["Fat"] = pd.to_numeric(df.get("fat"), errors="coerce")

    return df


def convert_daily_info_to_df(daily_info: List[dict]) -> pd.DataFrame:
    """dailyInfo JSON -> DataFrame."""
    df = pd.DataFrame(daily_info)
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["meal_type"] = df["mealType"]
    df["served_count"] = df["servedProxy"]
    df["leftover_kg"] = df["leftoverKg"]
    df["leftover_rate"] = df["leftover_kg"] / df["served_count"].replace(0, np.nan)
    return df


def convert_daily_analysis_to_df(daily_analysis: List[dict]) -> pd.DataFrame:
    """dailyAnalysis JSON -> DataFrame."""
    records = []
    for item in daily_analysis:
        kpis = item.get("kpis", {})
        records.append(
            {
                "date": item["date"],
                "mealType": item["mealType"],
                "review_count": kpis.get("review_count", 0),
                "post_count": kpis.get("post_count", 0),
                "avg_rating": kpis.get("avg_rating_5", 0),
                "avg_sentiment": kpis.get("avg_review_sentiment", 0),
                "overall_sentiment": kpis.get("overall_sentiment", 0),
                "top_negative_aspects": item.get("top_negative_aspects", []),
                "top_issues": item.get("top_issues", []),
                "alerts": item.get("alerts", []),
            }
        )
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["meal_type"] = df["mealType"]
    return df


def convert_review_analysis_to_df(review_analysis: List[dict]) -> pd.DataFrame:
    """reviewAnalysis JSON -> DataFrame."""
    df = pd.DataFrame(review_analysis)
    meal_type_map = {"LUNCH": "중식", "DINNER": "석식", "BREAKFAST": "조식"}
    df["meal_type"] = df["meal_type"].map(meal_type_map).fillna(df["meal_type"])
    return df


def merge_master_table(
    meal_plan_df: pd.DataFrame,
    daily_info_df: pd.DataFrame,
    daily_analysis_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge meal plan, daily info, and daily analysis."""
    master = meal_plan_df.merge(
        daily_info_df[
            ["Date", "meal_type", "served_count", "leftover_kg", "leftover_rate"]
        ],
        on=["Date", "meal_type"],
        how="left",
    )

    if daily_analysis_df is not None and len(daily_analysis_df) > 0:
        master = master.merge(
            daily_analysis_df[
                [
                    "Date",
                    "meal_type",
                    "avg_rating",
                    "avg_sentiment",
                    "review_count",
                    "top_negative_aspects",
                    "top_issues",
                ]
            ],
            on=["Date", "meal_type"],
            how="left",
        )
    else:
        master["avg_rating"] = 3.5
        master["avg_sentiment"] = 0.5
        master["review_count"] = 0

    return master


def analyze_nutrition_compliance(
    master: pd.DataFrame, target_group: str = ""
) -> NutritionCompliance:
    """Analyze nutrition compliance."""
    standards = NUTRI_STANDARDS.get(
        target_group, {"energy_kcal": 900, "protein_g": 21.7}
    )
    E0 = standards["energy_kcal"]
    P0 = standards["protein_g"]

    master["energy_ok"] = master["Kcal"].between(E0 * 0.9, E0 * 1.1)
    master["protein_ok"] = master["Prot"] >= P0
    master["protein_energy_ratio"] = (master["Prot"] * 4) / master["Kcal"].replace(
        0, np.nan
    )
    master["protein_ratio_ok"] = master["protein_energy_ratio"] <= 0.20
    master["carb_energy_ratio"] = (master["Carb"] * 4) / master["Kcal"].replace(
        0, np.nan
    )
    master["carb_ratio_ok"] = master["carb_energy_ratio"].between(0.55, 0.65)
    master["fat_energy_ratio"] = (master["Fat"] * 9) / master["Kcal"].replace(0, np.nan)
    master["fat_ratio_ok"] = master["fat_energy_ratio"].between(0.15, 0.30)

    master["nutri_compliance_ok"] = (
        master["energy_ok"]
        & master["protein_ok"]
        & master["protein_ratio_ok"]
        & master["carb_ratio_ok"]
        & master["fat_ratio_ok"]
    )

    total_meals = len(master)

    compliance = {
        "energy": {
            "count": int(master["energy_ok"].sum()),
            "rate": round(master["energy_ok"].mean() * 100, 1),
        },
        "protein": {
            "count": int(master["protein_ok"].sum()),
            "rate": round(master["protein_ok"].mean() * 100, 1),
        },
        "protein_ratio": {
            "count": int(master["protein_ratio_ok"].sum()),
            "rate": round(master["protein_ratio_ok"].mean() * 100, 1),
        },
        "carb_ratio": {
            "count": int(master["carb_ratio_ok"].sum()),
            "rate": round(master["carb_ratio_ok"].mean() * 100, 1),
        },
        "fat_ratio": {
            "count": int(master["fat_ratio_ok"].sum()),
            "rate": round(master["fat_ratio_ok"].mean() * 100, 1),
        },
        "all": {
            "count": int(master["nutri_compliance_ok"].sum()),
            "rate": round(master["nutri_compliance_ok"].mean() * 100, 1),
        },
    }

    failed_meals = master[master["nutri_compliance_ok"] == False].copy()
    failed_list = []
    for _, row in failed_meals.head(5).iterrows():
        issues = []
        if not row["energy_ok"]:
            issues.append(f"에너지 {row['Kcal']:.0f}kcal")
        if not row["protein_ok"]:
            issues.append(f"단백질 {row['Prot']:.1f}g")
        if not row["protein_ratio_ok"]:
            issues.append("단백질비율 초과")
        if not row["carb_ratio_ok"]:
            issues.append(f"탄수화물비율 {row['carb_energy_ratio']*100:.0f}%")
        if not row["fat_ratio_ok"]:
            issues.append(f"지방비율 {row['fat_energy_ratio']*100:.0f}%")
        failed_list.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "mealType": row["meal_type"],
                "issues": issues,
            }
        )

    master["weekday"] = master["Date"].dt.dayofweek
    weekday_names = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_stats = (
        master.groupby("weekday")
        .agg(total=("nutri_compliance_ok", "count"), ok=("nutri_compliance_ok", "sum"))
        .reset_index()
    )
    weekday_compliance = {}
    for _, row in weekday_stats.iterrows():
        day_name = weekday_names[int(row["weekday"])]
        weekday_compliance[day_name] = {
            "rate": round(row["ok"] / row["total"] * 100, 1),
            "ok": int(row["ok"]),
            "total": int(row["total"]),
        }

    return NutritionCompliance(
        targetGroup=target_group,
        totalMeals=total_meals,
        energyStandard=E0,
        proteinStandard=P0,
        compliance=compliance,
        failedMealsSample=failed_list,
        failedMealsTotal=len(failed_meals),
        weekdayCompliance=weekday_compliance,
    )


def get_grade(score: float) -> Tuple[str, str]:
    """Convert score to grade."""
    if score >= 90:
        return "A+", "🏆"
    if score >= 85:
        return "A", "🥇"
    if score >= 80:
        return "B+", "🥈"
    if score >= 75:
        return "B", "🥉"
    if score >= 70:
        return "C+", "📊"
    if score >= 65:
        return "C", "📈"
    return "D", "📉"


def analyze_quality_scorecard(
    master: pd.DataFrame,
    meal_plan_df: pd.DataFrame,
    has_nutrition: bool = True,
) -> QualityScorecard:
    """Analyze quality scorecard."""
    if has_nutrition and "nutri_compliance_ok" in master.columns:
        nutri_rate = master["nutri_compliance_ok"].mean() * 100
        nutri_score = min(nutri_rate, 100)
    else:
        nutri_rate = 0
        nutri_score = 50

    if "avg_rating" in master.columns and master["avg_rating"].notna().sum() > 0:
        avg_rating = master["avg_rating"].mean()
        satisfaction_score = (avg_rating / 5) * 100
    else:
        avg_rating = 3.5
        satisfaction_score = 70

    if "leftover_rate" in master.columns and master["leftover_rate"].notna().sum() > 0:
        avg_leftover = master["leftover_rate"].mean() * 100
        leftover_score = max(0, 100 - (avg_leftover * 5))
    else:
        avg_leftover = 10
        leftover_score = 50

    if "menus" in meal_plan_df.columns:
        all_menus = []
        for menus in meal_plan_df["menus"]:
            if isinstance(menus, list):
                all_menus.extend(menus)
        unique_menus = len(set(all_menus))
        total_menus = len(all_menus)
        diversity_score = (unique_menus / total_menus * 100) if total_menus > 0 else 50
    else:
        diversity_score = 50
        unique_menus = 0

    if has_nutrition:
        weights = {
            "nutrition": 0.25,
            "satisfaction": 0.30,
            "leftover": 0.30,
            "diversity": 0.15,
        }
    else:
        weights = {
            "nutrition": 0.0,
            "satisfaction": 0.40,
            "leftover": 0.40,
            "diversity": 0.20,
        }

    total_score = (
        nutri_score * weights["nutrition"]
        + satisfaction_score * weights["satisfaction"]
        + leftover_score * weights["leftover"]
        + diversity_score * weights["diversity"]
    )

    grade, _ = get_grade(total_score)

    scores = {
        "nutrition": {
            "score": round(nutri_score, 1),
            "grade": get_grade(nutri_score)[0],
            "detail": f"{nutri_rate:.0f}% 충족" if has_nutrition else "데이터 없음",
        },
        "satisfaction": {
            "score": round(satisfaction_score, 1),
            "grade": get_grade(satisfaction_score)[0],
            "detail": f"평균 {avg_rating:.2f}점",
        },
        "leftover": {
            "score": round(leftover_score, 1),
            "grade": get_grade(leftover_score)[0],
            "detail": f"평균 {avg_leftover:.1f}%",
        },
        "diversity": {
            "score": round(diversity_score, 1),
            "grade": get_grade(diversity_score)[0],
            "detail": f"{unique_menus}종",
        },
    }

    good_points = []
    improve_points = []

    if has_nutrition:
        if nutri_score >= 85:
            good_points.append(f"영양기준 충족률 우수 ({nutri_rate:.0f}%)")
        elif nutri_score < 80:
            improve_points.append(f"영양기준 충족률 향상 필요 (현재 {nutri_rate:.0f}%)")

    if satisfaction_score >= 75:
        good_points.append(f"학생 만족도 양호 (평점 {avg_rating:.1f})")
    elif satisfaction_score < 70:
        improve_points.append(f"학생 만족도 개선 필요 (현재 {avg_rating:.1f}점)")

    if leftover_score >= 80:
        good_points.append(f"잔반율 관리 우수 ({avg_leftover:.1f}%)")
    elif leftover_score < 70:
        improve_points.append(f"잔반율 감소 노력 필요 (현재 {avg_leftover:.1f}%)")

    if diversity_score >= 70:
        good_points.append("메뉴 다양성 확보")
    elif diversity_score < 60:
        improve_points.append("메뉴 다양성 확대 필요")

    return QualityScorecard(
        scores=scores,
        total={"score": round(total_score, 1), "grade": grade},
        goodPoints=good_points,
        improvePoints=improve_points,
        nextMonthTarget=round(min(total_score + 5, 100), 1),
    )


def train_risk_model(master: pd.DataFrame):
    """Train risk model for leftovers."""
    from sklearn.ensemble import RandomForestClassifier

    threshold = master["leftover_rate"].quantile(0.7)
    master["high_risk"] = (master["leftover_rate"] >= threshold).astype(int)

    feature_cols = ["Kcal", "Prot", "Carb", "Fat", "avg_rating", "avg_sentiment"]
    available_cols = [
        c for c in feature_cols if c in master.columns and master[c].notna().sum() > 0
    ]

    if len(available_cols) < 2:
        return None

    X = master[available_cols].fillna(master[available_cols].median())
    y = master["high_risk"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    return model, available_cols


def predict_risk(master: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    """Predict leftover risk."""
    X = master[feature_cols].fillna(master[feature_cols].median())
    master["risk_prob"] = model.predict_proba(X)[:, 1]
    master["risk_label"] = (master["risk_prob"] >= 0.5).map(
        {True: "고위험", False: "저위험"}
    )
    return master


def analyze_risk_forecast(master: pd.DataFrame) -> RiskForecast:
    """Analyze leftover risk forecast."""
    result = train_risk_model(master.copy())
    if result is None:
        return RiskForecast(
            avgRisk=0.0, highRiskRatio=0.0, highRiskCount=0, highRiskMeals=[]
        )

    model, feature_cols = result
    master = predict_risk(master, model, feature_cols)

    avg_risk = master["risk_prob"].mean()
    high_risk_count = (master["risk_label"] == "고위험").sum()
    high_risk_ratio = high_risk_count / len(master)

    high_risk_meals = master[master["risk_label"] == "고위험"].nlargest(5, "risk_prob")
    high_risk_list = []
    for _, row in high_risk_meals.iterrows():
        high_risk_list.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "mealType": row["meal_type"],
                "riskProb": round(row["risk_prob"], 3),
                "leftoverRate": round(row["leftover_rate"] * 100, 1),
            }
        )

    return RiskForecast(
        avgRisk=round(avg_risk, 3),
        highRiskRatio=round(high_risk_ratio, 3),
        highRiskCount=int(high_risk_count),
        highRiskMeals=high_risk_list,
    )


def generate_menu_strategies(
    nutrition_compliance: NutritionCompliance,
    quality_scorecard: QualityScorecard,
    risk_forecast: RiskForecast,
    llm,
    trend_analysis: Optional[object] = None,
) -> List[MenuStrategyItem]:
    """Generate menu strategies."""
    strategies = []

    if nutrition_compliance is not None:
        compliance = nutrition_compliance.compliance
        for key, data in compliance.items():
            if key != "all" and data["rate"] < 80:
                issue_name = {
                    "energy": "에너지",
                    "protein": "단백질",
                    "protein_ratio": "단백질 에너지비",
                    "carb_ratio": "탄수화물 에너지비",
                    "fat_ratio": "지방 에너지비",
                }.get(key, key)

                strategies.append(
                    MenuStrategyItem(
                        strategyType="category_improvement",
                        targetCategory=f"영양기준({issue_name})",
                        negativeRatio=100 - data["rate"],
                        priority="high" if data["rate"] < 70 else "medium",
                        topIssues=[
                            f"{issue_name} 기준 미충족 {100 - data['rate']:.1f}%"
                        ],
                        description=f"{issue_name} 기준 충족률이 {data['rate']}%로 낮습니다. 개선이 필요합니다.",
                    )
                )

    for point in quality_scorecard.improvePoints:
        plan = _generate_plan(
            llm, point, nutrition_compliance, quality_scorecard, risk_forecast
        )
        print(point)
        print(plan)
        trigger: str = plan.get("trigger", "")
        adjustment: str = plan.get("adjustment", "")
        howToApply: list[str] = plan.get("howToApply", [])
        strategies.append(
            MenuStrategyItem(
                strategyType="policy_update",
                trigger=trigger,
                adjustment=adjustment,
                howToApply=howToApply,
            )
        )

    if risk_forecast.highRiskCount > 0:
        strategies.append(
            MenuStrategyItem(
                strategyType="risk_alert",
                description=(
                    f"고위험 식단 {risk_forecast.highRiskCount}건 예측됨 "
                    f"(전체의 {risk_forecast.highRiskRatio*100:.1f}%)"
                ),
                topIssues=[f"평균 위험도: {risk_forecast.avgRisk:.1%}"],
            )
        )

    if trend_analysis:
        if isinstance(trend_analysis, TrendAnalysisResult):
            for cat in trend_analysis.problem_categories[:3]:
                strategies.append(
                    MenuStrategyItem(
                        strategyType="category_improvement",
                        targetCategory=cat.category,
                        negativeRatio=cat.neg_ratio,
                        priority="high",
                        topIssues=[],
                    )
                )
            for cat in trend_analysis.preferred_categories[:3]:
                strategies.append(
                    MenuStrategyItem(
                        strategyType="recommend",
                        targetCategory=cat.category,
                        preferenceScore=cat.pos_ratio,
                    )
                )
        elif isinstance(trend_analysis, dict):
            problem_cats = trend_analysis.get("problemCategories", [])
            for cat in problem_cats[:3]:
                strategies.append(
                    MenuStrategyItem(
                        strategyType="category_improvement",
                        targetCategory=cat.get("category", ""),
                        negativeRatio=cat.get("neg_ratio", 0),
                        priority="high",
                        topIssues=[],
                    )
                )

            pref_map = trend_analysis.get("categoryPreference", {})
            top_prefs = sorted(pref_map.items(), key=lambda x: -x[1])[:3]
            for cat, score in top_prefs:
                strategies.append(
                    MenuStrategyItem(
                        strategyType="recommend",
                        targetCategory=cat,
                        preferenceScore=score,
                    )
                )

    return strategies


def _normalize_meal_plan_df(df_meal_plan: pd.DataFrame) -> pd.DataFrame:
    df = df_meal_plan.copy()
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    if "meal_type" not in df.columns and "mealType" in df.columns:
        df["meal_type"] = df["mealType"]

    col_map = {}
    for base, options in {
        "rice": ["rice", "Rice"],
        "soup": ["soup", "Soup"],
        "main1": ["main1", "Main1"],
        "main2": ["main2", "Main2"],
        "side": ["side", "Side"],
        "kimchi": ["kimchi", "Kimchi"],
        "dessert": ["dessert", "Dessert"],
    }.items():
        for option in options:
            if option in df.columns:
                col_map[option] = base
                break
    if col_map:
        df = df.rename(columns=col_map)

    menu_cols = ["rice", "soup", "main1", "main2", "side", "kimchi", "dessert"]
    for col in menu_cols:
        if col not in df.columns:
            df[col] = None

    df["menus"] = df[menu_cols].apply(
        lambda row: [str(v).strip() for v in row if pd.notna(v) and str(v).strip()],
        axis=1,
    )
    df["menu_key"] = df["menus"].apply(lambda x: " | ".join(x))

    if "Kcal" not in df.columns:
        df["Kcal"] = pd.to_numeric(df.get("kcal"), errors="coerce")
    if "Prot" not in df.columns:
        df["Prot"] = pd.to_numeric(df.get("protein"), errors="coerce")
    if "Carb" not in df.columns:
        df["Carb"] = pd.to_numeric(df.get("carb"), errors="coerce")
    if "Fat" not in df.columns:
        df["Fat"] = pd.to_numeric(df.get("fat"), errors="coerce")

    return df


def _normalize_daily_info_df(df_daily_info: pd.DataFrame) -> pd.DataFrame:
    df = df_daily_info.copy()
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    if "meal_type" not in df.columns and "mealType" in df.columns:
        df["meal_type"] = df["mealType"]
    if "served_count" not in df.columns:
        df["served_count"] = df.get("servedProxy")
    if "leftover_kg" not in df.columns:
        df["leftover_kg"] = df.get("leftoverKg")
    df["leftover_rate"] = df["leftover_kg"] / df["served_count"].replace(0, np.nan)
    return df


def _normalize_daily_report_df(df_daily_report: pd.DataFrame) -> pd.DataFrame:
    df = df_daily_report.copy()
    if "Date" not in df.columns and "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    if "meal_type" not in df.columns and "mealType" in df.columns:
        df["meal_type"] = df["mealType"]

    def _pick_col(candidates, default):
        for col in candidates:
            if col in df.columns:
                return df[col]
        return default

    df["avg_rating"] = _pick_col(["avg_rating", "kpis.avg_rating_5", "avg_rating_5"], 0)
    df["avg_sentiment"] = _pick_col(["avg_sentiment", "kpis.avg_review_sentiment"], 0)
    df["review_count"] = _pick_col(["review_count", "kpis.review_count"], 0)
    df["top_negative_aspects"] = _pick_col(
        ["top_negative_aspects"], [[] for _ in range(len(df))]
    )
    df["top_issues"] = _pick_col(["top_issues"], [[] for _ in range(len(df))])

    return df


# ============================================================
# 통합 분석 함수 (팀 통일 스키마)
# ============================================================


def run_menu_strategy_analysis(request: dict) -> MenuStrategyResponse:
    """모델5 통합 분석 실행"""

    # 요청 정보 추출
    user_name = request.get("userName", "영양사")
    year = request.get("year", 2026)
    month = request.get("month", 2)
    target_month = f"{year}-{month:02d}"
    target_group = request.get("targetGroup", "")

    print(f"📊 분석 시작: {user_name}님의 {target_month} 데이터")

    # print(request)

    # 1. 데이터 변환
    meal_plan_df = convert_meal_plan_to_df(request["mealPlan"])
    daily_info_df = convert_daily_info_to_df(request["dailyInfo"])

    daily_analysis_df = None
    if request.get("dailyAnalysis"):
        daily_analysis_df = convert_daily_analysis_to_df(request["dailyAnalysis"])

    review_analysis_df = None
    if request.get("reviewAnalysis"):
        review_analysis_df = convert_review_analysis_to_df(request["reviewAnalysis"])

    print(f"  - 식단표: {len(meal_plan_df)}건")
    print(f"  - 잔반정보: {len(daily_info_df)}건")

    # 2. 마스터 테이블 병합
    master = merge_master_table(meal_plan_df, daily_info_df, daily_analysis_df)
    print(f"  - 마스터 테이블: {len(master)}건")

    # 3. 영양정보 확인
    has_nutrition = bool(master["Kcal"].notna().sum() > 0)
    print(f"  - 영양정보: {'있음' if has_nutrition else '없음'}")

    # 4. 영양기준 분석 (영양정보 있는 경우만)
    nutrition_compliance = None
    if has_nutrition:
        nutrition_compliance = analyze_nutrition_compliance(master.copy(), target_group)
        print(f"  - 영양기준 분석 완료")

    # 5. 품질 성적표 분석
    quality_scorecard = analyze_quality_scorecard(master, meal_plan_df, has_nutrition)
    print(f"  - 품질 성적표 완료")

    # 6. 위험도 예측
    risk_forecast = analyze_risk_forecast(master)
    print(f"  - 위험도 예측 완료")

    # 7. 메뉴 전략 생성
    menu_strategies = generate_menu_strategies(
        nutrition_compliance=nutrition_compliance,
        quality_scorecard=quality_scorecard,
        risk_forecast=risk_forecast,
        llm=request.get("llm", None),
        trend_analysis=request.get("trendAnalysis"),
    )
    print(f"  - 전략 {len(menu_strategies)}개 생성")

    # 8. 응답 생성
    return MenuStrategyResponse(
        nutritionCompliance=nutrition_compliance,
        qualityScorecard=quality_scorecard,
        riskForecast=risk_forecast,
        menuStrategies=menu_strategies,
    )
