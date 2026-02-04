from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

import pandas as pd

from app.models.operation import *
from app.models.analysis_sentiment import *
from app.models.analysis_periodic import *


def compute_periodic_rating_kpi(df_daily_reports: pd.DataFrame) -> RatingKPI | dict:
    """
    들어온 운영 기록 '분석' 데이터프레임 상에서 통계적 수치들을 뽑아낸다
    - df_daily_reports: 운영 기록 '분석' 데이터프레임
    """
    # 빈 데이터 처리
    if df_daily_reports.empty or "kpis.review_count" not in df_daily_reports.columns:
        return {
            "avg_rating": None,
            "min_rating_day": {"date": None, "value": None},
            "max_rating_day": {"date": None, "value": None},
            "rating_trend": [],
        }

    df = df_daily_reports.copy()

    # 가중 평균
    weighted_avg = (
        (df["kpis.avg_rating_5"] * df["kpis.review_count"]).sum()
        / df["kpis.review_count"].sum()
        if df["kpis.review_count"].sum() > 0
        else None
    )

    min_row = df.loc[df["kpis.avg_rating_5"].idxmin()]
    max_row = df.loc[df["kpis.avg_rating_5"].idxmax()]

    trend = df.sort_values("date")[["date", "kpis.avg_rating_5"]].to_dict(
        orient="records"
    )

    return {
        "avg_rating": float(round(weighted_avg, 4)) if weighted_avg else None,
        "min_rating_day": {
            "date": min_row["date"],
            "value": float(min_row["kpis.avg_rating_5"]),
        },
        "max_rating_day": {
            "date": max_row["date"],
            "value": float(max_row["kpis.avg_rating_5"]),
        },
        "rating_trend": trend,
    }


def compute_periodic_leftover_kpi_from_operational(
    df_daily_info: pd.DataFrame,
) -> LeftoverKPI | dict:
    """
    운영 기록 데이터프레임에서 잔반률 관련 통계를 뽑아낸다
    - df_leftover: '영 기록 데이터프레임 (운영 기록 '분석' 데이터프레임과 혼동하지 말 것)
    """
    # 빈 데이터 처리
    if df_daily_info.empty:
        return {
            "avg_leftover_rate": 0.0,
            "avg_missed_rate": 0.0,
            "worst_cases": [],
        }

    df = df_daily_info.copy()

    df["leftover_rate"] = df["leftoverKg"] / df["servedProxy"].replace(0, 1)
    df["missed_rate"] = df["missedProxy"].fillna(0) / df["servedProxy"].replace(0, 1)

    return {
        "avg_leftover_rate": float(df["leftover_rate"].mean()),
        "avg_missed_rate": float(df["missed_rate"].mean()),
        "worst_cases": (
            df.sort_values("leftover_rate", ascending=False)
            .head(3)
            .to_dict(orient="records")
        ),
    }


def aggregate_reviews_periodic(
    df_sentiments_reviews: pd.DataFrame,
) -> ReviewAggregate | dict:
    """
    리뷰 감정분석 결과를 집계한다.
    - df_sentiments_reviews: 리뷰 '감정분석' 데이터프레임
    """
    # 빈 데이터 처리
    if df_sentiments_reviews.empty:
        return {
            "count": 0,
            "sentiment_distribution": {},
            "mealType_distribution": {},
            "rating_distribution": {},
        }

    rating_dist = (
        df_sentiments_reviews["rating_5"].value_counts().sort_index().to_dict()
    )

    return {
        "count": int(len(df_sentiments_reviews)),
        "sentiment_distribution": (
            df_sentiments_reviews["sentiment_label"].value_counts().to_dict()
        ),
        "mealType_distribution": (
            df_sentiments_reviews["meal_type"].value_counts().to_dict()
        ),
        "rating_distribution": rating_dist,
    }


def aggregate_posts_periodic(df_sentiments_posts: pd.DataFrame) -> PostAggregate | dict:
    """
    게시물 감정분석 결과를 집계한다.
    - df_sentiments_posts: 게시물 '감정분석' 데이터프레임
    """
    # 빈 데이터 처리
    if df_sentiments_posts.empty:
        return {
            "count": 0,
            "category_distribution": {},
            "sentiment_distribution": {},
            "category_sentiment_matrix": {},
            "issue_flags": {},
        }

    category_dist = df_sentiments_posts["category"].value_counts().to_dict()
    sentiment_dist = df_sentiments_posts["sentiment_label"].value_counts().to_dict()

    matrix = df_sentiments_posts.pivot_table(
        index="category", columns="sentiment_label", aggfunc="size", fill_value=0
    ).to_dict()

    # issue_flags explode
    if "issue_flags" in df_sentiments_posts.columns:
        issues = (
            df_sentiments_posts["issue_flags"]
            .explode()
            .dropna()
            .value_counts()
            .to_dict()
        )
    else:
        issues = {}

    return {
        "count": int(len(df_sentiments_posts)),
        "category_distribution": category_dist,
        "sentiment_distribution": sentiment_dist,
        "category_sentiment_matrix": matrix,
        "issue_flags": issues,
    }


def aggregate_aspects_from_reviews(df_sentiments_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    리뷰 감정분석 결과에 대해 긍정/중립/부정 평가들을 집계한다
    - df_sentiments_reviews: 리뷰 감정분석 데이터프레임
    """
    # 빈 데이터 처리
    if df_sentiments_reviews.empty:
        return pd.DataFrame(
            columns=["tag", "POSITIVE", "NEUTRAL", "NEGATIVE", "total", "neg_rate"]
        )

    rows = []

    for _, row in df_sentiments_reviews.iterrows():
        aspects = row.get("aspect_details") or {}
        if not isinstance(aspects, dict):
            continue
        for tag, info in aspects.items():
            if isinstance(info, dict):
                rows.append({"tag": tag, "polarity": info.get("polarity", "NEUTRAL")})
            else:
                rows.append({"tag": tag, "polarity": "NEUTRAL"})

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=["tag", "POSITIVE", "NEUTRAL", "NEGATIVE", "total", "neg_rate"]
        )

    pivot = df.pivot_table(
        index="tag", columns="polarity", aggfunc="size", fill_value=0
    ).reset_index()

    for col in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        if col not in pivot:
            pivot[col] = 0

    pivot["total"] = pivot["POSITIVE"] + pivot["NEUTRAL"] + pivot["NEGATIVE"]
    pivot["neg_rate"] = pivot["NEGATIVE"] / pivot["total"]

    return pivot.sort_values("total", ascending=False)


def select_problem_areas(
    df_aspects_reviews: pd.DataFrame, top_k: int = 5, min_mentions: int = 5
) -> ProblemArea | dict:
    """
    제공된 리뷰들 내에서의 부정적 언급 수에 기반하여 문제가 발생하는 영역을 걸러낸다.
    - df_aspects_reviews: 리뷰 감정분석 데이터프레임을 aggregate_aspects_from_reviews()에 통과시킨 결과물
    - top_k: 선정 갯수 (기본값 5)
    - min_mention: 후보로 걸리기 위한 '부정적인 언급 수' 하한선 (기본값 5)
    """
    df = df_aspects_reviews.copy()
    df = df[df["total"] >= min_mentions]

    if df.empty:
        return []

    df["problem_score"] = df["total"] * df["neg_rate"]

    top = df.sort_values("problem_score", ascending=False).head(top_k)

    return [
        {
            "tag": r["tag"],
            "total_mentions": int(r["total"]),
            "neg_rate": float(round(r["neg_rate"], 4)),
            "problem_score": float(round(r["problem_score"], 4)),
        }
        for _, r in top.iterrows()
    ]


def select_deepdive_targets(
    df_daily_reports: pd.DataFrame, metric: str = "rating", top_k: int = 3
) -> List[Dict]:
    """
    심층 분석을 수행할 일자 선정
    - df_daily_reports: 일일 분석 데이터프레임
    - metric: 기준이 되는 수치 (기본값은 rating)
    - top_k: 선정 갯수 (rating일 경우에는 최하위, 그렇지 않은 경우는 최상위) (기본값 3)
    """
    df = df_daily_reports.copy()

    if metric == "rating":
        target = df.nsmallest(top_k, "kpis.avg_rating_5")
        key = "kpis.avg_rating_5"
    else:
        target = df.nlargest(top_k, metric)
        key = metric

    return [{"date": r["date"], "value": float(r[key])} for _, r in target.iterrows()]


def select_deepdive_targets_from_leftover(
    df_daily_info: pd.DataFrame, top_k: int = 3
) -> List[Dict]:
    # 빈 데이터 처리
    if df_daily_info.empty:
        return []

    df = df_daily_info.copy()
    df["leftover_rate"] = df["leftoverKg"] / df["servedProxy"].replace(0, 1)

    targets = df.sort_values("leftover_rate", ascending=False).head(top_k)

    return [
        {
            "date": r["date"],
            "mealType": r["mealType"],
            "leftover_rate": float(round(r["leftover_rate"], 4)),
        }
        for _, r in targets.iterrows()
    ]


def build_deepdive_analysis(
    df_sentiments_reviews: pd.DataFrame,
    df_meal_plans: pd.DataFrame,
    target_date: str,
    meal_type: str,
) -> DeepDiveResult | dict:
    """
    실제 심층분석 진행 (특정 일자에 대해 리뷰, 식단 데이터를 사용해서 추가 분석)
    - df_sentiments_reviews: 리뷰 감정분석 데이터프레임
    - df_meal_plans: 식단표 데이터프레임
    - target_date: 대상 일자 (YYYY-MM-DD)
    - meal_type: 식사 유형 (중식/석식)
    """
    # 빈 데이터 처리
    if df_sentiments_reviews.empty and df_meal_plans.empty:
        return {
            "date": target_date,
            "mealType": meal_type,
            "menus": {},
            "aspect_summary": [],
            "evidence_phrases": [],
        }

    target_date_ = pd.to_datetime(target_date)

    # 복사 후 날짜 포맷 설정 (감정분석)
    if not df_sentiments_reviews.empty:
        sentiments_reviews_df = df_sentiments_reviews.copy()
        sentiments_reviews_df["date"] = pd.to_datetime(sentiments_reviews_df["date"])
    else:
        sentiments_reviews_df = pd.DataFrame()

    # 복사 후 날짜 포맷 설정 (식단표)
    if not df_meal_plans.empty:
        meal_plans_df = df_meal_plans.copy()
        meal_plans_df["date"] = pd.to_datetime(meal_plans_df["date"])
    else:
        meal_plans_df = pd.DataFrame()

    # 리뷰 감정분석 데이터 필터링
    if not sentiments_reviews_df.empty:
        subset = sentiments_reviews_df[
            (sentiments_reviews_df["date"] == target_date_)
            & (sentiments_reviews_df["meal_type"] == meal_type)
        ]
    else:
        subset = pd.DataFrame()

    aspect_df = aggregate_aspects_from_reviews(subset)

    # 증거 문구 추출
    if not subset.empty and "evidence_phrases" in subset.columns:
        evidence = (
            subset["evidence_phrases"]
            .explode()
            .dropna()
            .value_counts()
            .head(5)
            .index.tolist()
        )
    else:
        evidence = []

    menus = {
        "rice": "(RICE_UNKNOWN)",
        "soup": "(SOUP_UNKNOWN)",
        "main1": "(MAIN1_UNKNOWN)",
        "main2": "(MAIN2_UNKNOWN)",
        "side": "(SIDE_UNKNOWN)",
        "kimchi": "(KIMCHI_UNKNOWN)",
        "dessert": "(DESSERT_UNKNOWN)",
    }

    if not meal_plans_df.empty:
        menu_row = meal_plans_df[
            (meal_plans_df["date"].dt.date == target_date_.date())
            & (meal_plans_df["mealType"] == meal_type)
        ]
        if len(menu_row) > 0:
            menus = {
                "rice": menu_row.iloc[0].get("rice", "(RICE_UNKNOWN)"),
                "soup": menu_row.iloc[0].get("soup", "(SOUP_UNKNOWN)"),
                "main1": menu_row.iloc[0].get("main1", "(MAIN1_UNKNOWN)"),
                "main2": menu_row.iloc[0].get("main2", "(MAIN2_UNKNOWN)"),
                "side": menu_row.iloc[0].get("side", "(SIDE_UNKNOWN)"),
                "kimchi": menu_row.iloc[0].get("kimchi", "(KIMCHI_UNKNOWN)"),
                "dessert": menu_row.iloc[0].get("dessert", "(DESSERT_UNKNOWN)"),
            }

    return {
        "date": str(target_date_.date()),
        "mealType": meal_type,
        "menus": menus,
        "aspect_summary": aspect_df.to_dict(orient="records"),
        "evidence_phrases": evidence,
    }


def run_periodic_analysis(
    df_meal_plans: pd.DataFrame,
    df_daily_info: pd.DataFrame,
    df_sentiments_reviews: pd.DataFrame,
    df_sentiments_posts: pd.DataFrame,
    df_daily_report: pd.DataFrame,
    df_reviews: pd.DataFrame,
) -> PeriodicAnalysisResult:

    rating_kpi = compute_periodic_rating_kpi(df_daily_report)
    leftover_kpi = compute_periodic_leftover_kpi_from_operational(df_daily_info)

    review_agg = aggregate_reviews_periodic(df_sentiments_reviews)
    post_agg = aggregate_posts_periodic(df_sentiments_posts)

    aspect_df = aggregate_aspects_from_reviews(df_sentiments_reviews)

    problem_areas = select_problem_areas(aspect_df)

    deep_targets = select_deepdive_targets_from_leftover(df_daily_info)
    deepdives = []
    for t in deep_targets:
        deepdives.append(
            build_deepdive_analysis(
                df_sentiments_reviews,
                df_meal_plans,
                target_date=t["date"],
                meal_type=t["mealType"],
            )
        )
    deepdives

    return PeriodicAnalysisResult(
        kpis={
            "rating": RatingKPI(**rating_kpi),
            "leftover": LeftoverKPI(**leftover_kpi),
        },
        reviews=ReviewAggregate(**review_agg),
        posts=PostAggregate(**post_agg),
        problem_areas=[ProblemArea(**p) for p in problem_areas],
        deepdives=[DeepDiveResult(**d) for d in deepdives],
    )
