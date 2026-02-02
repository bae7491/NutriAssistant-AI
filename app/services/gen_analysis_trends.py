from typing import Dict, List, Optional, Tuple

import pandas as pd

from collections import Counter
import re

from app.models.analysis_trends import (
    CategorySentiment,
    PreferenceChange,
    ComplaintTagChange,
    ProblemCategory,
    PreferredCategory,
    CategoryComplaint,
    TrendAnalysisResult,
)


# ============================================================
# 메뉴명 정규화 함수
# ============================================================
import re


def normalize_menu_name(name: str) -> str:
    """메뉴명 정규화 (괄호 제거, 동의어 통일)"""
    if pd.isna(name) or not isinstance(name, str):
        return ""

    # 괄호 내용 제거
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\[[^\]]*\]", "", name)

    # 공백 정리
    name = " ".join(name.split())

    # 동의어 통일
    synonyms = {
        "흰밥": "쌀밥",
        "백미밥": "쌀밥",
        "공기밥": "쌀밥",
        "된장찌게": "된장찌개",
        "김치찌게": "김치찌개",
    }
    for old, new in synonyms.items():
        name = name.replace(old, new)

    return name.strip()


def norm_light(s: str) -> str:
    """공백 제거 (DB 매칭용)"""
    if pd.isna(s):
        return ""
    return str(s).replace(" ", "")


# ============================================================
# 메뉴 → 카테고리 매핑 함수
# ============================================================


def map_menu_to_category(
    df_reviews: pd.DataFrame,
    df_meal_plans: pd.DataFrame,
    df_food_db: pd.DataFrame,
    menu_col: str = "menu_key",
    category_col: str = "식품대분류명",
) -> pd.DataFrame:
    """
    리뷰 데이터에 메뉴 카테고리를 매핑

    Args:
        df_reviews: 리뷰 데이터 (Date, meal_type 포함)
        df_meal_plans: 식단표 (Date, meal_type, menu_key 포함)
        df_food_db: 음식 DB (음식명, 식품대분류명 포함)

    Returns:
        trend_df: 카테고리가 매핑된 분석용 DataFrame
    """

    # 1) 리뷰 + 식단표 조인 (Date + meal_type)
    df = df_reviews.merge(
        df_meal_plans[["Date", "meal_type", "menu_key"]],
        on=["Date", "meal_type"],
        how="left",
    )

    df["menu_key"] = df["menu_key"].fillna("")

    # 2) menu_key 분리 ("|" 구분)
    df_splitted = df.assign(menu_name_raw=df["menu_key"].str.split(r" \| "))
    df_exploded = df_splitted.explode("menu_name_raw")
    df_exploded["menu_name_raw"] = df_exploded["menu_name_raw"].fillna("")

    # 3) 메뉴명 정규화
    df_exploded["menu_name_norm"] = df_exploded["menu_name_raw"].apply(
        normalize_menu_name
    )
    df_exploded["menu_name_key"] = df_exploded["menu_name_norm"].apply(norm_light)

    # 4) 음식 DB 준비
    food_db = df_food_db.copy()
    # food_db["db_key"] = food_db["음식명"].apply(norm_light)
    food_db["db_key"] = food_db["식품명"].apply(norm_light)

    # 5) 매핑
    df_merged = df_exploded.merge(
        food_db[["db_key", category_col]],
        left_on="menu_name_key",
        right_on="db_key",
        how="left",
    )
    df_merged.rename(columns={category_col: "menu_category"}, inplace=True)

    # 6) 매핑 안 된 메뉴 fallback
    fallback_map = {
        "쌀밥": "밥류",
        "잡곡밥": "밥류",
        "흰밥": "밥류",
        "배추김치": "김치류",
        "깍두기": "김치류",
        "총각김치": "김치류",
        "된장찌개": "국류",
        "김치찌개": "국류",
        "미역국": "국류",
    }

    def apply_fallback(row):
        if pd.isna(row["menu_category"]) or row["menu_category"] == "":
            for keyword, category in fallback_map.items():
                if keyword in str(row["menu_name_norm"]):
                    return category
        return row["menu_category"]

    df_merged["menu_category"] = df_merged.apply(apply_fallback, axis=1)

    return df_merged


# ============================================================
# 카테고리별 감정 분포 분석
# ============================================================


def compute_category_sentiment(
    trend_df: pd.DataFrame, sentiment_col: str = "sentiment_label"
) -> List[CategorySentiment]:
    """
    카테고리별 긍정/중립/부정 비율 계산

    Args:
        trend_df: 카테고리 매핑된 DataFrame
        sentiment_col: 감정 라벨 컬럼명

    Returns:
        List[CategorySentiment]
    """

    # NaN 카테고리 제외
    df = trend_df.dropna(subset=["menu_category"])

    # 카테고리별 감정 비율
    cat_sentiment = (
        pd.crosstab(df["menu_category"], df[sentiment_col], normalize="index") * 100
    )

    # 모든 감정 라벨 보장
    for label in ["pos", "neu", "neg"]:
        if label not in cat_sentiment.columns:
            cat_sentiment[label] = 0.0

    # 카테고리별 건수
    cat_count = df.groupby("menu_category").size()

    result = []
    for cat in cat_sentiment.index:
        result.append(
            CategorySentiment(
                category=cat,
                pos_ratio=round(cat_sentiment.loc[cat, "pos"], 1),
                neu_ratio=round(cat_sentiment.loc[cat, "neu"], 1),
                neg_ratio=round(cat_sentiment.loc[cat, "neg"], 1),
                count=int(cat_count.get(cat, 0)),
            )
        )

    return result


# ============================================================
# 주차별 부정 비율 분석
# ============================================================


def compute_weekly_neg_trend(
    trend_df: pd.DataFrame, sentiment_col: str = "sentiment_label"
) -> Dict[int, float]:
    """
    주차별 부정 비율 추이 계산

    Args:
        trend_df: 분석용 DataFrame (Date, week_in_month 포함)

    Returns:
        Dict[int, float]: {주차: 부정비율}
    """

    # 주차 정보 추가 (없으면 생성)
    if "week_in_month" not in trend_df.columns:
        trend_df = trend_df.copy()
        trend_df["week_in_month"] = trend_df["Date"].apply(
            lambda d: (d.day - 1) // 7 + 1
        )

    # 주차별 감정 비율
    weekly_sentiment = (
        pd.crosstab(
            trend_df["week_in_month"], trend_df[sentiment_col], normalize="index"
        )
        * 100
    )

    if "neg" not in weekly_sentiment.columns:
        weekly_sentiment["neg"] = 0.0

    result = {int(week): round(pct, 1) for week, pct in weekly_sentiment["neg"].items()}

    return result


# ============================================================
# 카테고리 선호도 변화 분석 (주차별)
# ============================================================

# 안전장치 파라미터
MIN_N_PER_WEEK = 10  # 주차-카테고리 최소 표본
DELTA_PCT_MAX = 30.0  # ±30%p 초과는 급변 → 제외
DELTA_MIN_SHOW = 5.0  # 5%p 이상만 변화 후보


def compute_preference_changes(
    trend_df: pd.DataFrame, sentiment_col: str = "sentiment_label"
) -> List[PreferenceChange]:
    """
    카테고리별 선호도 변화 (첫 주차 vs 마지막 주차)

    Args:
        trend_df: 분석용 DataFrame

    Returns:
        List[PreferenceChange]
    """
    df = trend_df.dropna(subset=["menu_category", "week_in_month"])

    # 주차-카테고리별 긍정 비율
    weekly_cat_pos = (
        df.groupby(["week_in_month", "menu_category"])[sentiment_col]
        .apply(lambda s: (s == "pos").sum() / len(s) * 100 if len(s) > 0 else 0)
        .unstack(fill_value=0)
    )

    # 주차-카테고리별 표본 수
    weekly_cat_n = (
        df.groupby(["week_in_month", "menu_category"]).size().unstack(fill_value=0)
    )

    weeks = sorted(df["week_in_month"].dropna().unique())
    if len(weeks) < 2:
        return []

    first_week, last_week = int(weeks[0]), int(weeks[-1])

    changes = []
    for cat in weekly_cat_pos.columns:
        n_first = (
            weekly_cat_n.loc[first_week, cat] if first_week in weekly_cat_n.index else 0
        )
        n_last = (
            weekly_cat_n.loc[last_week, cat] if last_week in weekly_cat_n.index else 0
        )

        # 표본 부족 → 제외
        if n_first < MIN_N_PER_WEEK or n_last < MIN_N_PER_WEEK:
            continue

        pos_first = (
            weekly_cat_pos.loc[first_week, cat]
            if first_week in weekly_cat_pos.index
            else 0
        )
        pos_last = (
            weekly_cat_pos.loc[last_week, cat]
            if last_week in weekly_cat_pos.index
            else 0
        )
        delta = pos_last - pos_first

        # 급변 → 제외
        if abs(delta) > DELTA_PCT_MAX:
            continue

        # 최소 변화 기준
        if abs(delta) >= DELTA_MIN_SHOW:
            changes.append(
                PreferenceChange(
                    category=cat,
                    direction="increase" if delta > 0 else "decrease",
                    change_percent=round(delta, 1),
                )
            )

    # 변화량 기준 정렬
    changes.sort(key=lambda x: abs(x.change_percent), reverse=True)
    return changes[:5]


# ============================================================
# 불만 태그 변화 분석
# ============================================================

MIN_TAG_BASE = 10  # 첫주 최소 카운트
DELTA_RATIO_SHOW = 30.0  # 30% 이상 변화


def compute_complaint_tag_changes(
    trend_df: pd.DataFrame, aspect_col: str = "aspect_tags"
) -> List[ComplaintTagChange]:
    """
    불만 태그 변화 (첫 주차 vs 마지막 주차)

    Args:
        trend_df: 분석용 DataFrame (aspect_tags 리스트 포함)

    Returns:
        List[ComplaintTagChange]
    """
    df = trend_df.copy()

    # 주차 정보 확인
    if "week_in_month" not in df.columns:
        return []

    weeks = sorted(df["week_in_month"].dropna().unique())
    if len(weeks) < 2:
        return []

    first_week, last_week = int(weeks[0]), int(weeks[-1])

    # 주차별 태그 카운트
    tag_by_week = df.explode(aspect_col).dropna(subset=[aspect_col])
    tag_week_cnt = (
        tag_by_week.groupby(["week_in_month", aspect_col]).size().unstack(fill_value=0)
    )

    if first_week not in tag_week_cnt.index or last_week not in tag_week_cnt.index:
        return []

    changes = []
    for tag in tag_week_cnt.columns:
        cnt_first = tag_week_cnt.loc[first_week, tag]
        cnt_last = tag_week_cnt.loc[last_week, tag]

        # 첫주 최소 카운트 미달 → 제외
        if cnt_first < MIN_TAG_BASE:
            continue

        ratio_change = (cnt_last - cnt_first) / cnt_first * 100

        # 최소 변화율 미달 → 제외
        if abs(ratio_change) < DELTA_RATIO_SHOW:
            continue

        changes.append(
            ComplaintTagChange(
                tag=tag,
                direction="increase" if ratio_change > 0 else "decrease",
                change_percent=round(ratio_change, 1),
            )
        )

    changes.sort(key=lambda x: abs(x.change_percent), reverse=True)
    return changes[:5]


# ============================================================
# 문제/선호 카테고리 추출
# ============================================================


def compute_problem_preferred_categories(
    category_sentiments: List[CategorySentiment], top_k: int = 3
) -> tuple:
    """
    문제 카테고리(부정 상위)와 선호 카테고리(긍정 상위) 추출

    Args:
        category_sentiments: 카테고리별 감정 분포 리스트
        top_k: 상위 몇 개 추출

    Returns:
        (problem_categories, preferred_categories)
    """
    # 부정 비율 상위 → 문제 카테고리
    sorted_by_neg = sorted(category_sentiments, key=lambda x: x.neg_ratio, reverse=True)
    problem_categories = [
        ProblemCategory(category=c.category, neg_ratio=c.neg_ratio)
        for c in sorted_by_neg[:top_k]
    ]

    # 긍정 비율 상위 → 선호 카테고리
    sorted_by_pos = sorted(category_sentiments, key=lambda x: x.pos_ratio, reverse=True)
    preferred_categories = [
        PreferredCategory(category=c.category, pos_ratio=c.pos_ratio)
        for c in sorted_by_pos[:top_k]
    ]

    return problem_categories, preferred_categories


# ============================================================
# 카테고리별 불만 태그 집계
# ============================================================


def compute_category_complaints(
    trend_df: pd.DataFrame, aspect_col: str = "aspect_tags", top_k: int = 3
) -> Dict[str, List[CategoryComplaint]]:
    """
    카테고리별 주요 불만 태그 집계

    Args:
        trend_df: 분석용 DataFrame
        aspect_col: 태그 컬럼명
        top_k: 카테고리당 상위 태그 수

    Returns:
        Dict[str, List[CategoryComplaint]]
    """
    df = trend_df.dropna(subset=["menu_category"])

    result = {}
    for cat in df["menu_category"].unique():
        cat_df = df[df["menu_category"] == cat]

        # 태그 펼치기
        all_tags = []
        for tags in cat_df[aspect_col]:
            if isinstance(tags, list):
                all_tags.extend(tags)

        if not all_tags:
            continue

        # 빈도 집계
        tag_counts = Counter(all_tags).most_common(top_k)
        result[cat] = [CategoryComplaint(tag=tag, count=cnt) for tag, cnt in tag_counts]

    return result


# ============================================================
# 통합 함수: run_trend_analysis
# ============================================================


def run_trend_analysis(
    df_sentiments_reviews: pd.DataFrame,
    df_meal_plans: pd.DataFrame,
    df_food_db: pd.DataFrame,
    period_analysis: dict = None,  # 연호님 분석 결과 (optional)
    target_month: int = None,  # 분석 대상 월 (None이면 전체)
) -> TrendAnalysisResult:
    """
    트렌드 분석 통합 실행 함수

    Args:
        df_sentiments_reviews: 리뷰 감정분석 결과 (연호님 출력)
            - review_id, meal_type, sentiment_label, sentiment_score, aspect_tags
        df_meal_plans: 식단표
            - Date, meal_type, menu_key (또는 Rice, Soup, Main1, ...)
        df_food_db: 음식 DB
            - 음식명, 식품대분류명
        period_analysis: 연호님 run_period_analysis() 결과 (optional)
        target_month: 분석 대상 월 (예: 2 → 2월만 분석)

    Returns:
        TrendAnalysisResult
    """
    # --------------------------------------------------------
    # 1) 데이터 전처리
    # --------------------------------------------------------
    df = df_sentiments_reviews.copy()

    # Date 추출 (review_id에서)
    if "Date" not in df.columns and "review_id" in df.columns:
        df["Date"] = pd.to_datetime(
            df["review_id"].str.extract(r"R-(\d{8})-")[0], format="%Y%m%d"
        )

    # meal_type 매핑
    #    meal_type_map = {"LUNCH": "중식", "DINNER": "석식", "BREAKFAST": "조식"}
    #    if df["meal_type"].iloc[0] in meal_type_map:
    #        df["meal_type"] = df["meal_type"].map(meal_type_map)

    # sentiment_label 매핑
    sentiment_map = {"POSITIVE": "pos", "NEGATIVE": "neg", "NEUTRAL": "neu"}
    if df["sentiment_label"].iloc[0] in sentiment_map:
        df["sentiment_label"] = df["sentiment_label"].map(sentiment_map)

    # aspect_tags 컬럼 확인
    if "aspect_tags" not in df.columns:
        df["aspect_tags"] = df.get("aspect_hints", [[]]).apply(
            lambda x: x if isinstance(x, list) else []
        )

    # --------------------------------------------------------
    # 2) 월 필터링 (optional)
    # --------------------------------------------------------
    if target_month:
        df = df[df["Date"].dt.month == target_month]

    # --------------------------------------------------------
    # 3) 메뉴-카테고리 매핑
    # --------------------------------------------------------
    # 식단표 전처리
    meal_plans = df_meal_plans.copy()
    if "menu_key" not in meal_plans.columns:
        # (column 이름 차이 보정)
        menu_cols = ["rice", "soup", "main1", "main2", "side", "kimchi", "dessert"]
        # menu_cols = ["Rice", "Soup", "Main1", "Main2", "Side", "Kimchi", "Dessert"]
        existing_cols = [c for c in menu_cols if c in meal_plans.columns]
        if existing_cols:
            meal_plans["menu_key"] = (
                meal_plans[existing_cols]
                .fillna("")
                .apply(
                    lambda row: " | ".join([v for v in row if str(v).strip()]), axis=1
                )
            )

    # (column 및 column 값 동기화)
    df = df.rename(columns={"mealType": "meal_type"})
    meal_plans = meal_plans.rename(columns={"date": "Date", "mealType": "meal_type"})

    trend_df = map_menu_to_category(df, meal_plans, df_food_db)

    # --------------------------------------------------------
    # 4) 주차 정보 추가
    # --------------------------------------------------------
    trend_df["week_in_month"] = trend_df["Date"].apply(
        lambda d: (d.day - 1) // 7 + 1 if pd.notna(d) else None
    )

    # aspect_tags 컬럼 전파
    trend_df["aspect_tags_menu"] = trend_df["aspect_tags"]

    # --------------------------------------------------------
    # 5) 분석 실행
    # --------------------------------------------------------
    # 카테고리별 감정 분포
    category_sentiments = compute_category_sentiment(trend_df)

    # 주차별 부정 비율
    weekly_neg = compute_weekly_neg_trend(trend_df)

    # 선호도 변화
    pref_changes = compute_preference_changes(trend_df)

    # 불만 태그 변화
    tag_changes = compute_complaint_tag_changes(trend_df, "aspect_tags_menu")

    # 문제/선호 카테고리
    problem_cats, preferred_cats = compute_problem_preferred_categories(
        category_sentiments
    )

    # 카테고리별 불만 태그
    cat_complaints = compute_category_complaints(trend_df, "aspect_tags_menu")

    # --------------------------------------------------------
    # 6) 결과 생성
    # --------------------------------------------------------
    result = TrendAnalysisResult(
        period_start=trend_df["Date"].min().strftime("%Y-%m-%d"),
        period_end=trend_df["Date"].max().strftime("%Y-%m-%d"),
        total_records=len(trend_df),
        weekly_neg_trend=weekly_neg,
        category_sentiment_distribution=category_sentiments,
        preference_changes=pref_changes,
        complaint_tag_changes=tag_changes,
        problem_categories=problem_cats,
        preferred_categories=preferred_cats,
        category_complaints=cat_complaints,
    )

    # --------------------------------------------------------
    # 7) 연호님 데이터 통합 (optional)
    # --------------------------------------------------------
    if period_analysis:
        result.overall_problem_tags = period_analysis.get("problem_areas", [])
        result.overall_sentiment_distribution = period_analysis.get("reviews", {}).get(
            "sentiment_distribution", {}
        )
        result.deepdive_targets = [
            {"date": d.get("date"), "mealType": d.get("mealType")}
            for d in period_analysis.get("deepdives", [])
        ]

    return result
