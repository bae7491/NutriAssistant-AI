from __future__ import annotations

__list__ = [
    "RatingKPI",
    "LeftoverKPI",
    "ReviewAggregate",
    "PostAggregate",
    "AspectSummary",
    "ProblemArea",
    "DeepDiveResult",
    "PeriodicAnalysisResult",
]

from pydantic import BaseModel


class RatingKPI(BaseModel):
    """
    <월간> KPI - 평점: 평균, 최저, 최대, 추세
    """

    avg_rating: float | None
    min_rating_day: dict
    max_rating_day: dict
    rating_trend: list


class LeftoverKPI(BaseModel):
    """
    <월간> KPI - 잔반률: 평균, 결식률, 지표 상 최하위 일자
    """

    avg_leftover_rate: float | None
    avg_missed_rate: float | None
    worst_cases: list


class ReviewAggregate(BaseModel):
    """
    <월간> 리뷰 집계 결과: 갯수, 분포(감정), 분포(조식/석식), 분포(평점)
    """

    count: int
    sentiment_distribution: dict
    mealType_distribution: dict
    rating_distribution: dict


class PostAggregate(BaseModel):
    """
    <월간> 게시물 집계 결과: 갯수, 분포(게시물 분류), 분포(감정), 게시물-감정 Matrix, Issue들
    """

    count: int
    category_distribution: dict
    sentiment_distribution: dict
    category_sentiment_matrix: dict
    issue_flags: dict


class AspectSummary(BaseModel):
    """
    감정 Aspect 요약: tag, 긍정 수, 중립 수, 부정 수, 총합, 부정 비율
    """

    tag: str
    POSITIVE: int
    NEUTRAL: int
    NEGATIVE: int
    total: int
    neg_rate: float


class ProblemArea(BaseModel):
    """
    문제 발생 영역: tag, 총 언급 수, 부정 비율, 부정 점수
    """

    tag: str
    total_mentions: int
    neg_rate: float
    problem_score: float


class DeepDiveResult(BaseModel):
    """
    심층분석 결과: 일자, (조식/중식/석식), 메뉴, 감정 요약, (게시물/리뷰 내의) 근거 문구
    """

    date: str
    mealType: str
    menus: dict
    aspect_summary: list[AspectSummary]
    evidence_phrases: list[str]


class PeriodicAnalysisResult(BaseModel):
    """
    [기간 분석 결과]
    """

    kpis: dict
    reviews: ReviewAggregate
    posts: PostAggregate
    problem_areas: list[ProblemArea]
    deepdives: list[DeepDiveResult]
