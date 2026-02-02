from __future__ import annotations

__list__ = [
    "CategorySentiment",
    "PreferenceChange",
    "ComplaintTagChange",
    "ProblemCategory",
    "PreferredCategory",
    "CategoryComplaint",
    "TrendAnalysisResult",
]

from typing import Dict, List, Optional

from pydantic import BaseModel


class CategorySentiment(BaseModel):
    """카테고리별 감정 분포"""

    category: str
    pos_ratio: float
    neu_ratio: float
    neg_ratio: float
    count: int


class PreferenceChange(BaseModel):
    """카테고리 선호도 변화"""

    category: str
    direction: str  # "increase" or "decrease"
    change_percent: float


class ComplaintTagChange(BaseModel):
    """불만 태그 변화"""

    tag: str
    direction: str  # "increase" or "decrease"
    change_percent: float


class ProblemCategory(BaseModel):
    """문제 카테고리"""

    category: str
    neg_ratio: float


class PreferredCategory(BaseModel):
    """선호 카테고리"""

    category: str
    pos_ratio: float


class CategoryComplaint(BaseModel):
    """카테고리별 불만 태그"""

    tag: str
    count: int


class TrendAnalysisResult(BaseModel):
    """트렌드 분석 최종 결과"""

    # 분석 정보
    period_start: str
    period_end: str
    total_records: int

    # 주차별 부정 비율
    weekly_neg_trend: Dict[int, float]

    # 카테고리별 감정 분포
    category_sentiment_distribution: List[CategorySentiment]

    # 선호도 변화
    preference_changes: List[PreferenceChange]

    # 불만 태그 변화
    complaint_tag_changes: List[ComplaintTagChange]

    # 문제/선호 카테고리
    problem_categories: List[ProblemCategory]
    preferred_categories: List[PreferredCategory]

    # 카테고리별 불만 태그
    category_complaints: Dict[str, List[CategoryComplaint]]

    # 연호님 데이터 통합 (optional)
    overall_problem_tags: Optional[List[dict]] = None
    overall_sentiment_distribution: Optional[dict] = None
    deepdive_targets: Optional[List[dict]] = None
