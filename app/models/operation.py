from __future__ import annotations

__list__ = [
    'Review',
    'Post',
    'MealPlanItem',
    'DailyInfo',
    'DailyAnalysisKpis',
    'DailyAnalysisDistributions',
    'DailyTopAspect',
    'DailyTopNegativeAspect',
    'DailyAnalysis'
]

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class Review(BaseModel):
    """
    [운영 데이터] (식사) 리뷰
    """
    reviewId: str
    date: date
    mealType: str
    rating: float
    content: str

class Post(BaseModel):
    """
    [운영 데이터] 게시글
    """
    postId: str
    date: date
    category: str
    title: str
    content: str

class MealPlanItem(BaseModel):
    date: str
    mealType: str
    rice: Optional[str] = None
    soup: Optional[str] = None
    main1: Optional[str] = None
    main2: Optional[str] = None
    side: Optional[str] = None
    kimchi: Optional[str] = None
    dessert: Optional[str] = None
    kcal: Optional[float] = None
    protein: Optional[float] = None
    carb: Optional[float] = None
    fat: Optional[float] = None

class DailyInfo(BaseModel):
    date: date
    mealType: str
    servedProxy: int
    missedProxy: Optional[int] = None
    leftoverKg: float

class DailyAnalysisKpis(BaseModel):
    review_count: int = 0
    post_count: int = 0
    avg_rating_5: float = 0.0
    avg_review_sentiment: float = 0.0
    avg_post_sentiment: float = 0.0
    overall_sentiment: float = 0.0

class DailyAnalysisDistributions(BaseModel):
    reviews: Dict[str, int]
    posts: Dict[str, int]
    post_categories: Dict[str, int]

class DailyTopAspect(BaseModel):
    tag: str
    count: int
    polarity: str

class DailyTopNegativeAspect(BaseModel):
    tag: str
    negative_rate: float
    counts: Dict[str, int]

class DailyAnalysis(BaseModel):
    date: date
    mealType: str
    kpis: DailyAnalysisKpis
    distributions: DailyAnalysisDistributions
    top_aspects: Optional[List[DailyTopAspect]] = None
    aspect_polarity_distribution: Optional[Dict[str, Dict[str, int]]] = None
    top_negative_aspects: Optional[List[DailyTopNegativeAspect]] = None
    top_issues: Optional[List[str]] = None
    alerts: Optional[List[str]] = None
