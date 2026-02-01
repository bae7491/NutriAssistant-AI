from __future__ import annotations

__list__ = [
    'SentimentAspectDetail',
    'ReviewAnalysis',
    'PostAnalysis'
]

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class SentimentAspectDetail(BaseModel):
    """
    감정분석 내용 (세부)
    """
    polarity: str
    hint: str

class ReviewAnalysis(BaseModel):
    """
    감정분석 결과 (게시글)
    """
    review_id: str
    meal_type: str
    rating_5: float
    sentiment_label: str
    sentiment_score: float
    sentiment_conf: Optional[float] = None
    aspect_tags: Optional[List[str]] = None
    aspect_hints: Optional[Dict[str, str]] = None
    aspect_details: Optional[Dict[str, SentimentAspectDetail]] = None
    evidence_phrases: Optional[List[str]] = None
    issue_flags: Optional[List[str]] = None

class PostAnalysis(BaseModel):
    """
    감정분석 결과 (글)
    """
    post_id: str
    category: str
    sentiment_label: str
    sentiment_score: float
    sentiment_conf: Optional[float] = None
    aspect_tags: Optional[List[str]] = None
    aspect_hints: Optional[Dict[str, str]] = None
    aspect_details: Optional[Dict[str, SentimentAspectDetail]] = None
    evidence_phrases: Optional[List[str]] = None
    issue_flags: Optional[List[str]] = None
