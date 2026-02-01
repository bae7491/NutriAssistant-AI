from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .operation import *
from .analysis_sentiment import *
from .analysis_periodic import *
from .analysis_trends import *
from .strategies import *


class MonthlyReportRequestPayload(BaseModel):
    """
    [월간 분석 요청 내용]
    """
    userName: str
    year: int
    month: int
    targetGroup: Optional[str] = ""
    mealPlan: Optional[List[MealPlanItem]] = None
    reviews: Optional[List[Review]] = None
    posts: Optional[List[Post]] = None
    reviewAnalyses: Optional[List[ReviewAnalysis]] = None
    postAnalyses: Optional[List[PostAnalysis]] = None
    dailyInfo: Optional[list[DailyInfo]] = None
    dailyAnalyses: Optional[List[DailyAnalysis]] = None

class PeriodicReportMetadata(BaseModel):
    """
    [월간 분석 결과 - 메타데이터]
    """
    apiVersion:int = 1
    generatedAt:datetime = datetime.now().isoformat()

class PeriodicReportData(BaseModel):
    """
    [월간 분석 결과 - 데이터 부분]
    """
    periodicAnalysis: PeriodicAnalysisResult
    trendAnalysis: TrendAnalysisResult
    menuStrategy: MenuStrategyResponse

class PeriodicReportDoc(BaseModel):
    """
    [월간 분석 결과 - 문서 부분]
    """
    summary: str = ""
    leftover: str = ""
    satisfaction: str = ""
    issues: str = ""
    trendAnalysis: str = ""
    menuStrategies: str = ""
    opStrategies: str = ""

class MonthlyReport(BaseModel):
    """
    [월간 분석 결과]
    * 최종적으로 Response로서 리턴될 값
    """
    userName: str = "(NO_USERNAME)"
    year: int = 1
    month: int = 1
    metadata: PeriodicReportMetadata
    data: PeriodicReportData
    doc: PeriodicReportDoc
