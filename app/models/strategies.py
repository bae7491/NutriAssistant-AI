from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .operation import *
from .analysis_sentiment import *
from .analysis_periodic import *


class NutritionCompliance(BaseModel):
    """영양기준 준수 현황"""

    targetGroup: str
    totalMeals: int
    energyStandard: float
    proteinStandard: float
    compliance: Dict[str, Dict[str, Any]]
    failedMealsSample: List[Dict[str, Any]]
    failedMealsTotal: int
    weekdayCompliance: Dict[str, Dict[str, Any]]


class QualityScorecard(BaseModel):
    """식단 품질 종합 성적표"""

    scores: Dict[str, Dict[str, Any]]
    total: Dict[str, Any]
    goodPoints: List[str]
    improvePoints: List[str]
    nextMonthTarget: float


class RiskForecast(BaseModel):
    """잔반 위험도 예측"""

    avgRisk: float = 0.0
    highRiskRatio: float = 0.0
    highRiskCount: int = 0
    highRiskMeals: List[Dict[str, Any]] = []


class MenuStrategyItem(BaseModel):
    """메뉴 전략 항목"""

    strategyType: str
    targetCategory: Optional[str] = ""
    negativeRatio: Optional[float] = 0.0
    preferenceScore: Optional[float] = 0.0
    topIssues: Optional[List[str]] = []
    priority: Optional[str] = ""
    exampleMenus: Optional[List[str]] = []
    description: Optional[str] = ""
    trigger: Optional[str] = ""
    adjustment: Optional[str] = ""
    howToApply: Optional[List[str]] = []


class MenuStrategyResponse(BaseModel):
    #    meta: Dict[str, Any]
    nutritionCompliance: Optional[NutritionCompliance] = None
    qualityScorecard: Optional[QualityScorecard] = None
    riskForecast: Optional[RiskForecast] = None
    menuStrategies: List[MenuStrategyItem] = []
