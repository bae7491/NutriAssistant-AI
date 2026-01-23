from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FacilityFlags(BaseModel):
    has_oven: bool = True
    has_fryer: bool = True
    has_griddle: bool = True


class Options(BaseModel):
    seed: Optional[int] = None
    numGenerations: int = 150
    solPerPop: int = 60
    numParentsMating: int = 40
    keepParents: int = 0
    mutationPercentGenes: int = 20

    targetPrice: int = 5600
    costTolerance: float = 0.10
    maxPriceLimit: int = 6000
    facilityFlags: FacilityFlags = FacilityFlags()


class GenerateMonthRequest(BaseModel):
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    options: Options = Options()


class MealRow(BaseModel):
    Date: str
    Type: str  # "중식" | "석식"
    Rice: str
    Soup: str
    Main1: str
    Main2: str
    Side: str
    Kimchi: str
    Dessert: Optional[str] = None
    RawMenus: List[str]
    Kcal: int
    Carb: int
    Prot: int
    Fat: int
    Cost: int


class GenerateMonthResponse(BaseModel):
    year: int
    month: int
    generatedAt: str
    meals: List[MealRow]
    meta: Dict[str, Any] = {}


class FacilityAnalysisRequest(BaseModel):
    facility_text: str


class FacilityAnalysisResponse(BaseModel):
    has_oven: bool
    has_fryer: bool
    has_griddle: bool


class ReportAnalysisRequest(BaseModel):
    report_data: dict
    valid_menu_names: List[str]


class MenuWeight(BaseModel):
    menu_name: str
    weight: float
    reason: Optional[str] = None


class ReportAnalysisResponse(BaseModel):
    weights: List[MenuWeight]
    total_analyzed: int


class FacilityAnalysisRequest(BaseModel):
    """시설 분석 요청"""

    facility_text: str

    class Config:
        json_schema_extra = {
            "example": {"facility_text": "오븐 고장, 튀김기 없음, 회전식 조리기 2대"}
        }


class FacilityAnalysisResponse(BaseModel):
    """시설 분석 응답"""

    has_oven: bool
    has_fryer: bool
    has_griddle: bool

    class Config:
        json_schema_extra = {
            "example": {"has_oven": False, "has_fryer": False, "has_griddle": True}
        }


class ReportAnalysisRequest(BaseModel):
    """리포트 분석 요청"""

    report_data: dict
    valid_menu_names: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "report_data": {
                    "best_menus": ["김치찌개", "돈까스"],
                    "worst_menus": ["미역국"],
                    "comments": ["맛있었어요", "별로였어요"],
                },
                "valid_menu_names": ["김치찌개", "돈까스", "미역국", "된장찌개"],
            }
        }


class MenuWeight(BaseModel):
    """메뉴 가중치"""

    menu_name: str
    weight: float
    reason: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "menu_name": "김치찌개",
                "weight": 0.8,
                "reason": "학생들이 좋아하는 메뉴",
            }
        }


class ReportAnalysisResponse(BaseModel):
    """리포트 분석 응답"""

    weights: List[MenuWeight]
    total_analyzed: int

    class Config:
        json_schema_extra = {
            "example": {
                "weights": [
                    {"menu_name": "김치찌개", "weight": 0.8, "reason": "인기 메뉴"},
                    {"menu_name": "돈까스", "weight": 0.7, "reason": "만족도 높음"},
                ],
                "total_analyzed": 2,
            }
        }
