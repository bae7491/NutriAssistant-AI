from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FacilityFlags(BaseModel):
    """시설 플래그"""

    has_oven: bool = True
    has_fryer: bool = True
    has_griddle: bool = True


class Constraints(BaseModel):
    """식단 생성 제약사항"""

    target_price: int = Field(default=5600, description="목표 단가(원)")
    cost_tolerance: float = Field(default=0.10, description="단가 허용 오차 비율 (0~1)")
    max_price_limit: int = Field(default=6000, description="최대 단가 상한선(원)")
    cook_staff: int = Field(default=8, description="조리 인원")
    facility_flags: FacilityFlags = Field(
        default_factory=FacilityFlags, description="시설 사용 가능 여부 (체크박스 방식)"
    )
    facility_text: Optional[str] = Field(
        default=None,
        description="시설 현황 텍스트 (AI 자동 분석). null 또는 생략 시 facility_flags 사용",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_price": 5000,
                "cost_tolerance": 0.15,
                "max_price_limit": 5500,
                "cook_staff": 5,
                "facility_flags": {
                    "has_oven": False,
                    "has_fryer": True,
                    "has_griddle": True,
                },
            }
        }
    )


class Options(BaseModel):
    """식단 생성 옵션"""

    seed: Optional[int] = Field(default=None, description="랜덤 시드 (재현성)")
    numGenerations: int = Field(default=150, ge=50, le=500, description="세대 수")
    solPerPop: int = Field(default=60, ge=20, le=200, description="인구 크기")
    numParentsMating: int = Field(default=40, ge=10, le=100, description="교배 부모 수")
    keepParents: int = Field(default=0, ge=0, le=50, description="유지할 부모 수")
    mutationPercentGenes: int = Field(
        default=20, ge=1, le=100, description="돌연변이 비율(%)"
    )

    constraints: Constraints = Field(
        default_factory=Constraints, description="식단 생성 제약사항"
    )

    report: Optional[Dict[str, Any]] = Field(
        default=None, description="리포트 JSON (있으면 AI가 가중치 분석)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "numGenerations": 150,
                "solPerPop": 60,
                "constraints": {
                    "target_price": 5000,
                    "facility_flags": {
                        "has_oven": False,
                        "has_fryer": True,
                        "has_griddle": True,
                    },
                },
            }
        }
    )


class MonthMenuRequest(BaseModel):
    """월간 식단 생성 요청"""

    year: int = Field(..., ge=2020, le=2030, description="연도")
    month: int = Field(..., ge=1, le=12, description="월")
    options: Optional[Options] = Field(
        default=None, description="식단 생성 옵션 및 제약사항 (생략 시 기본값 사용)"
    )
    report: Optional[Dict[str, Any]] = Field(
        default=None, description="월간 리포트 JSON (Spring이 DB에서 조회하여 전달)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year": 2026,
                "month": 3,
                "options": {
                    "numGenerations": 150,
                    "constraints": {
                        "target_price": 5000,
                        "facility_text": (""),
                        "facility_flags": {
                            "has_oven": True,
                            "has_fryer": True,
                            "has_griddle": True,
                        },
                    },
                },
            }
        }
    )


class GenerateMonthRequest(BaseModel):
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    options: Options = Options()


class MealRow(BaseModel):
    """식단 행"""

    Date: str = Field(..., description="날짜 (YYYY-MM-DD)")
    Type: str = Field(..., description="식사 유형 (중식/석식)")
    Rice: str = Field(..., description="밥")
    Soup: str = Field(..., description="국")
    Main1: str = Field(..., description="주찬1")
    Main2: str = Field(..., description="주찬2")
    Side: str = Field(..., description="부찬")
    Kimchi: str = Field(..., description="김치")
    Dessert: Optional[str] = Field(default=None, description="후식")
    RawMenus: List[str] = Field(..., description="원본 메뉴명 리스트")
    Kcal: int = Field(..., description="칼로리")
    Carb: int = Field(..., description="탄수화물(g)")
    Prot: int = Field(..., description="단백질(g)")
    Fat: int = Field(..., description="지방(g)")
    Cost: int = Field(..., description="1인 식단 단가(원)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Date": "2026-03-03",
                "Type": "중식",
                "Rice": "쌀밥",
                "Soup": "된장찌개",
                "Main1": "제육볶음",
                "Main2": "계란찜",
                "Side": "콩나물무침",
                "Kimchi": "배추김치",
                "Dessert": None,
                "RawMenus": [
                    "쌀밥",
                    "된장찌개",
                    "제육볶음",
                    "계란찜",
                    "콩나물무침",
                    "배추김치",
                ],
                "Kcal": 850,
                "Carb": 120,
                "Prot": 35,
                "Fat": 25,
                "Cost": 4580,
            }
        }
    )


class GenerateMonthResponse(BaseModel):
    """월간 식단 생성 응답"""

    year: int = Field(..., description="연도")
    month: int = Field(..., description="월")
    generatedAt: str = Field(..., description="생성 시각 (ISO 8601)")
    meals: List[MealRow] = Field(..., description="생성된 식단 목록")
    meta: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year": 2026,
                "month": 3,
                "generatedAt": "2026-01-23T14:30:00",
                "meals": [
                    {
                        "Date": "2026-03-03",
                        "Type": "중식",
                        "Rice": "쌀밥",
                        "Soup": "된장찌개",
                        "Main1": "제육볶음",
                        "Main2": "계란찜",
                        "Side": "콩나물무침",
                        "Kimchi": "배추김치",
                        "Dessert": None,
                        "RawMenus": [
                            "쌀밥",
                            "된장찌개",
                            "제육볶음",
                            "계란찜",
                            "콩나물무침",
                            "배추김치",
                        ],
                        "Kcal": 850,
                        "Carb": 120,
                        "Prot": 35,
                        "Fat": 25,
                        "Cost": 4580,
                    }
                ],
                "meta": {
                    "gaParams": {"num_generations": 150},
                    "appliedConstraints": {
                        "target_price": 5600,
                        "facility_flags": {
                            "has_oven": True,
                            "has_fryer": True,
                            "has_griddle": True,
                        },
                    },
                },
            }
        }
    )


class FacilityAnalysisRequest(BaseModel):
    """시설 분석 요청"""

    facility_text: str = Field(..., description="시설 현황 설명 텍스트")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "facility_text": "회전식 조리기 2대, 스팀솥 4대가 있습니다. 오븐과 튀김기는 고장났습니다."
            }
        }
    )


class FacilityAnalysisResponse(BaseModel):
    """시설 분석 응답"""

    has_oven: bool = Field(..., description="오븐 사용 가능 여부")
    has_fryer: bool = Field(..., description="튀김기 사용 가능 여부")
    has_griddle: bool = Field(..., description="철판 사용 가능 여부")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"has_oven": False, "has_fryer": False, "has_griddle": True}
        }
    )


class ReportAnalysisRequest(BaseModel):
    """리포트 분석 요청"""

    report_data: dict = Field(..., description="리포트 데이터 (JSON)")
    valid_menu_names: List[str] = Field(..., description="유효한 메뉴명 목록")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report_data": {
                    "best_menus": ["김치찌개", "돈까스"],
                    "worst_menus": ["미역국"],
                    "comments": ["맛있었어요", "별로였어요"],
                },
                "valid_menu_names": ["김치찌개", "돈까스", "미역국", "된장찌개"],
            }
        }
    )


class MenuWeight(BaseModel):
    """메뉴 가중치"""

    menu_name: str = Field(..., description="메뉴명")
    weight: float = Field(..., description="가중치 (-1.0 ~ 1.0)")
    reason: Optional[str] = Field(default=None, description="가중치 부여 이유")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "menu_name": "김치찌개",
                "weight": 0.8,
                "reason": "학생들이 좋아하는 메뉴",
            }
        }
    )


class ReportAnalysisResponse(BaseModel):
    """리포트 분석 응답"""

    weights: List[MenuWeight] = Field(..., description="메뉴별 가중치 목록")
    total_analyzed: int = Field(..., description="분석된 메뉴 수")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weights": [
                    {"menu_name": "김치찌개", "weight": 0.8, "reason": "인기 메뉴"},
                    {"menu_name": "돈까스", "weight": 0.7, "reason": "만족도 높음"},
                ],
                "total_analyzed": 2,
            }
        }
    )
