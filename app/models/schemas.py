from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class FacilityFlags(BaseModel):
    """시설 플래그"""

    has_oven: bool = True
    has_fryer: bool = True
    has_griddle: bool = True


class Constraints(BaseModel):
    """식단 생성 제약사항"""

    nutrition_key: Optional[str] = Field(
        default=None,
        description="영양 기준 키 (ELEMENTARY, MIDDLE_MALE, MIDDLE_FEMALE, MIDDLE_COED, HIGH_MALE, HIGH_FEMALE, HIGH_COED)",
    )
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
                "nutrition_key": "HIGH_MALE",
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


class NewMenuInput(BaseModel):
    """신메뉴 입력 (Spring에서 전달)"""

    food_code: str = Field(..., description="음식 코드")
    food_name: str = Field(..., description="음식명")
    category: str = Field(..., description="카테고리 (예: 디저트류, 구이류)")
    kcal: float = Field(default=0, description="칼로리(kcal)")
    protein: float = Field(default=0, description="단백질(g)")
    fat: float = Field(default=0, description="지방(g)")
    carbs: float = Field(default=0, description="탄수화물(g)")
    calcium: float = Field(default=0, description="칼슘(mg)")
    iron: float = Field(default=0, description="철분(mg)")
    vitamin_a: float = Field(default=0, description="비타민A(μg RAE)")
    thiamin: float = Field(default=0, description="티아민/비타민B1(mg)")
    riboflavin: float = Field(default=0, description="리보플라빈/비타민B2(mg)")
    vitamin_c: float = Field(default=0, description="비타민C(mg)")
    ingredients: Optional[str] = Field(default=None, description="재료 목록")
    allergy_info: Optional[str] = Field(default=None, description="알레르기 정보 (예: 1, 2, 5, 6)")
    recipe: Optional[str] = Field(default=None, description="레시피")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "food_code": "두바이초콜릿쿠키",
                "food_name": "두바이초콜릿쿠키",
                "category": "디저트류",
                "kcal": 500,
                "protein": 6,
                "fat": 25,
                "carbs": 62,
                "calcium": 60,
                "iron": 3.2,
                "vitamin_a": 50,
                "thiamin": 0.22,
                "riboflavin": 0.18,
                "vitamin_c": 0,
                "ingredients": "밀가루, 버터, 설탕, ...",
                "allergy_info": "1, 2, 5, 6",
                "recipe": "1. 오븐을 170도로 예열한다.\n2. ...",
            }
        }
    )


class MonthMenuRequest(BaseModel):
    """월간 식단 생성 요청"""

    year: int = Field(..., ge=2020, le=2030, description="연도")
    month: int = Field(..., ge=1, le=12, description="월")
    school_id: Optional[int] = Field(default=None, description="학교 ID")
    nutrition_key: Optional[str] = Field(
        default=None,
        description="영양 기준 키 (ELEMENTARY, MIDDLE_MALE, MIDDLE_FEMALE, MIDDLE_COED, HIGH_MALE, HIGH_FEMALE, HIGH_COED)",
    )
    options: Optional[Options] = Field(
        default=None, description="식단 생성 옵션 및 제약사항 (생략 시 기본값 사용)"
    )
    report: Optional[Dict[str, Any]] = Field(
        default=None, description="월간 리포트 JSON (Spring이 DB에서 조회하여 전달)"
    )
    new_menus: Optional[List[NewMenuInput]] = Field(
        default=None,
        description="신메뉴 목록 (Spring에서 전달, 기존 음식 DB와 함께 사용)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year": 2026,
                "month": 3,
                "school_id": 1,
                "nutrition_key": "HIGH_MALE",
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
                "new_menus": [
                    {
                        "food_code": "두바이초콜릿쿠키",
                        "food_name": "두바이초콜릿쿠키",
                        "category": "디저트류",
                        "kcal": 500,
                        "protein": 6,
                        "fat": 25,
                        "carbs": 62,
                        "allergy_info": "1, 2, 5, 6",
                    }
                ],
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


# ==============================================================================
# 신메뉴 생성 관련 스키마
# ==============================================================================
class NewMenuGenerationRequest(BaseModel):
    """신메뉴 생성 요청"""

    use_trend: bool = Field(default=True, description="트렌드 분석 사용 여부")
    use_board: bool = Field(default=True, description="게시판 분석 사용 여부")
    trend_days: int = Field(
        default=30, ge=1, le=30, description="트렌드 분석 기간 (일)"
    )
    count: int = Field(default=2, ge=1, le=20, description="생성할 신메뉴 수")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "use_trend": True,
                "use_board": True,
                "trend_days": 30,
                "count": 2,
            }
        }
    )


class NutritionInfo(BaseModel):
    """영양 정보"""

    kcal: float = Field(..., description="칼로리(kcal)")
    carbs: float = Field(..., description="탄수화물(g)")
    protein: float = Field(..., description="단백질(g)")
    fat: float = Field(..., description="지방(g)")
    calcium: float = Field(default=0, description="칼슘(mg)")
    iron: float = Field(default=0, description="철분(mg)")
    vitamin_a: float = Field(default=0, description="비타민A(μg RAE)")
    thiamin: float = Field(default=0, description="티아민/비타민B1(mg)")
    riboflavin: float = Field(default=0, description="리보플라빈/비타민B2(mg)")
    vitamin_c: float = Field(default=0, description="비타민C(mg)")
    serving_basis: str = Field(
        default="100g", description="영양성분 기준량 (예: 100g, 100ml)"
    )
    food_weight: str = Field(default="100ml", description="식품 중량 (예: 150ml)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kcal": 350,
                "carbs": 45,
                "protein": 5,
                "fat": 18,
                "calcium": 20,
                "iron": 1.5,
                "vitamin_a": 10,
                "thiamin": 0.1,
                "riboflavin": 0.1,
                "vitamin_c": 5,
                "serving_basis": "100g",
                "food_weight": 150,
            }
        }
    )


class NewMenuItem(BaseModel):
    """신메뉴 항목"""

    menu_name: str = Field(..., description="메뉴명")
    category: str = Field(..., description="카테고리")
    source: Literal["trend", "board", "hybrid"] = Field(
        ..., description="메뉴 출처 (trend/board/hybrid)"
    )
    ingredients: List[str] = Field(..., description="재료 목록")
    recipe_steps: List[str] = Field(..., description="레시피 단계")
    allergens: List[int] = Field(..., description="알레르기 번호 목록")
    nutrition: NutritionInfo = Field(..., description="영양 정보")
    matched_menu: Optional[str] = Field(None, description="영양 매칭된 기존 메뉴")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도 (0~1)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "menu_name": "두바이초콜릿쿠키",
                "category": "디저트류",
                "source": "trend",
                "ingredients": ["밀가루", "카다이프", "피스타치오", "초콜릿"],
                "recipe_steps": ["1. 반죽 준비", "2. 성형", "3. 굽기"],
                "allergens": [1, 2, 6],
                "nutrition": {"kcal": 350, "carbs": 45, "protein": 5, "fat": 18},
                "matched_menu": "초코칩쿠키",
                "confidence": 0.85,
            }
        }
    )


class AnalysisSummary(BaseModel):
    """분석 요약"""

    trend_keywords: List[str] = Field(
        default_factory=list, description="트렌드 키워드 목록"
    )
    board_votes: Dict[str, int] = Field(
        default_factory=dict, description="게시판 득표 현황"
    )
    total_candidates: int = Field(..., description="총 후보 수")
    final_count: int = Field(..., description="최종 선정 수")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trend_keywords": ["두바이초콜릿", "로제떡볶이"],
                "board_votes": {"로제파스타": 15, "마라탕": 12},
                "total_candidates": 25,
                "final_count": 5,
            }
        }
    )


class NewMenuGenerationResponse(BaseModel):
    """신메뉴 생성 응답"""

    generated_at: str = Field(..., description="생성 시각 (ISO 8601)")
    new_menus: List[NewMenuItem] = Field(..., description="생성된 신메뉴 목록")
    analysis_summary: AnalysisSummary = Field(..., description="분석 요약")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "generated_at": "2026-01-31T10:00:00",
                "new_menus": [
                    {
                        "menu_name": "두바이초콜릿쿠키",
                        "category": "디저트류",
                        "source": "trend",
                        "ingredients": ["밀가루", "카다이프", "피스타치오"],
                        "recipe_steps": ["1. 반죽 준비", "2. 성형"],
                        "allergens": [1, 2, 6],
                        "nutrition": {
                            "kcal": 350,
                            "carbs": 45,
                            "protein": 5,
                            "fat": 18,
                        },
                        "matched_menu": "초코칩쿠키",
                        "confidence": 0.85,
                    }
                ],
                "analysis_summary": {
                    "trend_keywords": ["두바이초콜릿"],
                    "board_votes": {"로제파스타": 15},
                    "total_candidates": 25,
                    "final_count": 5,
                },
            }
        }
    )


# ==============================================================================
# [NEW] 월간 운영 자료 (MonthlyOpsDoc) 관련 스키마
# ==============================================================================

class ReportFile(BaseModel):
    """보고서 내 파일 정보"""
    id: int
    file_name: str
    file_type: str
    s3_path: str
    created_at: str

class MonthlyOpsDocData(BaseModel):
    """월간 운영 자료 핵심 데이터"""
    id: int
    school_id: int
    title: str
    year: int
    month: int
    status: str
    created_at: str
    files: Optional[List[ReportFile]] = Field(default=None, description="상세 조회 시 포함")

class MonthlyOpsDocCreateRequest(BaseModel):
    """생성 요청 (POST)"""
    school_id: int
    year: int
    month: int

class MonthlyOpsDocCreateResponse(BaseModel):
    """생성 응답 (POST)"""
    status: str
    message: str
    data: MonthlyOpsDocData

class PaginationInfo(BaseModel):
    """페이지네이션 정보"""
    current_page: int
    total_pages: int
    total_items: int
    page_size: int

# ✅ [개선] 목록 조회 시 data 내부 구조를 명시적으로 정의 (Swagger 문서를 위해)
class MonthlyOpsDocListData(BaseModel):
    reports: List[MonthlyOpsDocData]
    pagination: PaginationInfo

class MonthlyOpsDocListResponse(BaseModel):
    """목록 조회 응답 (GET)"""
    status: str
    data: MonthlyOpsDocListData

class MonthlyOpsDocDetailResponse(BaseModel):
    """상세 조회 응답 (GET)"""
    status: str
    data: MonthlyOpsDocData
