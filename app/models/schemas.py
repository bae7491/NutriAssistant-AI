from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# ==============================================================================
# 기존 식단/시설 관련 스키마 (변경 없음, 주석 보강)
# ==============================================================================

class FacilityFlags(BaseModel):
    """
    시설 보유 여부 플래그
    - 식단 생성 시 조리 가능 여부를 판단하는 기준이 됩니다.
    """
    has_oven: bool = True       # 오븐 보유 여부
    has_fryer: bool = True      # 튀김기 보유 여부
    has_griddle: bool = True    # 부침기/철판 보유 여부


class Constraints(BaseModel):
    """
    식단 생성 시 적용할 제약사항 정의
    - 영양 기준, 단가, 조리 인원, 시설 정보 등을 포함합니다.
    """
    nutrition_key: Optional[str] = Field(
        default=None,
        description="영양 기준 키 (예: HIGH_MALE, MIDDLE_COED 등). 학교급식법 기준 적용.",
    )
    target_price: int = Field(default=5600, description="목표 식단 단가(원)")
    cost_tolerance: float = Field(default=0.10, description="단가 허용 오차 비율 (0.1 = 10%)")
    max_price_limit: int = Field(default=6000, description="최대 허용 단가(원)")
    cook_staff: int = Field(default=8, description="조리 종사자 수 (인건비/노동강도 고려용)")
    facility_flags: FacilityFlags = Field(
        default_factory=FacilityFlags, description="시설 사용 가능 여부 (체크박스 선택값)"
    )
    facility_text: Optional[str] = Field(
        default=None,
        description="시설 현황 텍스트 (AI가 분석하여 flags를 덮어씌울 수 있음)",
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
    """
    유전 알고리즘(GA) 기반 식단 생성 옵션
    """
    seed: Optional[int] = Field(default=None, description="결과 재현을 위한 랜덤 시드")
    numGenerations: int = Field(default=150, ge=50, le=500, description="진화 세대 수")
    solPerPop: int = Field(default=60, ge=20, le=200, description="한 세대의 개체 수")
    numParentsMating: int = Field(default=40, ge=10, le=100, description="교배에 참여할 부모 수")
    keepParents: int = Field(default=0, ge=0, le=50, description="다음 세대로 유지할 우수 부모 수")
    mutationPercentGenes: int = Field(
        default=20, ge=1, le=100, description="유전자 돌연변이 발생 확률(%)"
    )

    constraints: Constraints = Field(
        default_factory=Constraints, description="식단 생성 제약조건 객체"
    )

    report: Optional[Dict[str, Any]] = Field(
        default=None, description="월간 리포트 JSON (이전 데이터 기반 가중치 분석용)"
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
    """
    신메뉴 정보 입력 DTO
    - Spring Boot에서 DB에 있는 신메뉴 정보를 조회해서 Python으로 보낼 때 사용
    """
    food_code: str = Field(..., description="음식 코드 (고유 식별자)")
    food_name: str = Field(..., description="음식명")
    category: str = Field(..., description="카테고리 (예: 디저트류, 구이류)")
    kcal: float = Field(default=0, description="에너지(kcal)")
    protein: float = Field(default=0, description="단백질(g)")
    fat: float = Field(default=0, description="지방(g)")
    carbs: float = Field(default=0, description="탄수화물(g)")
    calcium: float = Field(default=0, description="칼슘(mg)")
    iron: float = Field(default=0, description="철분(mg)")
    vitamin_a: float = Field(default=0, description="비타민A(μg RAE)")
    thiamin: float = Field(default=0, description="티아민(mg)")
    riboflavin: float = Field(default=0, description="리보플라빈(mg)")
    vitamin_c: float = Field(default=0, description="비타민C(mg)")
    ingredients: Optional[str] = Field(default=None, description="주재료 목록 문자열")
    allergy_info: Optional[str] = Field(
        default=None, description="알레르기 유발 물질 번호 (예: 1, 2, 5)"
    )
    recipe: Optional[str] = Field(default=None, description="조리법 설명")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "food_code": "NEW001",
                "food_name": "두바이초콜릿쿠키",
                "category": "디저트류",
                "kcal": 500,
                "protein": 6,
                "allergy_info": "1, 2, 5, 6",
            }
        }
    )


class MonthMenuRequest(BaseModel):
    """
    월간 식단 생성 요청 DTO
    - 사용자가 입력한 조건과 DB 데이터를 모두 포함하여 Python으로 전달됨
    """
    year: int = Field(..., ge=2020, le=2030, description="대상 연도")
    month: int = Field(..., ge=1, le=12, description="대상 월")
    school_id: Optional[int] = Field(default=None, description="학교 ID")
    nutrition_key: Optional[str] = Field(
        default=None,
        description="영양 기준 키 (생략 시 기본값)",
    )
    options: Optional[Options] = Field(
        default=None, description="알고리즘 옵션 및 제약사항"
    )
    report: Optional[Dict[str, Any]] = Field(
        default=None, description="월간 리포트 데이터 (메뉴 선호도 반영용)"
    )
    new_menus: Optional[List[NewMenuInput]] = Field(
        default=None,
        description="강제로 포함하거나 추천할 신메뉴 리스트",
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
                    "constraints": {"target_price": 5000},
                },
                "new_menus": [],
            }
        }
    )


class GenerateMonthRequest(BaseModel):
    """간소화된 월간 생성 요청 (테스트용)"""
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    options: Options = Options()


class MealRow(BaseModel):
    """
    생성된 식단 결과 (1끼니) 구조
    - 날짜, 메뉴 구성, 영양 정보, 단가 등을 포함
    """
    Date: str = Field(..., description="날짜 (YYYY-MM-DD)")
    Type: str = Field(..., description="급식 유형 (중식/석식)")
    Rice: str = Field(..., description="밥 메뉴명")
    Soup: str = Field(..., description="국 메뉴명")
    Main1: str = Field(..., description="주찬1 메뉴명")
    Main2: str = Field(..., description="주찬2 메뉴명")
    Side: str = Field(..., description="부찬 메뉴명")
    Kimchi: str = Field(..., description="김치 메뉴명")
    Dessert: Optional[str] = Field(default=None, description="후식 메뉴명")
    RawMenus: List[str] = Field(..., description="모든 메뉴명을 리스트로 나열")
    Kcal: int = Field(..., description="총 칼로리")
    Carb: int = Field(..., description="탄수화물(g)")
    Prot: int = Field(..., description="단백질(g)")
    Fat: int = Field(..., description="지방(g)")
    Cost: int = Field(..., description="1인당 예상 단가(원)")

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
                "RawMenus": ["쌀밥", "된장찌개", "제육볶음", "계란찜", "콩나물무침", "배추김치"],
                "Kcal": 850,
                "Carb": 120,
                "Prot": 35,
                "Fat": 25,
                "Cost": 4580,
            }
        }
    )


class GenerateMonthResponse(BaseModel):
    """
    월간 식단 생성 최종 응답 DTO
    """
    year: int = Field(..., description="연도")
    month: int = Field(..., description="월")
    generatedAt: str = Field(..., description="생성 시점 (ISO 8601)")
    meals: List[MealRow] = Field(..., description="생성된 식단 리스트")
    meta: Dict[str, Any] = Field(default_factory=dict, description="메타데이터 (알고리즘 파라미터 등)")


class FacilityAnalysisRequest(BaseModel):
    """시설 현황 텍스트 분석 요청"""
    facility_text: str = Field(..., description="영양사가 입력한 시설 설명")


class FacilityAnalysisResponse(BaseModel):
    """시설 현황 분석 결과 (플래그 반환)"""
    has_oven: bool = Field(..., description="오븐 사용 가능 여부")
    has_fryer: bool = Field(..., description="튀김기 사용 가능 여부")
    has_griddle: bool = Field(..., description="철판 사용 가능 여부")


class ReportAnalysisRequest(BaseModel):
    """리포트 분석 요청 (메뉴 가중치 산출용)"""
    report_data: dict = Field(..., description="기존 리포트 데이터")
    valid_menu_names: List[str] = Field(..., description="DB에 존재하는 유효 메뉴명 목록")


class MenuWeight(BaseModel):
    """개별 메뉴 가중치 정보"""
    menu_name: str = Field(..., description="메뉴명")
    weight: float = Field(..., description="가중치 (-1.0 ~ 1.0)")
    reason: Optional[str] = Field(default=None, description="가중치 부여 사유")


class ReportAnalysisResponse(BaseModel):
    """리포트 분석 결과 응답"""
    weights: List[MenuWeight] = Field(..., description="메뉴별 가중치 리스트")
    total_analyzed: int = Field(..., description="분석된 총 메뉴 수")


# ==============================================================================
# 신메뉴 생성(Trend/Board) 관련 스키마
# ==============================================================================

class NewMenuGenerationRequest(BaseModel):
    """트렌드/게시판 기반 신메뉴 생성 요청"""
    use_trend: bool = Field(default=True, description="외부 트렌드 데이터 사용 여부")
    use_board: bool = Field(default=True, description="내부 게시판 데이터 사용 여부")
    trend_days: int = Field(default=30, ge=1, le=30, description="분석 기간")
    count: int = Field(default=2, ge=1, le=20, description="생성할 메뉴 개수")


class NutritionInfo(BaseModel):
    """신메뉴 영양 정보"""
    kcal: float = Field(..., description="칼로리")
    carbs: float = Field(..., description="탄수화물")
    protein: float = Field(..., description="단백질")
    fat: float = Field(..., description="지방")
    calcium: float = Field(default=0, description="칼슘")
    iron: float = Field(default=0, description="철분")
    vitamin_a: float = Field(default=0, description="비타민A")
    thiamin: float = Field(default=0, description="티아민")
    riboflavin: float = Field(default=0, description="리보플라빈")
    vitamin_c: float = Field(default=0, description="비타민C")
    serving_basis: str = Field(default="100g", description="기준 중량")
    food_weight: str = Field(default="100ml", description="1인 제공량")


class NewMenuItem(BaseModel):
    """생성된 신메뉴 상세 정보"""
    menu_name: str = Field(..., description="메뉴명")
    category: str = Field(..., description="카테고리")
    source: Literal["trend", "board", "hybrid"] = Field(..., description="출처")
    ingredients: List[str] = Field(..., description="식재료")
    recipe_steps: List[str] = Field(..., description="조리법")
    allergens: List[int] = Field(..., description="알레르기 번호")
    nutrition: NutritionInfo = Field(..., description="영양 정보")
    matched_menu: Optional[str] = Field(None, description="유사한 기존 메뉴")
    confidence: float = Field(..., ge=0, le=1, description="AI 신뢰도")


class AnalysisSummary(BaseModel):
    """신메뉴 생성 분석 요약"""
    trend_keywords: List[str] = Field(default_factory=list, description="추출된 트렌드 키워드")
    board_votes: Dict[str, int] = Field(default_factory=dict, description="게시판 투표 현황")
    total_candidates: int = Field(..., description="후보군 수")
    final_count: int = Field(..., description="최종 선정 수")


class NewMenuGenerationResponse(BaseModel):
    """신메뉴 생성 응답"""
    generated_at: str = Field(..., description="생성 시간")
    new_menus: List[NewMenuItem] = Field(..., description="신메뉴 리스트")
    analysis_summary: AnalysisSummary = Field(..., description="분석 요약 정보")


# ==============================================================================
# 월간 운영 자료 (MonthlyOpsDoc) 관련 스키마
# ==============================================================================

class ReportFile(BaseModel):
    """운영 보고서 첨부 파일 정보"""
    id: int
    file_name: str
    file_type: str
    s3_path: str
    created_at: str

class MonthlyOpsDocData(BaseModel):
    """월간 운영 보고서 본문 데이터"""
    id: int
    school_id: int
    title: str
    year: int
    month: int
    status: str
    created_at: str
    files: Optional[List[ReportFile]] = Field(
        default=None, description="첨부 파일 리스트"
    )

class MonthlyOpsDocCreateRequest(BaseModel):
    """운영 보고서 생성 요청"""
    school_id: int
    year: int
    month: int

class MonthlyOpsDocCreateResponse(BaseModel):
    """운영 보고서 생성 응답"""
    status: str
    message: str
    data: MonthlyOpsDocData

class PaginationInfo(BaseModel):
    """페이지네이션 정보"""
    current_page: int
    total_pages: int
    total_items: int
    page_size: int

class MonthlyOpsDocListData(BaseModel):
    """운영 보고서 목록 데이터"""
    reports: List[MonthlyOpsDocData]
    pagination: PaginationInfo

class MonthlyOpsDocListResponse(BaseModel):
    """운영 보고서 목록 응답 (GET)"""
    status: str
    data: MonthlyOpsDocListData

class MonthlyOpsDocDetailResponse(BaseModel):
    """운영 보고서 상세 응답 (GET)"""
    status: str
    data: MonthlyOpsDocData


# ==============================================================================
# [★신규 추가] Java Spring Boot 연동용 일일 감성 분석 스키마
# - Java의 FastApiDto.Request / Response와 정확히 일치해야 합니다.
# ==============================================================================

class DailyAnalysisRequest(BaseModel):
    """
    일일 리뷰 감성 분석 요청 DTO
    - Java: FastApiDto.Request 클래스와 대응
    """
    schoolId: int = Field(..., description="학교 식별 ID (Java: Long)")
    targetDate: str = Field(..., description="분석 대상 날짜 YYYY-MM-DD (Java: String)")
    reviewTexts: List[str] = Field(..., description="분석할 리뷰 텍스트 리스트 (Java: List<String>)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "schoolId": 101,
                "targetDate": "2026-05-05",
                "reviewTexts": ["급식이 너무 맛있어요!", "김치가 좀 짜요.", "양이 적어요"]
            }
        }
    )

class DailyAnalysisResponse(BaseModel):
    """
    일일 리뷰 감성 분석 응답 DTO
    - Java: FastApiDto.Response 클래스와 대응
    """
    sentimentLabel: str = Field(..., description="전체적인 감성 라벨 (POSITIVE/NEGATIVE/NEUTRAL)")
    sentimentScore: float = Field(..., description="긍정 확률 점수 (0.0 ~ 1.0)")
    sentimentConf: float = Field(..., description="분석 결과 신뢰도")
    positiveCount: int = Field(..., description="긍정 리뷰 개수")
    negativeCount: int = Field(..., description="부정 리뷰 개수")
    aspectTags: List[str] = Field(..., description="주요 키워드 태그 리스트 (맛, 양, 위생 등)")
    evidencePhrases: List[str] = Field(..., description="판단 근거가 된 주요 문장들")
    issueFlags: bool = Field(..., description="특이사항(이슈) 발생 여부")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sentimentLabel": "POSITIVE",
                "sentimentScore": 0.85,
                "sentimentConf": 0.92,
                "positiveCount": 15,
                "negativeCount": 2,
                "aspectTags": ["맛", "영양"],
                "evidencePhrases": ["정말 맛있었어요", "최고의 식단"],
                "issueFlags": False
            }
        }
    )