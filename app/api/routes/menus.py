from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException, Query, Body
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from app.models.schemas import MonthMenuRequest, GenerateMonthResponse
from app.services.generator import generate_one_month

from app.core.config import INTERNAL_TOKEN
from app.models.schemas import (
    GenerateMonthRequest,
    GenerateMonthResponse,
    Options,
    FacilityAnalysisRequest,
    FacilityAnalysisResponse,
    ReportAnalysisRequest,
    ReportAnalysisResponse,
    MenuWeight,
    NewMenuGenerationRequest,
    NewMenuGenerationResponse,
)

# [확인] 여기가 '주방장'을 불러오는 부분입니다.
# generate_one_month: 월간 식단 생성 함수 (generator.py)
# generate_single_candidate: 1끼 생성 함수 (generator.py) -> Java에서 AI 대체 요청 시 사용
from app.services.generator import generate_one_month, generate_single_candidate
from app.services.food_loader import get_context
from app.services.ai_analyzer import AIAnalyzer
from app.services.new_menu_service import NewMenuService

router = APIRouter()
logger = logging.getLogger(__name__)


# AIAnalyzer를 함수로 생성하여 매 요청마다 새로운 인스턴스 사용
def get_ai_analyzer():
    return AIAnalyzer()


# ==============================================================================
# 1. 월간 식단 생성 API
# 요청: POST /v1/menus/month:generate
# 역할: Java가 "3월 식단 짜줘"라고 하면 generator.py의 generate_one_month를 호출
# ==============================================================================
@router.post("/month/generate", response_model=GenerateMonthResponse)
async def generate_monthly_menu(request: MonthMenuRequest):
    """월간 식단 생성"""
    try:
        logger.info("=" * 60)
        logger.info(f"📅 식단 생성 요청: {request.year}년 {request.month}월")
        logger.info("=" * 60)

        # ✅ 리포트 유무 확인
        if request.report:
            logger.info("   리포트 포함 (AI 가중치 분석 예정)")
        else:
            logger.info("   리포트 없음 (기본 가중치 사용)")

        # ✅ 신메뉴 유무 확인
        if request.new_menus:
            logger.info(f"   신메뉴 포함: {len(request.new_menus)}개")
        else:
            logger.info("   신메뉴 없음 (기존 DB만 사용)")

        # ✅ 영양 기준 확인
        if request.nutrition_key:
            logger.info(f"   영양 기준: {request.nutrition_key}")
        else:
            logger.info("   영양 기준 없음 (기본값 고등_남 사용)")

        # ✅ 리포트, 신메뉴, 영양 기준을 generator에 전달
        meals, meta = await generate_one_month(
            request.year,
            request.month,
            request.options or Options(),
            request.report,  # ← 리포트 전달
            request.new_menus,  # ← 신메뉴 전달
            request.nutrition_key,  # ← 영양 기준 전달
        )

        logger.info(f"✅ 식단 생성 완료: {len(meals)}개")
        logger.info("=" * 60)

        return GenerateMonthResponse(
            year=request.year,
            month=request.month,
            generatedAt=datetime.now().isoformat(),
            meals=meals,
            meta=meta,
        )

    except Exception as e:
        logger.error(f"❌ 식단 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 2. [신규 기능] 단일 식단(1끼) AI 생성 API
# 요청: POST /v1/menus/single:generate
# 역할: Java에서 "이 날 점심 메뉴만 AI로 바꿔줘"라고 할 때 호출됨
# ==============================================================================
@router.post("/v1/menus/single:generate")
def generate_single_meal(
    req: Dict[str, Any] = Body(...),
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
):
    """
    단일 식단(1끼) AI 생성
    Java 요청 Body 예시: { "date": "2026-03-03", "meal_type": "중식" }
    """
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    meal_type = req.get("meal_type", "중식")

    # generator.py의 generate_single_candidate 함수를 호출하여 결과 반환
    result = generate_single_candidate(meal_type)
    return result


# ==============================================================================
# 3. 시설 현황 분석 API
# ==============================================================================
@router.post("/v1/facility/analyze", response_model=FacilityAnalysisResponse)
async def analyze_facility(
    request: FacilityAnalysisRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
):
    """시설 현황 텍스트 분석"""
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        analyzer = get_ai_analyzer()
        result = await analyzer.analyze_facility_condition(request.facility_text)
        return FacilityAnalysisResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시설 분석 실패: {str(e)}")


# ==============================================================================
# 4. 급식 리포트 분석 API
# ==============================================================================
@router.post("/v1/reports/analyze", response_model=ReportAnalysisResponse)
async def analyze_report(
    request: ReportAnalysisRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
):
    """급식 리포트 분석"""
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        analyzer = get_ai_analyzer()
        valid_names = set(request.valid_menu_names)
        weights_dict = await analyzer.analyze_reviews_and_generate_weights(
            request.report_data, valid_names
        )

        weights_list = [
            MenuWeight(menu_name=k, weight=v) for k, v in weights_dict.items()
        ]

        return ReportAnalysisResponse(
            weights=weights_list, total_analyzed=len(weights_list)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 분석 실패: {str(e)}")


# ==============================================================================
# 5. 신메뉴 생성 API
# 요청: POST /v1/menus/new-menu:generate
# 역할: 트렌드 분석과 게시판 피드백을 기반으로 신메뉴 추천/생성
# ==============================================================================
@router.post("/v1/menus/new-menu:generate", response_model=NewMenuGenerationResponse)
async def generate_new_menu(
    request: NewMenuGenerationRequest,
    x_internal_token: str = Header(default="", alias="X-Internal-Token"),
):
    """
    신메뉴 생성

    게시판 피드백과 네이버 트렌드 분석을 통해 신메뉴를 추천합니다.
    """
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        logger.info("=" * 60)
        logger.info("🍽️ 신메뉴 생성 API 요청")
        logger.info(f"   - use_trend: {request.use_trend}")
        logger.info(f"   - use_board: {request.use_board}")
        logger.info(f"   - trend_days: {request.trend_days}")
        logger.info(f"   - count: {request.count}")
        logger.info("=" * 60)

        service = NewMenuService()
        result = await service.generate_new_menus(
            use_trend=request.use_trend,
            use_board=request.use_board,
            trend_days=request.trend_days,
            count=request.count,
        )

        logger.info(f"✅ 신메뉴 생성 완료: {len(result.new_menus)}개")
        return result

    except ValueError as e:
        logger.error(f"❌ 신메뉴 생성 실패 (ValueError): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 신메뉴 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"신메뉴 생성 실패: {str(e)}")
