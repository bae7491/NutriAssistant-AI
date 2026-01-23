from __future__ import annotations
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

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
)
from app.services.generator import generate_one_month
from app.services.food_loader import get_context
from app.services.ai_analyzer import AIAnalyzer

router = APIRouter()


# AIAnalyzer를 함수로 생성하여 매 요청마다 새로운 인스턴스 사용
def get_ai_analyzer():
    return AIAnalyzer()


@router.post("/v1/menus/month:generate", response_model=GenerateMonthResponse)
def generate_month(
    req: Optional[GenerateMonthRequest] = None,
    year: Optional[int] = Query(default=None, ge=2000, le=2100),
    month: Optional[int] = Query(default=None, ge=1, le=12),
    x_internal_token: str = Header(default=""),
):
    """월간 식단 생성"""
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        _ = get_context()
    except Exception:
        raise HTTPException(status_code=503, detail="dataset not ready (check /health)")

    if req is None:
        if year is None or month is None:
            raise HTTPException(
                status_code=422, detail="year/month required (body or query)"
            )
        req = GenerateMonthRequest(year=year, month=month, options=Options())

    meals, meta = generate_one_month(req.year, req.month, req.options)
    return GenerateMonthResponse(
        year=req.year,
        month=req.month,
        generatedAt=datetime.now().isoformat(),
        meals=meals,
        meta=meta,
    )


@router.post("/v1/facility/analyze", response_model=FacilityAnalysisResponse)
async def analyze_facility(request: FacilityAnalysisRequest):
    """
        시설 현황 텍스트를 AI로 분석하여 시설 가능 여부 반환

        Example:
    ```json
        POST /v1/facility/analyze
        {
            "facility_text": "오븐 고장, 튀김기 없음, 회전식 조리기 2대"
        }
    ```

        Response:
    ```json
        {
            "has_oven": false,
            "has_fryer": false,
            "has_griddle": true
        }
    ```
    """
    try:
        analyzer = get_ai_analyzer()
        result = await analyzer.analyze_facility_condition(request.facility_text)
        return FacilityAnalysisResponse(**result)
    except ValueError as e:
        # API 키 없음 등의 설정 오류
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시설 분석 실패: {str(e)}")


@router.post("/v1/reports/analyze", response_model=ReportAnalysisResponse)
async def analyze_report(request: ReportAnalysisRequest):
    """
        급식 리포트를 AI로 분석하여 메뉴별 가중치 생성

        Example:
    ```json
        POST /v1/reports/analyze
        {
            "report_data": {
                "best_menus": ["김치찌개", "돈까스"],
                "worst_menus": ["미역국"],
                "comments": ["맛있었어요", "별로였어요"]
            },
            "valid_menu_names": ["김치찌개", "돈까스", "미역국", "된장찌개"]
        }
    ```

        Response:
    ```json
        {
            "weights": [
                {
                    "menu_name": "김치찌개",
                    "weight": 0.8,
                    "reason": "학생들이 좋아하는 메뉴"
                },
                {
                    "menu_name": "돈까스",
                    "weight": 0.7,
                    "reason": "만족도 높음"
                },
                {
                    "menu_name": "미역국",
                    "weight": -0.6,
                    "reason": "불만 의견 많음"
                }
            ],
            "total_analyzed": 3
        }
    ```
    """
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
        # API 키 없음 등의 설정 오류
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 분석 실패: {str(e)}")
