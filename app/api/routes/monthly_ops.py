from fastapi import APIRouter, HTTPException, Header
from typing import Any, Dict
import logging

# 스키마 임포트
from app.models.report import MonthlyReportRequestPayload, MonthlyReport
from app.services.periodic_report import generate_periodic_report

router = APIRouter()
logger = logging.getLogger(__name__)


class MonthlyReportResponse:
    """월간 리포트 응답"""
    status: str
    message: str
    data: MonthlyReport


# ==============================================================================
# 월간 운영 자료 생성 (AI 분석 요청)
# POST /reports/monthly
# ==============================================================================
@router.post("/reports/monthly")
async def create_monthly_ops_doc(
        payload: MonthlyReportRequestPayload,
        authorization: str = Header(..., alias="Authorization")
):
    """
    월간 운영 자료 생성
    - Spring Boot에서 호출됨
    - AI 분석 결과만 반환 (실제 DB 저장은 Spring 담당)
    """
    try:
        logger.info(f"📊 월간 리포트 생성 요청: {payload.year}년 {payload.month}월")
        logger.info(f"   - dailyInfo: {len(payload.dailyInfo) if payload.dailyInfo else 0}건")
        logger.info(f"   - dailyAnalyses: {len(payload.dailyAnalyses) if payload.dailyAnalyses else 0}건")
        logger.info(f"   - reviews: {len(payload.reviews) if payload.reviews else 0}건")
        logger.info(f"   - mealPlan: {len(payload.mealPlan) if payload.mealPlan else 0}건")

        # AI 분석 실행
        report = generate_periodic_report(payload)

        logger.info(f"✅ 월간 리포트 분석 완료")

        # 응답 반환
        return {
            "status": "success",
            "message": "월간 운영 자료 분석이 완료되었습니다.",
            "data": report.model_dump()
        }

    except Exception as e:
        logger.error(f"❌ 월간 리포트 생성 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "서버 내부 오류가 발생했습니다."}
        )