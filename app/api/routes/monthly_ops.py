from fastapi import APIRouter, HTTPException, Header
import logging

# 스키마 임포트
from app.models.schemas import (
    MonthlyOpsDocCreateRequest,
    MonthlyOpsDocCreateResponse
)
from app.models.report import MonthlyReportRequestPayload
from app.services.periodic_report import generate_periodic_report

router = APIRouter()
logger = logging.getLogger(__name__)

# ==============================================================================
# 월간 운영 자료 생성 (AI 분석 요청)
# POST /reports/monthly
# ==============================================================================
@router.post("/reports/monthly", response_model=MonthlyOpsDocCreateResponse)
async def create_monthly_ops_doc(
        payload: MonthlyOpsDocCreateRequest,
        authorization: str = Header(..., alias="Authorization")
):
    """
    월간 운영 자료 생성
    - Spring Boot에서 호출됨
    - AI 분석 결과만 반환 (실제 DB 저장은 Spring 담당)
    """
    try:
        logger.info(f"📊 월간 리포트 생성 요청: {payload.year}년 {payload.month}월 (School ID: {payload.school_id})")

        # 1. 서비스 로직용 페이로드 변환
        analysis_payload = MonthlyReportRequestPayload(
            year=payload.year,
            month=payload.month,
            school_id=payload.school_id,
            userName="Administrator"  # ✅ 수정: user_name → userName
        )

        # 2. AI 분석 실행 (오래 걸릴 수 있음)
        report = generate_periodic_report(analysis_payload)

        logger.info(f"✅ 월간 리포트 분석 완료")

        # 3. 응답 반환
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