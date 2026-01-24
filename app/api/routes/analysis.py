from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import ReportAnalysisRequest, ReportAnalysisResponse, MenuWeight
from app.services.report_analyzer import ReportAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/report:analyze", response_model=ReportAnalysisResponse)
async def analyze_report(request: ReportAnalysisRequest):
    """리포트 분석 → 일회성 가중치 생성"""
    try:
        logger.info("📊 리포트 분석 요청 (일회성)")

        analyzer = ReportAnalyzer()
        weights_dict = await analyzer.analyze_report_to_weights(
            report_data=request.report_data, valid_menu_names=request.valid_menu_names
        )

        if not weights_dict:
            return ReportAnalysisResponse(weights=[], total_analyzed=0)

        weight_list = [
            MenuWeight(menu_name=menu, weight=weight, reason="리포트 분석 기반")
            for menu, weight in sorted(
                weights_dict.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return ReportAnalysisResponse(
            weights=weight_list, total_analyzed=len(weight_list)
        )

    except Exception as e:
        logger.error(f"❌ 리포트 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
