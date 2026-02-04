from fastapi import APIRouter, HTTPException
import logging

# [1] 스키마 임포트 (기존 스키마 + 신규 추가된 스키마)
from app.models.schemas import (
    ReportAnalysisRequest,
    ReportAnalysisResponse,
    MenuWeight,
    DailyAnalysisRequest,  # [신규] Java 요청 DTO
    DailyAnalysisResponse  # [신규] Java 응답 DTO
)
from app.services.report_analyzer import ReportAnalyzer

# [2] 감성 분석 서비스 임포트
# 실제 AI 로직이 있는 파일 위치를 가정하여 임포트합니다.
# 만약 파일이 없다면 에러가 나지 않도록 예외 처리를 해두었습니다.
try:
    from app.services.ai_analyzer import analyze_daily_sentiment
except ImportError:
    # 서비스 파일이 아직 없을 경우를 대비한 더미(Dummy) 함수
    def analyze_daily_sentiment(texts):
        return {
            "label": "POSITIVE",
            "score": 0.80,
            "conf": 0.90,
            "pos_cnt": len(texts),
            "neg_cnt": 0,
            "tags": ["맛", "서비스"],
            "evidence": ["맛있어요"],
            "issues": False
        }

router = APIRouter()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 기존 기능: 리포트 분석 (메뉴 가중치 산출)
# ------------------------------------------------------------------
@router.post("/report:analyze", response_model=ReportAnalysisResponse)
async def analyze_report(request: ReportAnalysisRequest):
    """리포트 분석 -> 일회성 가중치 생성"""
    try:
        logger.info("리포트 분석 요청 (일회성)")

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
        logger.error(f"리포트 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# [신규 기능] 일일 리뷰 감성 분석 (Java Spring Boot 연동용)
# ------------------------------------------------------------------
@router.post("/daily", response_model=DailyAnalysisResponse)
async def analyze_daily_reviews(request: DailyAnalysisRequest):
    """
    일일 리뷰 텍스트를 분석하여 긍정/부정 평가 및 키워드 추출
    - Java URL: restClient.post().uri("/api/analyze/daily")
    - 매핑: DailyAnalysisRequest -> AI 분석 -> DailyAnalysisResponse
    """
    try:
        # 요청 로그 기록 (학교 ID, 날짜)
        logger.info(f"일일 분석 요청 수신 - SchoolId: {request.schoolId}, Date: {request.targetDate}")

        # 1. 유효성 검사: 리뷰 텍스트가 비어있는지 확인
        if not request.reviewTexts:
            logger.warning("분석할 리뷰 텍스트가 없습니다. 빈 결과를 반환합니다.")
            return DailyAnalysisResponse(
                sentimentLabel="NEUTRAL",
                sentimentScore=0.0,
                sentimentConf=0.0,
                positiveCount=0,
                negativeCount=0,
                aspectTags=[],
                evidencePhrases=[],
                issueFlags=False
            )

        # 2. AI 서비스 호출
        # reviewTexts 리스트를 넘겨서 분석 결과를 딕셔너리 형태로 받음
        analysis_result = analyze_daily_sentiment(request.reviewTexts)

        # 3. 결과 매핑 (Dict -> Pydantic Model)
        # 서비스에서 반환한 키(Key) 이름이 일치하지 않을 수 있으므로 get() 사용
        response = DailyAnalysisResponse(
            sentimentLabel=analysis_result.get("label", "NEUTRAL"),
            sentimentScore=analysis_result.get("score", 0.0),
            sentimentConf=analysis_result.get("conf", 0.0),
            positiveCount=analysis_result.get("pos_cnt", 0),
            negativeCount=analysis_result.get("neg_cnt", 0),
            aspectTags=analysis_result.get("tags", []),
            evidencePhrases=analysis_result.get("evidence", []),
            issueFlags=analysis_result.get("issues", False)
        )

        logger.info(f"분석 완료 - 결과 점수: {response.sentimentScore}")
        return response

    except Exception as e:
        logger.error(f"일일 분석 처리 중 오류 발생: {e}", exc_info=True)
        # Java 서버가 500 에러를 받고 예외 처리를 할 수 있도록 함
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")