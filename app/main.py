from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# ✅ [추가] 데이터 모델 정의를 위한 라이브러리 임포트
from pydantic import BaseModel
from typing import List

# 기존 라우터
from app.api.routes import health, menus, analysis
# 월간 리포트 라우터
from app.api.routes import monthly_ops

from app.services.food_loader import load_spring_and_build_context
from app.services.cost_loader import get_cost_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NutriAssistant Menu Generator API",
    version="2.0.0",
    description="급식 식단 생성 및 월간 리포트 분석 시스템"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# ✅ [추가] 1. 데이터 모델 정의 (Spring Boot와 통신할 규격)
# =================================================================

class AnalysisRequest(BaseModel):
    schoolId: int
    targetDate: str
    reviewTexts: List[str]  # Spring Boot에서 보내줄 리뷰 텍스트 리스트

class AnalysisResponse(BaseModel):
    sentimentLabel: str
    sentimentScore: float
    sentimentConf: float
    positiveCount: int      # 긍정 개수
    negativeCount: int      # 부정 개수
    aspectTags: List[str]
    evidencePhrases: List[str]
    issueFlags: bool

# =================================================================
# ✅ [추가] 2. 일일 분석 API 엔드포인트 (Spring Boot 스케줄러가 호출)
# =================================================================

@app.post("/api/analyze/daily", response_model=AnalysisResponse, tags=["Daily Analysis"])
async def analyze_daily_reviews(request: AnalysisRequest):
    reviews = request.reviewTexts

    # 리뷰 데이터가 없는 경우 기본값 반환
    if not reviews:
        return AnalysisResponse(
            sentimentLabel="NEUTRAL",
            sentimentScore=0.0,
            sentimentConf=0.0,
            positiveCount=0,
            negativeCount=0,
            aspectTags=[],
            evidencePhrases=[],
            issueFlags=False
        )

    # --- [분석 로직 시작] ---
    positive_cnt = 0
    negative_cnt = 0
    total_score = 0.0

    # (테스트용) 긍정 키워드 - 실제 AI 모델 적용 시 이 부분을 모델 추론 코드로 교체하세요.
    pos_keywords = ["맛있", "좋", "최고", "굿", "good", "yummy", "사랑", "추천"]

    for text in reviews:
        is_positive = False

        # 키워드가 포함되어 있으면 긍정으로 판단 (임시 로직)
        if any(keyword in text for keyword in pos_keywords):
            is_positive = True
            score = 0.9
        else:
            is_positive = False
            score = 0.2

        total_score += score

        # 개수 카운팅
        if is_positive:
            positive_cnt += 1
        else:
            negative_cnt += 1
    # --- [분석 로직 끝] ---

    # 평균 점수 계산
    avg_score = total_score / len(reviews) if reviews else 0.0
    final_label = "POSITIVE" if positive_cnt >= negative_cnt else "NEGATIVE"

    logger.info(f"일일 분석 완료 - SchoolID: {request.schoolId}, 긍정: {positive_cnt}, 부정: {negative_cnt}")

    return AnalysisResponse(
        sentimentLabel=final_label,
        sentimentScore=round(avg_score, 2),
        sentimentConf=0.95,
        positiveCount=positive_cnt,
        negativeCount=negative_cnt,
        aspectTags=["맛", "양", "위생"],
        evidencePhrases=["맛있어요", "양이 적어요"],
        issueFlags=False
    )

# =================================================================
# 기존 라우터 등록
# =================================================================
app.include_router(health.router, tags=["Health"])
app.include_router(menus.router, tags=["Menus"])
app.include_router(analysis.router, prefix="/v1/analysis", tags=["Analysis"])
app.include_router(monthly_ops.router, prefix="/api", tags=["Monthly Reports"])


@app.get("/")
def root():
    """API 루트 엔드포인트"""
    return {
        "name": "NutriAssistant Menu Generator API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "menus": "/month/generate",
            "analysis": "/v1/analysis",
            "daily_analysis": "/api/analyze/daily", # ✅ 목록에 추가됨
            "monthly_reports": "/api/reports/monthly"
        },
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    logger.info("=" * 80)
    logger.info("🚀 FastAPI 서버 시작")
    logger.info("=" * 80)

    # 1. 음식 DB 로드
    try:
        load_spring_and_build_context()
        logger.info("✅ 음식 DB 로드 완료")
    except Exception as e:
        logger.error(f"❌ 음식 DB 로드 실패: {e}")

    # 2. 단가 DB 사전 로드
    try:
        logger.info("🔄 단가 DB 사전 로드 시작...")
        cost_db = get_cost_db()
        logger.info(f"✅ 단가 DB 로드 완료: {len(cost_db)}개")
    except Exception as e:
        logger.error(f"❌ 단가 DB 로드 실패: {e}")

    logger.info("=" * 80)
    logger.info("📋 등록된 API 엔드포인트:")
    logger.info("   - GET  /health")
    logger.info("   - POST /month/generate")
    logger.info("   - POST /v1/analysis/report:analyze")
    logger.info("   - POST /api/reports/monthly")
    logger.info("   - POST /api/analyze/daily    ← ✅ [NEW] 일일 감성 분석")
    logger.info("=" * 80)