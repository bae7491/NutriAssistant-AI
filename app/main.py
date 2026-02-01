from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# 기존 라우터
from app.api.routes import health, menus, analysis

# ✅ 월간 리포트 라우터
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

# 기존 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(menus.router, tags=["Menus"])
app.include_router(analysis.router, prefix="/v1/analysis", tags=["Analysis"])

# ✅ 월간 리포트 라우터 (prefix="/api" 추가)
# monthly_ops.py의 "/reports/monthly" 앞에 "/api"가 붙어서
# 최종 경로: /api/reports/monthly ✅ (Spring Boot와 일치)
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
            "monthly_reports": "/api/reports/monthly"  # ✅ Spring Boot와 일치
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
    logger.info("   - POST /api/reports/monthly  ← 월간 운영 자료 생성")
    logger.info("   - GET  /api/reports/monthly  ← 월간 운영 자료 목록")
    logger.info("   - GET  /api/reports/monthly/{reportId}  ← 상세 조회")
    logger.info("   - GET  /api/test")
    logger.info("=" * 80)