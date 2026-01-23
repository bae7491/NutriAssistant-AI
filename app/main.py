from fastapi import FastAPI
import logging

from app.api.routes import health, menus
from app.services.food_loader import load_spring_and_build_context
from app.services.cost_loader import get_cost_db

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="NutriAssistant Menu Generator API", version="1.0.0")

app.include_router(health.router, tags=["Health"])
app.include_router(menus.router, tags=["Menus"])


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
