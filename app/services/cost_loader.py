from typing import Dict, Optional
import requests
import logging
from datetime import datetime

from app.core.config import SPRING_FOOD_API, INTERNAL_TOKEN, SPRING_TIMEOUT_SECONDS
from app.services.cost_generator import CostGenerator

logger = logging.getLogger(__name__)

# 전역 캐시
_COST_CACHE: Optional[Dict[str, int]] = None
_cost_generator: Optional[CostGenerator] = None


def get_cost_generator() -> CostGenerator:
    """CostGenerator 싱글톤"""
    global _cost_generator
    if _cost_generator is None:
        _cost_generator = CostGenerator()
    return _cost_generator


def get_cost_db() -> Dict[str, int]:
    """
    Spring에서 단가 DB 가져오기
    실패 시 AI로 자동 생성 후 Spring DB에 저장

    Returns:
        {메뉴명: 단가(원)} 딕셔너리
    """
    global _COST_CACHE

    # ✅ 이미 로드했으면 캐시 반환 (로그 출력 안 함)
    if _COST_CACHE is not None:
        # logger.info(f"✅ 캐시된 단가 DB 사용: {len(_COST_CACHE)}개")  # ❌ 삭제
        return _COST_CACHE

    logger.info("=" * 60)
    logger.info("💰 단가 DB 로딩 프로세스 시작")
    logger.info("=" * 60)

    try:
        # 1. Spring API 호출
        base_url = SPRING_FOOD_API.rsplit("/", 2)[0]
        cost_url = f"{base_url}/api/costs"

        headers = {}
        if INTERNAL_TOKEN:
            headers["X-INTERNAL-TOKEN"] = INTERNAL_TOKEN

        logger.info(f"🔄 Spring 단가 DB 확인: {cost_url}")

        try:
            response = requests.get(
                cost_url, headers=headers, timeout=SPRING_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            data = response.json()
            prices = data.get("prices", {})
            year = data.get("year", datetime.now().year)

            # 2. 단가 DB가 있으면 사용
            if prices and len(prices) > 0:
                _COST_CACHE = prices
                logger.info(
                    f"✅ Spring DB에서 단가 로드 완료: {len(prices)}개 (연도: {year}년)"
                )
                logger.info("=" * 60)
                return _COST_CACHE

            # 3. 단가 DB가 비어있으면 AI 생성
            logger.warning("⚠️ Spring 단가 DB가 비어있습니다")
            return _generate_and_save_costs(base_url, headers)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Spring API 호출 실패: {e}")
            logger.info("🤖 AI로 단가 생성을 시도합니다...")
            return _generate_and_save_costs(base_url, headers)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ 단가 DB 처리 중 오류 발생: {e}")
        logger.error("   기본값 1000원으로 폴백합니다")
        logger.error("=" * 60)
        _COST_CACHE = {}
        return _COST_CACHE


def _generate_and_save_costs(base_url: str, headers: dict) -> Dict[str, int]:
    """
    AI로 단가 생성 후 Spring DB에 저장

    Args:
        base_url: Spring API 기본 URL
        headers: 요청 헤더

    Returns:
        생성된 단가 딕셔너리
    """
    global _COST_CACHE

    logger.info("🤖 AI로 단가 자동 생성 시작...")
    logger.info("   - 예상 소요 시간: 5~10분")
    logger.info("=" * 60)

    try:
        # 1. 음식 DB에서 모든 메뉴명 가져오기
        from app.services.food_loader import get_context

        ctx = get_context()

        all_menu_names = set()

        # 각 역할별 풀에서 메뉴명 수집
        for pool in ctx.pools.values():
            menu_names = pool["menuName"].dropna().unique().tolist()
            all_menu_names.update(menu_names)

        # 디저트 풀도 추가
        all_menu_names.update(ctx.dessert_pool)

        menu_list = sorted(list(all_menu_names))
        logger.info(f"📋 분석 대상: {len(menu_list)}개 메뉴")

        if not menu_list:
            logger.error("❌ 메뉴 목록이 비어있습니다")
            _COST_CACHE = {}
            return _COST_CACHE

        # 2. AI로 2023년 기준 단가 생성
        generator = get_cost_generator()
        current_year = datetime.now().year

        logger.info(f"🔄 AI 단가 생성 중... ({len(menu_list)}개)")
        base_costs = generator.generate_costs_batch(menu_list)

        if not base_costs:
            logger.error("❌ AI 단가 생성 실패")
            _COST_CACHE = {}
            return _COST_CACHE

        # 3. 현재 연도로 물가상승률 적용
        logger.info(f"📈 물가상승률 적용: 2023년 → {current_year}년")
        inflated_costs = generator.apply_inflation(base_costs, current_year)

        # 4. Spring으로 전송하여 DB에 저장
        logger.info(f"💾 Spring DB에 저장 시도...")
        upload_url = f"{base_url}/api/costs/bulk"

        upload_data = {"year": current_year, "prices": inflated_costs}

        try:
            upload_response = requests.post(
                upload_url,
                json=upload_data,
                headers=headers,
                timeout=SPRING_TIMEOUT_SECONDS * 3,  # 타임아웃 3배 연장
            )
            upload_response.raise_for_status()

            logger.info("=" * 60)
            logger.info(
                f"✅ AI 생성 단가를 Spring DB에 저장 완료: {len(inflated_costs)}개"
            )
            logger.info(f"   - 기준 연도: 2023년")
            logger.info(f"   - 현재 연도: {current_year}년")
            logger.info(
                f"   - 물가 계수: {generator.calculate_inflation_multiplier(current_year):.4f}배"
            )
            logger.info("=" * 60)

        except Exception as save_error:
            logger.error(f"⚠️ Spring DB 저장 실패: {save_error}")
            logger.info("   하지만 메모리에서 단가를 사용합니다")

        # 5. 메모리 캐싱 (Spring 저장 실패해도 사용 가능)
        _COST_CACHE = inflated_costs
        return _COST_CACHE

    except Exception as e:
        logger.error(f"❌ AI 단가 생성 실패: {e}")
        logger.error("   기본값 1000원으로 폴백합니다")
        _COST_CACHE = {}
        return _COST_CACHE


def get_menu_cost(menu_name: str, default: int = 1000) -> int:
    """
    특정 메뉴의 단가 조회

    Args:
        menu_name: 메뉴명
        default: 단가가 없을 때 기본값

    Returns:
        단가(원)
    """
    cost_db = get_cost_db()  # ✅ 캐시되어 있으면 로그 없이 즉시 반환
    clean_name = menu_name.strip()

    # 알레르기 정보 제거 (예: "쌀밥 (5)" -> "쌀밥")
    if "(" in clean_name:
        clean_name = clean_name.split("(")[0].strip()

    return cost_db.get(clean_name, default)


def reload_cost_db() -> Dict[str, int]:
    """
    단가 DB 강제 재로드

    Returns:
        {메뉴명: 단가} 딕셔너리
    """
    global _COST_CACHE
    _COST_CACHE = None
    logger.info("🔄 단가 DB 강제 재로드 시작...")
    return get_cost_db()


def get_cost_stats() -> Dict:
    """
    단가 DB 통계 조회

    Returns:
        단가 DB 통계 정보
    """
    cost_db = get_cost_db()

    if not cost_db:
        return {
            "status": "empty",
            "total_menus": 0,
            "message": "단가 DB가 비어있습니다 (기본값 1000원 사용 중)",
        }

    prices = list(cost_db.values())

    return {
        "status": "loaded",
        "total_menus": len(cost_db),
        "price_range": {
            "min": min(prices),
            "max": max(prices),
            "avg": int(sum(prices) / len(prices)),
        },
        "sample_costs": dict(list(cost_db.items())[:5]),
    }


def is_cost_db_loaded() -> bool:
    """
    단가 DB 로드 여부 확인

    Returns:
        로드되었으면 True, 아니면 False
    """
    global _COST_CACHE
    return _COST_CACHE is not None and len(_COST_CACHE) > 0
