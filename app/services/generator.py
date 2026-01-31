from __future__ import annotations
import calendar, json, os, random, time, logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import time, random, re
import numpy as np
import pygad

from app.core.config import (
    COST_DB_PATH,
    WEIGHT_DB_PATH,
    ROLE_ORDER,
    DESSERT_FREQUENCY_PER_WEEK,
    STD_KCAL,
    STD_PROT,
    KCAL_TOLERANCE_RATIO,
    get_nutrition_standard,
)
from app.models.schemas import Options, NewMenuInput
from app.services.food_loader import get_context, build_context_with_new_menus, FoodContext
from app.utils.holidays import get_holidays
from app.services.cost_loader import get_menu_cost, get_cost_db
from app.services.ai_analyzer import AIAnalyzer
from app.services.report_analyzer import ReportAnalyzer
from app.services.food_loader import get_valid_menu_names

logger = logging.getLogger(__name__)


def _normalize_allergy(alg_val: str) -> Optional[str]:
    """알레르기 정보 정규화"""
    alg_val = str(alg_val).strip()
    if not alg_val or alg_val.lower() == "nan" or alg_val == "0":
        return None
    cleaned = alg_val.replace(".0", "")
    parts: List[int] = []
    for p in cleaned.replace(",", " ").split():
        if p.isdigit():
            parts.append(int(p))
    if not parts:
        return None
    return ",".join(map(str, sorted(set(parts))))


def _load_json_dict(path: str, outer_key: Optional[str] = None) -> Dict[str, Any]:
    """JSON 파일 로드"""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if outer_key and isinstance(data, dict):
            v = data.get(outer_key)
            return v if isinstance(v, dict) else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def generate_one_month(
    year: int,
    month: int,
    opt: Options,
    report_data: Optional[Dict] = None,
    new_menus: Optional[List[NewMenuInput]] = None,
    nutrition_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    월간 식단 생성

    Args:
        year: 연도
        month: 월
        opt: 옵션
        report_data: 리포트 JSON (Spring이 DB에서 조회하여 전달)
        new_menus: 신메뉴 목록 (Spring에서 전달, 기존 음식 DB와 함께 사용)
        nutrition_key: 영양 기준 키 (ELEMENTARY, MIDDLE_MALE, etc.)

    Returns:
        (식단 리스트, 메타데이터)
    """
    # 신메뉴가 있으면 병합된 컨텍스트 사용
    if new_menus:
        new_menus_dict = [m.model_dump() for m in new_menus]
        ctx = build_context_with_new_menus(new_menus_dict)
    else:
        ctx = get_context()

    constraints = opt.constraints

    # ========================================
    # 영양 기준 설정 (nutrition_key 기반)
    # constraints.nutrition_key 우선, 없으면 파라미터 nutrition_key 사용
    # ========================================
    effective_nutrition_key = constraints.nutrition_key or nutrition_key
    nutrition_std = get_nutrition_standard(effective_nutrition_key)
    std_kcal = float(nutrition_std["kcal"])
    std_prot = float(nutrition_std["prot"])

    # 칼로리 허용 범위 계산
    min_kcal_limit = int(std_kcal * (1.0 - KCAL_TOLERANCE_RATIO))
    max_kcal_limit = int(std_kcal * (1.0 + KCAL_TOLERANCE_RATIO))

    # 탄수화물 범위 계산 (55~65%)
    min_carb_g = (std_kcal * 0.55) / 4
    max_carb_g = (std_kcal * 0.65) / 4

    logger.info("=" * 60)
    logger.info(f"🎯 영양 기준 설정: [{effective_nutrition_key or 'DEFAULT(고등_남)'}]")
    logger.info(f"   - 목표 에너지: {std_kcal}kcal")
    logger.info(f"   - 목표 단백질: {std_prot}g")
    logger.info(f"   - 허용 칼로리 범위: {min_kcal_limit} ~ {max_kcal_limit} kcal")
    logger.info(f"   - 탄수화물 범위: {int(min_carb_g)}g ~ {int(max_carb_g)}g")
    logger.info("=" * 60)

    # ========================================
    # 1. 제약사항 처리
    # ========================================
    # ✅ 원본 제약사항 로깅
    logger.info("=" * 60)
    logger.info("📥 받은 제약사항 (원본)")
    logger.info("=" * 60)
    logger.info(f"   target_price: {constraints.target_price}")
    logger.info(f"   cost_tolerance: {constraints.cost_tolerance}")
    logger.info(f"   max_price_limit: {constraints.max_price_limit}")
    logger.info(f"   cook_staff: {constraints.cook_staff}")
    logger.info(f"   facility_text: {constraints.facility_text}")
    logger.info(f"   facility_flags (원본):")
    logger.info(f"      - has_oven: {constraints.facility_flags.has_oven}")
    logger.info(f"      - has_fryer: {constraints.facility_flags.has_fryer}")
    logger.info(f"      - has_griddle: {constraints.facility_flags.has_griddle}")
    logger.info("=" * 60)

    # 시설 현황 텍스트가 있으면 AI 분석
    if constraints.facility_text:
        text = constraints.facility_text.strip().lower()

        # 의미 없는 값 필터링
        if text and text not in ["string", "null", "none", "undefined", ""]:
            logger.info("🤖 시설 현황 AI 분석 중...")
            logger.info(f"   입력 텍스트: '{constraints.facility_text}'")

            try:
                analyzer = AIAnalyzer()
                analyzed_flags = await analyzer.analyze_facility_condition(
                    constraints.facility_text
                )

                logger.info(f"   AI 분석 결과: {analyzed_flags}")

                # AI 결과를 constraints에 반영하기 전에 "병합된 최종값"을 만든다
                old_flags = {
                    "has_oven": constraints.facility_flags.has_oven,
                    "has_fryer": constraints.facility_flags.has_fryer,
                    "has_griddle": constraints.facility_flags.has_griddle,
                }

                # analyzed_flags에서 None이 들어오거나 키가 없을 수 있으니 안전하게 정리
                def pick_bool(key: str, default: bool) -> bool:
                    v = analyzed_flags.get(key, None)
                    if v is None:
                        return default
                    return bool(v)

                new_flags = {
                    "has_oven": pick_bool("has_oven", old_flags["has_oven"]),
                    "has_fryer": pick_bool("has_fryer", old_flags["has_fryer"]),
                    "has_griddle": pick_bool("has_griddle", old_flags["has_griddle"]),
                }

                # ✅ 변경 전후 로그 (None 절대 안 뜸)
                logger.info("   변경 사항:")
                logger.info(
                    f"      - has_oven: {old_flags['has_oven']} → {new_flags['has_oven']}"
                )
                logger.info(
                    f"      - has_fryer: {old_flags['has_fryer']} → {new_flags['has_fryer']}"
                )
                logger.info(
                    f"      - has_griddle: {old_flags['has_griddle']} → {new_flags['has_griddle']}"
                )

                # ✅ 최종 반영 (None 절대 안 들어감)
                constraints.facility_flags.has_oven = new_flags["has_oven"]
                constraints.facility_flags.has_fryer = new_flags["has_fryer"]
                constraints.facility_flags.has_griddle = new_flags["has_griddle"]

                logger.info("✅ AI 분석 완료 및 적용")

            except Exception as e:
                logger.error(f"❌ AI 분석 실패: {e}", exc_info=True)
                logger.warning("   기본값(facility_flags 유지)으로 설정")
        else:
            logger.info(f"⚠️ facility_text가 의미 없는 값입니다: '{text}'")
            logger.info("   facility_flags 직접 사용")
    else:
        logger.info("ℹ️ facility_text 없음. facility_flags 직접 사용")

    # 최종 제약사항 로깅
    logger.info("=" * 60)
    logger.info("📋 최종 적용 제약사항")
    logger.info("=" * 60)

    # 단가 관련 상세 로깅
    target_price = constraints.target_price
    tolerance = constraints.cost_tolerance
    min_price = int(target_price * (1 - tolerance))
    max_price = int(target_price * (1 + tolerance))

    logger.info(f"   💰 단가 제약:")
    logger.info(f"      - 목표 단가: {target_price:,}원")
    logger.info(f"      - 허용 오차: ±{tolerance*100:.0f}%")
    logger.info(f"      - 허용 범위: {min_price:,}원 ~ {max_price:,}원")
    logger.info(f"      - 최대 상한 (절대): {constraints.max_price_limit:,}원")
    logger.info(f"   👨‍🍳 조리 인원: {constraints.cook_staff}명")
    logger.info(f"   🔧 시설 현황 (facility_text: '{constraints.facility_text}'):")
    logger.info(
        f"      - 오븐: {'✅ 사용 가능' if constraints.facility_flags.has_oven else '❌ 사용 불가 → 오븐구이/피자/그라탕 등 제외'}"
    )
    logger.info(
        f"      - 튀김기: {'✅ 사용 가능' if constraints.facility_flags.has_fryer else '❌ 사용 불가 → 튀김/돈까스/치킨 등 제외'}"
    )
    logger.info(
        f"      - 철판: {'✅ 사용 가능' if constraints.facility_flags.has_griddle else '❌ 사용 불가 → 전/부침개/철판볶음 등 제외'}"
    )
    logger.info("=" * 60)

    # ========================================
    # 2. 가중치 처리 (리포트 분석)
    # ========================================
    weights: Dict[str, float] = {}

    if report_data:
        logger.info("=" * 60)
        logger.info("📊 리포트 기반 가중치 분석 시작")
        logger.info("=" * 60)

        try:
            # 유효 메뉴명 조회
            valid_menu_names = get_valid_menu_names()
            logger.info(f"   유효 메뉴: {len(valid_menu_names)}개")

            # AI 분석
            analyzer = ReportAnalyzer()
            weights = await analyzer.analyze_report_to_weights(
                report_data=report_data, valid_menu_names=valid_menu_names
            )

            if weights:
                logger.info(f"✅ 가중치 생성 완료: {len(weights)}개 메뉴")
            else:
                logger.warning("⚠️ 가중치 생성 실패, 기본값 사용")

        except Exception as e:
            logger.error(f"❌ 리포트 분석 실패: {e}", exc_info=True)
            logger.warning("   가중치 없이 진행")
            weights = {}
    else:
        logger.info("ℹ️ 리포트 없음. 가중치 미사용")

    logger.info("=" * 60)

    # ========================================
    # 3. 단가 DB 로드
    # ========================================
    logger.info("=" * 60)
    logger.info("💰 단가 DB 로딩 중...")
    logger.info("=" * 60)

    try:
        cost_db = get_cost_db()

        if cost_db and len(cost_db) > 0:
            logger.info(f"✅ 단가 DB 로드 완료: {len(cost_db)}개 메뉴")
        else:
            logger.warning("⚠️ 단가 DB가 비어있습니다. 기본값 1000원 사용")

    except Exception as e:
        logger.error(f"❌ 단가 DB 로드 실패: {e}")
        logger.warning("⚠️ 기본값 1000원으로 식단 생성을 계속합니다")
        cost_db = {}

    logger.info("=" * 60)

    # ========================================
    # 4. 초기화
    # ========================================
    global_day_count = 0
    global_menu_tracker: Dict[str, Tuple[int, int, int]] = {}
    current_month_counts: Dict[str, int] = {}

    # 주간 중복 방지용 트래커
    current_week_menus: Dict[str, int] = {}  # 메뉴명 → 해당 주 사용 횟수
    current_week_number = 0

    holidays = get_holidays(year)
    last_day = calendar.monthrange(year, month)[1]

    # 디저트 주 2회 랜덤 배정 (평일 수에 비례)
    weekdays_by_week: Dict[int, List[int]] = {}
    for d in range(1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() >= 5 or dt.date() in holidays:
            continue
        wk = dt.isocalendar()[1]
        weekdays_by_week.setdefault(wk, []).append(d)

    lunch_dessert_days: set[int] = set()
    dinner_dessert_days: set[int] = set()

    # 기준: 5일 기준 DESSERT_FREQUENCY_PER_WEEK(2)회 → 40% 비율
    FULL_WEEK_DAYS = 5
    dessert_ratio = DESSERT_FREQUENCY_PER_WEEK / FULL_WEEK_DAYS  # 0.4

    for days in weekdays_by_week.values():
        num_days = len(days)
        if num_days == 0:
            continue

        # 평일 수에 비례한 디저트 횟수 계산
        # 5일 → 2회, 4일 → 1~2회, 3일 → 1회, 2일 → 1회, 1일 → 0회
        proportional_count = num_days * dessert_ratio
        k = int(round(proportional_count))

        # 최소 0회, 최대 평일 수
        k = max(0, min(k, num_days))

        if k > 0:
            lunch_dessert_days.update(random.sample(days, k))
            dinner_dessert_days.update(random.sample(days, k))

    logger.info(f"🍰 디저트 배정: 중식 {len(lunch_dessert_days)}일, 석식 {len(dinner_dessert_days)}일")

    ga_params = dict(
        num_generations=opt.numGenerations,
        sol_per_pop=opt.solPerPop,
        num_parents_mating=opt.numParentsMating,
        keep_parents=opt.keepParents,
        mutation_percent_genes=opt.mutationPercentGenes,
        stop_criteria=None,
    )

    logger.info(f"🧬 GA 파라미터 설정 완료")
    logger.info(f"   세대 수: {opt.numGenerations}")
    logger.info(f"   인구 크기: {opt.solPerPop}")

    current_meal_type = "중식"
    today_lunch_menus: List[str] = []
    current_day_for_fitness = 0  # fitness 함수에서 사용할 현재 날짜

    # ========================================
    # 5. Fitness 함수
    # ========================================
    def fitness_func(ga_instance, solution, solution_idx):
        nonlocal global_day_count, current_meal_type, today_lunch_menus, current_week_menus

        indices = solution.astype(int)
        display_names = [
            str(ctx.pool_display_names[role][idx])
            for role, idx in zip(ROLE_ORDER, indices)
        ]
        cats = [str(ctx.pool_cats[role][idx]) for role, idx in zip(ROLE_ORDER, indices)]
        nutr_values = np.array(
            [ctx.pool_matrices[role][idx] for role, idx in zip(ROLE_ORDER, indices)]
        )
        totals = nutr_values.sum(axis=0)
        t_kcal = float(totals[0])
        t_prot = float(totals[2])

        score = 1_000_000.0
        penalty = 0.0

        # 영양소 평가 (동적 영양 기준 사용)
        if (std_kcal * 0.9) <= t_kcal <= (std_kcal * 1.1):
            score += 200_000
        else:
            penalty += 100_000 + abs(t_kcal - std_kcal) * 200

        if t_prot < std_prot:
            penalty += (std_prot - t_prot) * 20_000

        # ========================================
        # 중복 방지 (강화)
        # ========================================

        # 1) 같은 끼니 내 주찬1/주찬2 중복 방지
        if display_names[2] == display_names[3]:
            penalty += 2_000_000
        if cats[2] == cats[3]:
            penalty += 1_000_000

        # 2) 같은 날 점심/저녁 중복 방지 (국, 주찬, 부찬 전체 체크)
        if current_meal_type == "석식" and today_lunch_menus:
            # 국(1), 주찬1(2), 주찬2(3), 부찬(4) 체크
            curr_main_menus = {display_names[i] for i in [1, 2, 3, 4]}
            overlap_count = len(curr_main_menus & set(today_lunch_menus))
            if overlap_count > 0:
                penalty += overlap_count * 2_000_000  # 겹치는 메뉴당 페널티

        # 3) 같은 주간 내 중복 방지 (쌀밥, 김치 제외)
        for i, name in enumerate(display_names):
            nm = name.strip()

            # 쌀밥/흰밥, 배추김치는 중복 허용
            if "쌀밥" in nm or "흰밥" in nm or "배추김치" in nm:
                continue

            week_count = current_week_menus.get(nm, 0)
            if week_count >= 1:
                # 같은 주에 이미 사용된 메뉴 → 페널티
                penalty += 1_500_000 * week_count  # 사용 횟수에 비례한 페널티

        # 제약사항: 단가
        current_cost = sum(get_menu_cost(name) for name in display_names)

        # 1) 최대 단가 상한 초과: 강한 페널티 (hard constraint)
        if current_cost > constraints.max_price_limit:
            over_amount = current_cost - constraints.max_price_limit
            penalty += 2_000_000 + (over_amount * 10_000)  # 초과 시 강력한 페널티

        # 2) 목표 단가 기준 평가
        cost_diff = abs(current_cost - constraints.target_price)
        tolerance_amount = constraints.target_price * constraints.cost_tolerance

        if cost_diff <= tolerance_amount:
            # 목표 단가 허용 범위 내: 보너스 점수
            score += 150_000
        else:
            # 허용 범위 초과: 초과 정도에 비례한 페널티
            over_tolerance = cost_diff - tolerance_amount
            penalty += over_tolerance * 500  # 원당 500점 페널티

        # 제약사항: 시설 (강화된 페널티)
        flags = constraints.facility_flags.model_dump()

        # 오븐 필요 메뉴 키워드
        OVEN_KEYWORDS = [
            "오븐", "베이크", "그라탕", "라자냐", "피자", "구이",
            "로스트", "그릴", "오븐구이", "치즈구이", "치즈오븐"
        ]
        # 튀김기 필요 메뉴 키워드
        FRYER_KEYWORDS = [
            "튀김", "돈까스", "탕수육", "치킨", "강정", "커틀릿",
            "까스", "프라이", "너겟", "텐더", "크로켓", "고로케"
        ]
        # 철판 필요 메뉴 키워드
        GRIDDLE_KEYWORDS = [
            "전", "부침", "지짐", "팬케이크", "빈대떡", "파전",
            "호떡", "철판", "볶음밥", "부침개"
        ]

        for name in display_names:
            n = str(name)

            # 오븐 없는데 오븐 필요 메뉴 선택
            if (not flags.get("has_oven", True)) and any(k in n for k in OVEN_KEYWORDS):
                penalty += 2_000_000

            # 튀김기 없는데 튀김 메뉴 선택
            if (not flags.get("has_fryer", True)) and any(k in n for k in FRYER_KEYWORDS):
                penalty += 2_000_000

            # 철판 없는데 철판 필요 메뉴 선택
            if (not flags.get("has_griddle", True)) and any(k in n for k in GRIDDLE_KEYWORDS):
                penalty += 2_000_000

        # 가중치 및 빈도 제한
        for i, name in enumerate(display_names):
            nm = name.strip()
            score += float(weights.get(nm, 0.0)) * 100_000

            is_rice = i == 0 and ("쌀밥" in nm or "흰밥" in nm)
            is_kimchi = i == 5 and ("배추김치" in nm)
            cnt = current_month_counts.get(nm, 0)

            if is_rice or is_kimchi:
                if cnt >= 13:
                    penalty += 2_000_000
                continue

            if i == 0:
                if cnt >= 1:
                    penalty += 2_000_000
            else:
                if cnt >= 2:
                    penalty += 2_000_000

            last_seen, _, cooldown = global_menu_tracker.get(nm, (-100, 0, 0))
            if (global_day_count - last_seen) < cooldown:
                penalty += 2_000_000

        return max(0.1, score - penalty)

    # ========================================
    # 6. 식단 생성 루프
    # ========================================
    rows: List[Dict[str, Any]] = []

    logger.info(f"🔄 {year}년 {month}월 식단 생성 시작...")

    for d in range(1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() >= 5 or dt.date() in holidays:
            continue

        # 주간 번호 확인 및 트래커 초기화
        week_number = dt.isocalendar()[1]
        if week_number != current_week_number:
            current_week_number = week_number
            current_week_menus = {}  # 새로운 주 시작 → 트래커 초기화
            logger.info(f"   📅 {week_number}주차 시작")

        global_day_count += 1
        today_lunch_menus = []

        for meal_type in ["중식", "석식"]:
            current_meal_type = meal_type
            base_seed = opt.seed if opt.seed is not None else int(time.time())
            seed = base_seed + global_day_count + (100 if meal_type == "석식" else 0)

            ga = pygad.GA(
                random_seed=seed,
                fitness_func=fitness_func,
                num_genes=len(ROLE_ORDER),
                gene_space=ctx.gene_space,
                gene_type=int,
                **ga_params,
            )
            ga.run()

            sol, fit, _ = ga.best_solution()
            idxs = sol.astype(int)

            raw_names: List[str] = []
            final_names: List[str] = []

            totals = np.array(
                [ctx.pool_matrices[r][i] for r, i in zip(ROLE_ORDER, idxs)]
            ).sum(axis=0)
            kcal, carb, prot, fat = (
                float(totals[0]),
                float(totals[1]),
                float(totals[2]),
                float(totals[3]),
            )

            for r, i in zip(ROLE_ORDER, idxs):
                original = str(ctx.pool_display_names[r][i])
                raw_names.append(original)

                alg_norm = _normalize_allergy(str(ctx.pool_allergies[r][i]))
                final_names.append(f"{original} ({alg_norm})" if alg_norm else original)

            dessert: Optional[str] = None
            is_dessert_day = (meal_type == "중식" and d in lunch_dessert_days) or (
                meal_type == "석식" and d in dinner_dessert_days
            )
            if is_dessert_day and ctx.dessert_pool:
                dessert_name = random.choice(ctx.dessert_pool)
                dessert_alg = _normalize_allergy(
                    ctx.dessert_allergies.get(dessert_name, "")
                )
                dessert = (
                    f"{dessert_name} ({dessert_alg})" if dessert_alg else dessert_name
                )

            cost = calculate_meal_cost(raw_names)

            iso_date = datetime(year, month, d).strftime("%Y-%m-%d")
            rows.append(
                {
                    "Date": iso_date,
                    "Type": meal_type,
                    "Rice": final_names[0],
                    "Soup": final_names[1],
                    "Main1": final_names[2],
                    "Main2": final_names[3],
                    "Side": final_names[4],
                    "Kimchi": final_names[5],
                    "Dessert": dessert,
                    "RawMenus": raw_names,
                    "Kcal": int(round(kcal)),
                    "Carb": int(round(carb)),
                    "Prot": int(round(prot)),
                    "Fat": int(round(fat)),
                    "Cost": cost,
                }
            )

            if meal_type == "중식":
                today_lunch_menus = [
                    raw_names[1],
                    raw_names[2],
                    raw_names[3],
                    raw_names[4],
                ]

            # tracker 업데이트 (월간 + 주간)
            for nm in raw_names:
                nm_clean = nm.strip()

                # 월간 카운트 업데이트
                current_month_counts[nm_clean] = (
                    current_month_counts.get(nm_clean, 0) + 1
                )

                # 쌀밥/흰밥, 배추김치는 중복 트래킹 제외
                if "쌀밥" in nm_clean or "흰밥" in nm_clean or "배추김치" in nm_clean:
                    continue

                # 주간 카운트 업데이트
                current_week_menus[nm_clean] = current_week_menus.get(nm_clean, 0) + 1

                # 글로벌 트래커 업데이트 (쿨다운)
                last_seen, cnt, _ = global_menu_tracker.get(nm_clean, (-100, 0, 0))
                global_menu_tracker[nm_clean] = (
                    global_day_count,
                    cnt + 1,
                    random.randint(4, 9),
                )

    logger.info(f"✅ 식단 생성 완료: {len(rows)}개 식단")

    # 단가 통계 로깅
    if rows:
        costs = [r["Cost"] for r in rows]
        avg_cost = sum(costs) / len(costs)
        min_cost = min(costs)
        max_cost = max(costs)
        within_target = sum(1 for c in costs if min_price <= c <= max_price)

        logger.info("=" * 60)
        logger.info("💰 단가 통계")
        logger.info("=" * 60)
        logger.info(f"   - 평균 단가: {int(avg_cost):,}원")
        logger.info(f"   - 최저 단가: {min_cost:,}원")
        logger.info(f"   - 최고 단가: {max_cost:,}원")
        logger.info(f"   - 목표 범위 내 식단: {within_target}/{len(rows)}개 ({within_target/len(rows)*100:.1f}%)")
        logger.info(f"   - 최대 상한 초과 식단: {sum(1 for c in costs if c > constraints.max_price_limit)}개")
        logger.info("=" * 60)

    # ========================================
    # 7. 메타데이터 생성
    # ========================================
    meta = {
        "gaParams": ga_params,
        "dessertFrequencyPerWeek": DESSERT_FREQUENCY_PER_WEEK,
        "appliedConstraints": {
            "target_price": constraints.target_price,
            "cost_tolerance": constraints.cost_tolerance,
            "max_price_limit": constraints.max_price_limit,
            "cook_staff": constraints.cook_staff,
            "facility_flags": constraints.facility_flags.model_dump(),
        },
        "nutritionStandard": {
            "nutrition_key": effective_nutrition_key or "DEFAULT(고등_남)",
            "kcal": std_kcal,
            "protein": std_prot,
            "kcal_range": {
                "min": min_kcal_limit,
                "max": max_kcal_limit,
            },
            "carb_range_g": {
                "min": int(min_carb_g),
                "max": int(max_carb_g),
            },
        },
    }

    return rows, meta


def calculate_meal_cost(raw_menus: list) -> int:
    """
    식단 비용 계산

    Args:
        raw_menus: 메뉴명 리스트

    Returns:
        총 비용(원)
    """
    total_cost = 0
    for menu_name in raw_menus:
        cost = get_menu_cost(menu_name)
        total_cost += cost
    return total_cost


def _normalize_token_no_allergy(s: str) -> str:
    """알레르기 괄호 제거 + 공백 제거(유사도/중복 판단용)"""
    s = str(s or "").strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def make_reason(
    best: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    std_kcal: float,
    std_prot: float,
    target_price: Optional[int] = None,
) -> str:
    """짧은 한글 1줄 사유: 칼로리/단백질/단가 핵심 요약"""

    others = [c for c in candidates if c.get("index") != best.get("index")]
    runner = max(others, key=lambda x: float(x.get("fitness", 0.0))) if others else None

    kcal_gap = abs(int(best["kcal"]) - int(round(std_kcal)))
    prot_short = max(0, int(round(std_prot - float(best["prot"]))))

    parts = []

    # 칼로리
    parts.append(f"칼로리±{kcal_gap}kcal")

    # 단백질
    if prot_short == 0:
        parts.append("단백질 충족")
    else:
        parts.append(f"단백질-{prot_short}g")

    # 단가
    if target_price and target_price > 0:
        price_gap = abs(int(best["cost"]) - int(target_price))
        parts.append(f"단가±{price_gap}원")

    # 다른 후보 대비 우수 여부
    if runner:
        r_kcal = abs(int(runner["kcal"]) - int(round(std_kcal)))
        r_prot = max(0, int(round(std_prot - float(runner["prot"]))))

        if kcal_gap < r_kcal or prot_short < r_prot:
            parts.append("후보 대비 우수")

    return " / ".join(parts)


def generate_single_candidate(meal_type: str) -> Dict[str, Any]:
    """
    단일 식단(1끼) 생성 함수
    - 8개의 후보를 생성한 뒤 "점수(영양/비용/중복/다양성)"가 가장 높은 후보를 선택합니다.
    - 후보별로 점수가 갈리도록(=다양하게 나오도록) fitness를 연속형으로 바꿉니다.
    - 8개 후보끼리도 다양하게 나오도록, 이미 뽑힌 후보와 너무 유사하면 페널티를 줍니다(다양성 페널티).
    """
    ctx = get_context()

    # -----------------------------
    # GA 파라미터 (속도/품질 균형)
    # -----------------------------
    ga_params = dict(
        num_generations=50,
        sol_per_pop=30,
        num_parents_mating=12,
        keep_parents=6,
        mutation_percent_genes=25,
        stop_criteria=None,
    )

    # -----------------------------
    # 튜닝 파라미터
    # -----------------------------
    TARGET_RATIO = {"carb": 0.55, "prot": 0.17, "fat": 0.28}
    RATIO_TOL = {"carb": 0.15, "prot": 0.10, "fat": 0.10}

    TARGET_PRICE = getattr(ctx, "target_price", None)  # 없으면 None
    PRICE_TOL = 0.20

    DUP_NAME_PENALTY = 1_500_000
    DUP_CAT_PENALTY = 700_000
    TOO_SIMILAR_PENALTY = 900_000
    SIM_THRESHOLD = 0.75

    N_CAND = 8

    # -----------------------------
    # 유틸(스무스 점수)
    # -----------------------------
    def smooth_gauss_score(x: float, target: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        z = (x - target) / sigma
        return float(np.exp(-0.5 * z * z))

    def smooth_hinge_penalty(x: float, low: float, high: float, k: float) -> float:
        if low <= x <= high:
            return 0.0
        if x < low:
            return k * (low - x)
        return k * (x - high)

    def build_signature(raw_names: List[str], cats: List[str]) -> Tuple[set, set]:
        name_set = {
            _normalize_token_no_allergy(x)
            for x in raw_names
            if _normalize_token_no_allergy(x)
        }
        cat_set = {
            _normalize_token_no_allergy(x)
            for x in cats
            if _normalize_token_no_allergy(x)
        }
        return name_set, cat_set

    # -----------------------------
    # 후보 다양성 관리(이미 뽑힌 후보와 유사하면 감점)
    # -----------------------------
    picked_name_sigs: List[set] = []
    picked_cat_sigs: List[set] = []

    def diversity_penalty(name_sig: set, cat_sig: set) -> float:
        if not picked_name_sigs:
            return 0.0

        max_sim = 0.0
        for ns, cs in zip(picked_name_sigs, picked_cat_sigs):
            sim_n = _jaccard(name_sig, ns)
            sim_c = _jaccard(cat_sig, cs)
            sim = 0.7 * sim_n + 0.3 * sim_c
            if sim > max_sim:
                max_sim = sim

        if max_sim >= SIM_THRESHOLD:
            return (
                TOO_SIMILAR_PENALTY
                * (max_sim - SIM_THRESHOLD)
                / (1.0 - SIM_THRESHOLD + 1e-9)
            )
        return 0.0

    # -----------------------------
    # Fitness: 연속형 점수 + 페널티
    # -----------------------------
    def single_fitness(ga_instance, solution, solution_idx):
        indices = solution.astype(int)

        raw_names = [
            str(ctx.pool_display_names[role][idx])
            for role, idx in zip(ROLE_ORDER, indices)
        ]
        cats = [str(ctx.pool_cats[role][idx]) for role, idx in zip(ROLE_ORDER, indices)]
        nutr_values = np.array(
            [ctx.pool_matrices[role][idx] for role, idx in zip(ROLE_ORDER, indices)]
        )
        totals = nutr_values.sum(axis=0)

        t_kcal = float(totals[0])
        t_carb = float(totals[1])
        t_prot = float(totals[2])
        t_fat = float(totals[3])

        # 1) 영양 점수(연속형)
        kcal_score = smooth_gauss_score(
            t_kcal, STD_KCAL, sigma=max(50.0, STD_KCAL * 0.12)
        )
        prot_score = smooth_gauss_score(
            t_prot, STD_PROT, sigma=max(3.0, STD_PROT * 0.20)
        )
        prot_short_pen = smooth_hinge_penalty(
            t_prot, low=STD_PROT, high=10_000_000, k=600.0
        )

        macro_sum = max(t_carb + t_prot + t_fat, 1e-9)
        r_carb = t_carb / macro_sum
        r_prot = t_prot / macro_sum
        r_fat = t_fat / macro_sum

        ratio_score = (
            smooth_gauss_score(r_carb, TARGET_RATIO["carb"], sigma=RATIO_TOL["carb"])
            * 0.4
            + smooth_gauss_score(r_prot, TARGET_RATIO["prot"], sigma=RATIO_TOL["prot"])
            * 0.3
            + smooth_gauss_score(r_fat, TARGET_RATIO["fat"], sigma=RATIO_TOL["fat"])
            * 0.3
        )

        # 2) 비용 점수/페널티
        total_cost = calculate_meal_cost(raw_names)
        if TARGET_PRICE is None or TARGET_PRICE <= 0:
            price_score = 0.5
            price_pen = 0.0
        else:
            price_score = smooth_gauss_score(
                float(total_cost),
                float(TARGET_PRICE),
                sigma=max(200.0, TARGET_PRICE * 0.15),
            )
            price_pen = smooth_hinge_penalty(
                float(total_cost),
                low=0.0,
                high=float(TARGET_PRICE) * (1.0 + PRICE_TOL),
                k=120.0,
            )

        # 3) 중복 페널티
        penalty = 0.0

        # 주찬1/주찬2 중복(ROLE_ORDER[2], ROLE_ORDER[3] 가정)
        if len(raw_names) >= 4:
            if _normalize_token_no_allergy(raw_names[2]) == _normalize_token_no_allergy(
                raw_names[3]
            ):
                penalty += DUP_NAME_PENALTY
            if _normalize_token_no_allergy(cats[2]) == _normalize_token_no_allergy(
                cats[3]
            ):
                penalty += DUP_CAT_PENALTY

        # 전체 중복
        uniq = set(map(_normalize_token_no_allergy, raw_names))
        dup_count = len(raw_names) - len(uniq)
        if dup_count > 0:
            penalty += dup_count * 400_000

        # 4) 후보 간 다양성 페널티
        name_sig, cat_sig = build_signature(raw_names, cats)
        penalty += diversity_penalty(name_sig, cat_sig)

        # 최종 점수
        score = 100_000.0
        score += 90_000.0 * kcal_score
        score += 90_000.0 * prot_score
        score += 70_000.0 * ratio_score
        score += 40_000.0 * price_score

        penalty += prot_short_pen
        penalty += price_pen

        final = score - penalty
        return max(0.1, float(final))

    # -----------------------------
    # 후보 생성
    # -----------------------------
    candidates: List[Dict[str, Any]] = []
    print("\n🔄 [Python] 8개 후보 식단 생성 중...")

    for candidate_idx in range(N_CAND):
        # ✅ seed 범위 에러 방지: 0 ~ 2**32-1 로 마스킹
        seed = (
            int(time.time() * 1000) + candidate_idx * 10_000 + random.randint(0, 9999)
        ) & 0xFFFFFFFF

        ga = pygad.GA(
            random_seed=seed,
            fitness_func=single_fitness,
            num_genes=len(ROLE_ORDER),
            gene_space=ctx.gene_space,
            gene_type=int,
            **ga_params,
        )
        ga.run()

        sol, fit, _ = ga.best_solution()
        idxs = sol.astype(int)

        # 영양 합산
        nutr_values = np.array(
            [ctx.pool_matrices[role][idx] for role, idx in zip(ROLE_ORDER, idxs)]
        )
        totals = nutr_values.sum(axis=0)
        kcal = float(totals[0])
        carb = float(totals[1])
        prot = float(totals[2])
        fat = float(totals[3])

        # 메뉴 구성 (알레르기 포함 display)
        raw_names: List[str] = []
        display_names: List[str] = []
        cats: List[str] = []

        for r, i in zip(ROLE_ORDER, idxs):
            original = str(ctx.pool_display_names[r][i])
            alg_norm = _normalize_allergy(str(ctx.pool_allergies[r][i]))
            cat = str(ctx.pool_cats[r][i])

            raw_names.append(original)
            cats.append(cat)

            if alg_norm:
                display_names.append(f"{original} ({alg_norm})")
            else:
                display_names.append(original)

        # 디저트(선택)
        dessert = None
        dessert_raw = None
        if getattr(ctx, "dessert_pool", None) and random.random() > 0.5:
            dessert_raw = random.choice(ctx.dessert_pool)
            dessert_alg = _normalize_allergy(ctx.dessert_allergies.get(dessert_raw, ""))
            dessert = f"{dessert_raw} ({dessert_alg})" if dessert_alg else dessert_raw
            raw_names.append(dessert_raw)
            display_names.append(dessert)

        # 비용
        total_cost = calculate_meal_cost(raw_names)

        # ✅ 이번 후보 시그니처 저장(다음 후보가 비슷하면 fitness에서 감점)
        name_sig, cat_sig = build_signature(raw_names, cats)
        picked_name_sigs.append(name_sig)
        picked_cat_sigs.append(cat_sig)

        candidate_info = {
            "index": candidate_idx + 1,
            "menus": display_names,
            "rawMenus": raw_names,
            "dessert": dessert,
            "kcal": int(round(kcal)),
            "carb": int(round(carb)),
            "prot": int(round(prot)),
            "fat": int(round(fat)),
            "cost": int(total_cost),
            "fitness": float(fit),
        }
        candidates.append(candidate_info)

        print(
            f"  후보 {candidate_idx + 1}/{N_CAND} 생성 완료 (적합도: {fit:.0f}, 비용: {total_cost}원, kcal: {int(round(kcal))})"
        )

    # -----------------------------
    # 최적 후보 선택
    # -----------------------------
    best_candidate = max(candidates, key=lambda x: float(x["fitness"]))

    # -----------------------------
    # reason 생성(후보 비교 기반)
    # -----------------------------
    reason = make_reason(
        best_candidate,
        candidates,
        std_kcal=float(STD_KCAL),
        std_prot=float(STD_PROT),
        target_price=(
            int(TARGET_PRICE)
            if (TARGET_PRICE is not None and TARGET_PRICE > 0)
            else None
        ),
    )

    print(
        f"\n✅ [Python] 최적 식단 선택: 후보 {best_candidate['index']} (적합도: {best_candidate['fitness']:.0f})"
    )
    print(
        f"   📊 영양: kcal={best_candidate['kcal']}, carb={best_candidate['carb']}, prot={best_candidate['prot']}, fat={best_candidate['fat']}"
    )
    print(f"   💰 비용: {best_candidate['cost']}원")
    print(f"   🍽️ 메뉴: {best_candidate['menus']}")
    print(f"   📝 사유: {reason}")

    return {
        "menus": best_candidate["menus"],
        "rawMenus": best_candidate["rawMenus"],
        "dessert": best_candidate["dessert"],
        "kcal": best_candidate["kcal"],
        "carb": best_candidate["carb"],
        "prot": best_candidate["prot"],
        "fat": best_candidate["fat"],
        "cost": best_candidate["cost"],
        "candidates": candidates,
        "reason": reason,
    }
