from __future__ import annotations
import calendar, json, os, random, time, logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygad

from app.core.config import (
    COST_DB_PATH,
    WEIGHT_DB_PATH,
    ROLE_ORDER,
    DESSERT_FREQUENCY_PER_WEEK,
    STD_KCAL,
    STD_PROT,
)
from app.models.schemas import Options
from app.services.food_loader import get_context
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
    year: int, month: int, opt: Options, report_data: Optional[Dict] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    월간 식단 생성

    Args:
        year: 연도
        month: 월
        opt: 옵션
        report_data: 리포트 JSON (Spring이 DB에서 조회하여 전달)

    Returns:
        (식단 리스트, 메타데이터)
    """
    ctx = get_context()
    constraints = opt.constraints

    # ========================================
    # 1. 제약사항 처리
    # ========================================
    constraints = opt.constraints

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
    logger.info(f"   목표 단가: {constraints.target_price:,}원")
    logger.info(f"   허용 오차: ±{constraints.cost_tolerance*100:.0f}%")
    logger.info(f"   최대 상한: {constraints.max_price_limit:,}원")
    logger.info(f"   조리 인원: {constraints.cook_staff}명")
    logger.info(f"   시설 현황:")
    logger.info(
        f"      - 오븐: {'✅ 사용 가능' if constraints.facility_flags.has_oven else '❌ 사용 불가'}"
    )
    logger.info(
        f"      - 튀김기: {'✅ 사용 가능' if constraints.facility_flags.has_fryer else '❌ 사용 불가'}"
    )
    logger.info(
        f"      - 철판: {'✅ 사용 가능' if constraints.facility_flags.has_griddle else '❌ 사용 불가'}"
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

    holidays = get_holidays(year)
    last_day = calendar.monthrange(year, month)[1]

    # 디저트 주 2회 랜덤 배정
    weekdays_by_week: Dict[int, List[int]] = {}
    for d in range(1, last_day + 1):
        dt = datetime(year, month, d)
        if dt.weekday() >= 5 or dt.date() in holidays:
            continue
        wk = dt.isocalendar()[1]
        weekdays_by_week.setdefault(wk, []).append(d)

    lunch_dessert_days: set[int] = set()
    dinner_dessert_days: set[int] = set()
    for days in weekdays_by_week.values():
        k = min(len(days), DESSERT_FREQUENCY_PER_WEEK)
        if k > 0:
            lunch_dessert_days.update(random.sample(days, k))
            dinner_dessert_days.update(random.sample(days, k))

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

    # ========================================
    # 5. Fitness 함수
    # ========================================
    def fitness_func(ga_instance, solution, solution_idx):
        nonlocal global_day_count, current_meal_type, today_lunch_menus

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

        # 영양소 평가
        if (STD_KCAL * 0.9) <= t_kcal <= (STD_KCAL * 1.1):
            score += 200_000
        else:
            penalty += 100_000 + abs(t_kcal - STD_KCAL) * 200

        if t_prot < STD_PROT:
            penalty += (STD_PROT - t_prot) * 20_000

        # 중복 방지
        if display_names[2] == display_names[3]:
            penalty += 2_000_000
        if cats[2] == cats[3]:
            penalty += 1_000_000

        if current_meal_type == "석식" and today_lunch_menus:
            curr_set = {display_names[i] for i in [1, 2, 3, 4]}
            if curr_set & set(today_lunch_menus):
                penalty += 2_000_000

        # 제약사항: 단가
        current_cost = sum(get_menu_cost(name) for name in display_names)

        if current_cost > constraints.max_price_limit:
            penalty += (current_cost - constraints.max_price_limit) * 5000

        cost_diff = abs(current_cost - constraints.target_price)
        if cost_diff > constraints.target_price * constraints.cost_tolerance:
            penalty += (cost_diff / 10.0) * 1000

        # 제약사항: 시설
        flags = constraints.facility_flags.model_dump()
        for name in display_names:
            n = str(name)

            if (not flags.get("has_oven", True)) and any(
                k in n for k in ["오븐", "베이크", "그라탕", "라자냐"]
            ):
                penalty += 500_000

            if (not flags.get("has_fryer", True)) and any(
                k in n for k in ["튀김", "돈까스", "탕수육", "치킨", "강정"]
            ):
                penalty += 500_000

            if (not flags.get("has_griddle", True)) and any(
                k in n for k in ["전", "부침", "지짐", "팬케이크", "빈대떡"]
            ):
                penalty += 500_000

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
                dessert = random.choice(ctx.dessert_pool)

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

            # tracker 업데이트
            for nm in raw_names:
                nm_clean = nm.strip()
                current_month_counts[nm_clean] = (
                    current_month_counts.get(nm_clean, 0) + 1
                )
                if "쌀밥" in nm_clean or "흰밥" in nm_clean or "배추김치" in nm_clean:
                    continue
                last_seen, cnt, _ = global_menu_tracker.get(nm_clean, (-100, 0, 0))
                global_menu_tracker[nm_clean] = (
                    global_day_count,
                    cnt + 1,
                    random.randint(4, 9),
                )

    logger.info(f"✅ 식단 생성 완료: {len(rows)}개 식단")

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


# ==============================================================================
# ★★★ [수정] Java의 'AI 자동 대체' 기능이 호출하는 함수
# 8개의 후보 식단을 생성하고 그 중 최적의 식단을 선택하여 반환합니다.
# ==============================================================================
def generate_single_candidate(meal_type: str) -> Dict[str, Any]:
    """
    단일 식단(1끼) 생성 함수
    8개의 후보를 생성한 뒤 영양 균형이 가장 잘 잡힌 식단을 선택합니다.
    """
    ctx = get_context()

    # 1끼 생성을 위한 가벼운 GA 파라미터 (속도 중요)
    ga_params = dict(
        num_generations=50,
        sol_per_pop=20,
        num_parents_mating=10,
        keep_parents=5,
        mutation_percent_genes=20,
        stop_criteria=None,
    )

    def single_fitness(ga_instance, solution, solution_idx):
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
        t_kcal, t_prot = float(totals[0]), float(totals[2])

        score = 100_000.0
        penalty = 0.0

        # 영양소 제약 (월간보다 조금 더 유연하게)
        if (STD_KCAL * 0.8) <= t_kcal <= (STD_KCAL * 1.2):
            score += 50_000
        else:
            penalty += abs(t_kcal - STD_KCAL) * 100

        if t_prot < STD_PROT:
            penalty += (STD_PROT - t_prot) * 1000

        # 중복 제약
        if display_names[2] == display_names[3]:
            penalty += 500_000
        if cats[2] == cats[3]:
            penalty += 200_000

        return max(0.1, score - penalty)

    # ==========================================================================
    # ★★★ [추가] 8개 후보 식단 생성 로직
    # ==========================================================================
    candidates = []

    print("\n🔄 [Python] 8개 후보 식단 생성 중...")

    for candidate_idx in range(8):
        # 각 후보마다 다른 시드를 사용하여 다양한 식단 생성
        seed = int(time.time()) + candidate_idx * 1000

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

        # ======================================================================
        # [추가] 영양 정보 계산
        # ======================================================================
        nutr_values = np.array(
            [ctx.pool_matrices[role][idx] for role, idx in zip(ROLE_ORDER, idxs)]
        )
        totals = nutr_values.sum(axis=0)
        kcal = float(totals[0])
        carb = float(totals[1])
        prot = float(totals[2])
        fat = float(totals[3])
        # ======================================================================

        # 메뉴 구성 (알레르기 정보 포함)
        raw_names = []
        display_names = []

        for r, i in zip(ROLE_ORDER, idxs):
            original = str(ctx.pool_display_names[r][i])
            alg_norm = _normalize_allergy(str(ctx.pool_allergies[r][i]))

            raw_names.append(original)

            if alg_norm:
                display_names.append(f"{original} ({alg_norm})")
            else:
                display_names.append(original)

        # 디저트 처리
        dessert = None
        if ctx.dessert_pool and random.random() > 0.5:
            dessert = random.choice(ctx.dessert_pool)
            raw_names.append(dessert)

        # 비용 계산
        total_cost = calculate_meal_cost(raw_names)

        # ======================================================================
        # [추가] 후보 정보 저장
        # ======================================================================
        candidate_info = {
            "index": candidate_idx + 1,
            "menus": display_names,
            "rawMenus": raw_names,
            "dessert": dessert,
            "kcal": int(round(kcal)),
            "carb": int(round(carb)),
            "prot": int(round(prot)),
            "fat": int(round(fat)),
            "cost": total_cost,
            "fitness": fit,  # 적합도 점수 (높을수록 좋음)
        }

        candidates.append(candidate_info)

        print(
            f"  후보 {candidate_idx + 1}/8 생성 완료 (적합도: {fit:.0f}, 비용: {total_cost}원, kcal: {int(round(kcal))})"
        )
    # ==========================================================================

    # ==========================================================================
    # ★★★ [추가] 8개 후보 중 최적의 식단 선택
    # 적합도(fitness)가 가장 높은 후보를 선택합니다.
    # ==========================================================================
    best_candidate = max(candidates, key=lambda x: x["fitness"])

    print(
        f"\n✅ [Python] 최적 식단 선택: 후보 {best_candidate['index']} (적합도: {best_candidate['fitness']:.0f})"
    )
    print(
        f"   📊 영양: kcal={best_candidate['kcal']}, carb={best_candidate['carb']}, prot={best_candidate['prot']}, fat={best_candidate['fat']}"
    )
    print(f"   💰 비용: {best_candidate['cost']}원")
    print(f"   🍽️ 메뉴: {best_candidate['menus']}")
    # ==========================================================================

    # ==========================================================================
    # ★★★ [수정] 반환 값에 영양 정보 + candidates 추가
    # Java에서 8개 후보를 확인할 수 있도록 candidates 필드 포함
    # ==========================================================================
    return {
        "menus": best_candidate["menus"],  # 최적 식단의 메뉴 (알레르기 정보 포함)
        "rawMenus": best_candidate["rawMenus"],  # 최적 식단의 순수 메뉴명
        "dessert": best_candidate["dessert"],  # 최적 식단의 디저트
        "kcal": best_candidate["kcal"],  # ★ 추가: 총 칼로리
        "carb": best_candidate["carb"],  # ★ 추가: 총 탄수화물
        "prot": best_candidate["prot"],  # ★ 추가: 총 단백질
        "fat": best_candidate["fat"],  # ★ 추가: 총 지방
        "cost": best_candidate["cost"],  # 총 비용
        "candidates": candidates,  # ★ 추가: 8개 후보 전체 정보 (Java 검증용)
        "reason": "AI가 8개의 후보 중 영양 균형과 선호도를 고려하여 최적의 메뉴를 선택했습니다.",
    }
    # ==========================================================================
