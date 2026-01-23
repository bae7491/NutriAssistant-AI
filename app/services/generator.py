from __future__ import annotations
import calendar, json, os, random, time
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


def _normalize_allergy(alg_val: str) -> Optional[str]:
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


def generate_one_month(
    year: int, month: int, opt: Options
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ctx = get_context()

    weights: Dict[str, float] = {
        k: float(v) for k, v in _load_json_dict(WEIGHT_DB_PATH, "weights").items()
    }

    # ✅ 수정: Spring에서 단가 DB 로드 (DB 없으면 AI 자동 생성)
    print("💰 단가 DB 로딩 중...")
    cost_db = get_cost_db()
    print(f"✅ 단가 DB 로드 완료: {len(cost_db)}개 메뉴")

    global_day_count = 0
    global_menu_tracker: Dict[str, Tuple[int, int, int]] = {}
    current_month_counts: Dict[str, int] = {}

    holidays = get_holidays(year)
    last_day = calendar.monthrange(year, month)[1]

    # 디저트 주 2회(주중+공휴일제외 기준)
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

    current_meal_type = "중식"
    today_lunch_menus: List[str] = []

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

        if (STD_KCAL * 0.9) <= t_kcal <= (STD_KCAL * 1.1):
            score += 200_000
        else:
            penalty += 100_000 + abs(t_kcal - STD_KCAL) * 200

        if t_prot < STD_PROT:
            penalty += (STD_PROT - t_prot) * 20_000

        if display_names[2] == display_names[3]:
            penalty += 2_000_000
        if cats[2] == cats[3]:
            penalty += 1_000_000

        if current_meal_type == "석식" and today_lunch_menus:
            curr_set = {display_names[i] for i in [1, 2, 3, 4]}
            if curr_set & set(today_lunch_menus):
                penalty += 2_000_000

        # ✅ 수정: get_menu_cost 사용
        current_cost = sum(get_menu_cost(name) for name in display_names)

        if current_cost > opt.maxPriceLimit:
            penalty += (current_cost - opt.maxPriceLimit) * 5000
        cost_diff = abs(current_cost - opt.targetPrice)
        if cost_diff > opt.targetPrice * opt.costTolerance:
            penalty += (cost_diff / 10.0) * 1000

        flags = opt.facilityFlags.model_dump()
        for name in display_names:
            n = str(name)
            if (not flags.get("has_oven", True)) and any(
                k in n for k in ["오븐", "베이크", "그라탕"]
            ):
                penalty += 200_000
            if (not flags.get("has_fryer", True)) and any(
                k in n for k in ["튀김", "돈까스", "탕수육", "치킨"]
            ):
                penalty += 200_000

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

    rows: List[Dict[str, Any]] = []

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
                raw_names.append(dessert)

            # calculate_meal_cost 함수 사용
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

            # tracker 업데이트(쿨타임 4~9)
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

    meta = {
        "gaParams": ga_params,
        "dessertFrequencyPerWeek": DESSERT_FREQUENCY_PER_WEEK,
    }
    return rows, meta


def calculate_meal_cost(raw_menus: list) -> int:
    """
    식단 비용 계산 (실제 단가 DB 사용)

    Args:
        raw_menus: 메뉴명 리스트 (예: ["쌀밥", "김치찌개", ...])

    Returns:
        총 비용(원)
    """
    total_cost = 0
    for menu_name in raw_menus:
        cost = get_menu_cost(menu_name)
        total_cost += cost
    return total_cost
