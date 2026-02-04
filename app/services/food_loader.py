from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import re
import logging
import sys

import numpy as np
import pandas as pd

from app.core.config import (
    INTERNAL_API_KEY,
    SPRING_FOOD_API,
    SPRING_PAGE_SIZE,
    SPRING_TIMEOUT_SECONDS,
    ROLE_ORDER,
    NUM_COLS,
    DESSERT_CATEGORIES,
)
from app.utils.text import get_role

# 로거 설정
logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


@dataclass
class FoodContext:
    ready: bool
    pools: Dict[str, pd.DataFrame]
    pool_matrices: Dict[str, np.ndarray]
    pool_display_names: Dict[str, np.ndarray]
    pool_cats: Dict[str, np.ndarray]
    pool_allergies: Dict[str, np.ndarray]
    default_rice_idx: int
    gene_space: List[List[int]]
    source: str
    dessert_pool: List[str]
    dessert_allergies: Dict[str, str]  # 디저트 메뉴명 → 알레르기 정보 매핑
    last_error: Optional[str] = None
    load_timestamp: Optional[str] = None  # 로드 시간 추가
    memory_size_mb: Optional[float] = None  # 메모리 크기 추가


_CTX: Optional[FoodContext] = None


def get_context() -> FoodContext:
    """메모리에 캐싱된 음식 DB 컨텍스트 반환"""
    if _CTX is None or not _CTX.ready:
        raise RuntimeError("dataset not ready")
    return _CTX


def get_context_stats() -> Dict[str, Any]:
    """컨텍스트 통계 정보 반환 (디버깅용)"""
    if _CTX is None:
        return {"status": "not_loaded", "error": "Context has not been initialized"}

    if not _CTX.ready:
        return {"status": "error", "error": _CTX.last_error, "source": _CTX.source}

    # 역할별 통계
    role_stats = {}
    total_menus = 0
    for role, pool in _CTX.pools.items():
        count = len(pool)
        total_menus += count
        role_stats[role] = {
            "count": count,
            "sample_menus": pool["menuName"].head(3).tolist() if not pool.empty else [],
        }

    # 영양소 범위 계산
    nutrition_ranges = {}
    for col in NUM_COLS:
        all_values = []
        for pool in _CTX.pools.values():
            if col in pool.columns:
                all_values.extend(pool[col].dropna().tolist())

        if all_values:
            nutrition_ranges[col] = {
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "mean": float(np.mean(all_values)),
            }

    return {
        "status": "ready",
        "source": _CTX.source,
        "load_timestamp": _CTX.load_timestamp,
        "memory_size_mb": _CTX.memory_size_mb,
        "total_menus": total_menus,
        "role_breakdown": role_stats,
        "dessert_pool_size": len(_CTX.dessert_pool),
        "dessert_samples": _CTX.dessert_pool[:5],
        "nutrition_ranges": nutrition_ranges,
        "gene_space_lengths": [len(space) for space in _CTX.gene_space],
    }


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError("requests 패키지가 필요합니다. `pip install requests`")


def fetch_foodinfo_all_from_spring() -> pd.DataFrame:
    """Spring API에서 모든 음식 정보 가져오기"""
    _require_requests()

    logger.info(f"🔄 Spring API에서 음식 DB 로딩 시작: {SPRING_FOOD_API}")

    rows: List[Dict[str, Any]] = []
    page = 0
    size = SPRING_PAGE_SIZE

    headers: Dict[str, str] = {}
    if INTERNAL_API_KEY:
        headers["X-Internal-API-Key"] = INTERNAL_API_KEY
        logger.info(f"   🔑 API Key 설정됨 (길이: {len(INTERNAL_API_KEY)})")
    else:
        logger.warning("   ⚠️ INTERNAL_API_KEY 환경변수가 설정되지 않았습니다!")

    def _unwrap(obj: Any) -> Any:
        while isinstance(obj, dict):
            for k in ("data", "result", "payload"):
                v = obj.get(k)
                if isinstance(v, (dict, list)):
                    obj = v
                    break
            else:
                break
        return obj

    start_time = time.time()
    total_fetched = 0

    while True:
        try:
            resp = requests.get(
                SPRING_FOOD_API,
                params={"page": page, "size": size},
                headers=headers,
                timeout=SPRING_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"❌ Spring API 요청 실패 (page={page}): {e}")
            raise RuntimeError(f"Spring API 요청 실패: {e}")

        try:
            data: Any = resp.json()
        except Exception as e:
            logger.error(f"❌ Spring API JSON 파싱 실패: {e}")
            raise RuntimeError(f"Spring API JSON 파싱 실패: {e}")

        data = _unwrap(data)

        batch: List[Dict[str, Any]] = []
        is_last = True

        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, list):
                batch = content
                is_last = bool(data.get("last", False))
            else:
                for k in ("items", "rows", "list"):
                    v = data.get(k)
                    if isinstance(v, list):
                        batch = v
                        is_last = True
                        break
                if not batch:
                    raise RuntimeError(
                        f"Spring 응답에 content/items/rows/list가 없습니다. keys={list(data.keys())}"
                    )
        elif isinstance(data, list):
            batch = data
            is_last = True
        else:
            raise RuntimeError(f"Spring 응답 형식이 예상과 다릅니다: type={type(data)}")

        if not batch:
            break

        rows.extend(batch)
        total_fetched += len(batch)

        logger.info(
            f"   📄 Page {page}: {len(batch)}개 로드됨 (누적: {total_fetched}개)"
        )

        if is_last:
            break
        page += 1

    elapsed = time.time() - start_time
    logger.info(
        f"✅ Spring API 데이터 로드 완료: {total_fetched}개 (소요시간: {elapsed:.2f}초)"
    )

    return pd.DataFrame(rows)


def _extract_number(val: Any, default: float = 100.0) -> float:
    """문자열에서 숫자 추출"""
    if val is None:
        return default
    s = str(val)
    m = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not m:
        return default
    try:
        return float(m[0])
    except Exception:
        return default


def _calculate_memory_size(ctx: FoodContext) -> float:
    """컨텍스트의 대략적인 메모리 크기 계산 (MB)"""
    try:
        total_bytes = 0

        # DataFrame 크기
        for pool in ctx.pools.values():
            total_bytes += pool.memory_usage(deep=True).sum()

        # NumPy 배열 크기
        for matrix in ctx.pool_matrices.values():
            total_bytes += matrix.nbytes

        for arr in ctx.pool_display_names.values():
            total_bytes += arr.nbytes

        for arr in ctx.pool_cats.values():
            total_bytes += arr.nbytes

        for arr in ctx.pool_allergies.values():
            total_bytes += arr.nbytes

        # 디저트 풀
        total_bytes += sys.getsizeof(ctx.dessert_pool)

        return total_bytes / 1024 / 1024  # MB로 변환

    except Exception as e:
        logger.warning(f"메모리 크기 계산 실패: {e}")
        return 0.0


def load_spring_and_build_context() -> None:
    """Spring(MySQL) 기반 food_info를 읽어 pools/행렬/디저트풀 구성"""
    global _CTX

    logger.info("=" * 80)
    logger.info("🚀 음식 DB 컨텍스트 빌드 시작")
    logger.info("=" * 80)

    load_start_time = time.time()
    last_err: Optional[str] = None
    df: pd.DataFrame = pd.DataFrame([])

    # Spring API에서 데이터 가져오기 (재시도 로직)
    for attempt in range(1, 11):
        try:
            df = fetch_foodinfo_all_from_spring()
            if df is not None and not df.empty:
                last_err = None
                break
            last_err = "Spring food API가 빈 데이터를 반환했습니다."
            logger.warning(f"⚠️ 시도 {attempt}/10: {last_err}")
        except Exception as e:
            last_err = str(e)
            logger.warning(f"⚠️ 시도 {attempt}/10 실패: {last_err}")

        if attempt < 10:
            wait_time = min(attempt, 10)
            logger.info(f"   ⏳ {wait_time}초 후 재시도...")
            time.sleep(wait_time)

    if df.empty:
        logger.error(f"❌ 음식 DB 로드 최종 실패: {last_err}")
        _CTX = FoodContext(
            ready=False,
            pools={},
            pool_matrices={},
            pool_display_names={},
            pool_cats={},
            pool_allergies={},
            default_rice_idx=0,
            gene_space=[],
            source=SPRING_FOOD_API,
            dessert_pool=[],
            dessert_allergies={},
            last_error=last_err,
            load_timestamp=None,
            memory_size_mb=0.0,
        )
        return

    logger.info(f"📊 데이터 전처리 시작 (총 {len(df)}개 행)")

    def _pick(*names: str) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([None] * len(df))

    merged = df.copy()

    # 컬럼 매핑
    merged["menuName"] = _pick("foodName", "food_name", "name").fillna("").astype(str)
    merged["category"] = (
        _pick("category", "식품대분류명", "foodCategory").fillna("").astype(str)
    )
    merged["allergy"] = (
        _pick("allergyInfo", "allergy_info", "allergy").fillna("").astype(str)
    )

    # 영양소 정규화
    merged["kcal"] = pd.to_numeric(_pick("kcal", "energy"), errors="coerce").fillna(0)
    merged["carbs"] = pd.to_numeric(
        _pick("carbs", "carbohydrate"), errors="coerce"
    ).fillna(0)
    merged["protein"] = pd.to_numeric(_pick("protein"), errors="coerce").fillna(0)
    merged["fat"] = pd.to_numeric(_pick("fat"), errors="coerce").fillna(0)
    merged["calcium"] = pd.to_numeric(_pick("calcium"), errors="coerce").fillna(0)
    merged["iron"] = pd.to_numeric(_pick("iron"), errors="coerce").fillna(0)
    merged["vitaminA"] = pd.to_numeric(
        _pick("vitaminA", "vitamin_a"), errors="coerce"
    ).fillna(0)
    merged["vitaminC"] = pd.to_numeric(
        _pick("vitaminC", "vitamin_c"), errors="coerce"
    ).fillna(0)

    # 1인분 환산
    basis = _pick("servingBasis", "영양성분함량기준량")
    weight = _pick("foodWeight", "식품중량")
    if basis is not None and weight is not None:
        base_vals = basis.apply(lambda x: _extract_number(x, 100.0)).replace(0, 100.0)
        serve_vals = weight.apply(lambda x: _extract_number(x, 100.0)).replace(0, 100.0)
        ratio = (serve_vals / base_vals).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        for c in NUM_COLS:
            merged[c] = (merged[c] * ratio).fillna(0)
        logger.info("   ✅ 1인분 기준 영양소 환산 완료")

    # 디저트 풀 분리 (알레르기 정보 포함)
    dessert_mask = merged["category"].astype(str).str.strip().isin(DESSERT_CATEGORIES)
    dessert_df = merged.loc[dessert_mask, ["menuName", "allergy"]].dropna(
        subset=["menuName"]
    )
    dessert_df["menuName"] = dessert_df["menuName"].astype(str).str.strip()
    dessert_df["allergy"] = dessert_df["allergy"].fillna("").astype(str)

    # 중복 제거 (첫 번째 등장 기준)
    dessert_df = dessert_df.drop_duplicates(subset=["menuName"], keep="first")

    dessert_pool = dessert_df["menuName"].tolist()
    dessert_allergies = dict(zip(dessert_df["menuName"], dessert_df["allergy"]))
    logger.info(f"🍰 디저트/음료 풀 분리: {len(dessert_pool)}개")

    # 일반 메뉴 후보
    candidates = merged.loc[~dessert_mask].copy()
    candidates["role"] = candidates["category"].apply(get_role)
    candidates = candidates.dropna(subset=["role"]).copy()

    for c in NUM_COLS:
        candidates[c] = pd.to_numeric(candidates[c], errors="coerce").fillna(0)

    # 역할별 풀 생성
    unique_roles = list(dict.fromkeys(ROLE_ORDER))
    pools = {
        r: candidates[candidates["role"] == r].reset_index(drop=True)
        for r in unique_roles
    }

    # 역할별 메뉴 수 로깅
    logger.info("📋 역할별 메뉴 수:")
    for role, pool in pools.items():
        logger.info(f"   - {role}: {len(pool)}개")

    missing = [r for r in unique_roles if r not in pools or pools[r].empty]
    if missing:
        error_msg = f"역할별 pool이 비어있습니다: {', '.join(missing)}"
        logger.error(f"❌ {error_msg}")
        _CTX = FoodContext(
            ready=False,
            pools={},
            pool_matrices={},
            pool_display_names={},
            pool_cats={},
            pool_allergies={},
            default_rice_idx=0,
            gene_space=[],
            source=SPRING_FOOD_API,
            dessert_pool=dessert_pool,
            dessert_allergies=dessert_allergies,
            last_error=error_msg,
            load_timestamp=None,
            memory_size_mb=0.0,
        )
        return

    # 행렬 및 배열 생성
    pool_matrices = {r: pools[r][NUM_COLS].values for r in pools}
    pool_display_names = {r: pools[r]["menuName"].values for r in pools}
    pool_cats = {r: pools[r]["category"].values for r in pools}
    pool_allergies = {
        r: pools[r]["allergy"].fillna("").astype(str).values for r in pools
    }

    gene_space = [list(range(len(pools[r]))) for r in ROLE_ORDER]

    # 기본 쌀밥 인덱스 찾기
    default_rice_idx = 0
    try:
        default_rice_idx = next(
            i
            for i, n in enumerate(pool_display_names["밥"])
            if "쌀밥" in str(n) or "흰밥" in str(n)
        )
        logger.info(f"   ✅ 기본 쌀밥 인덱스: {default_rice_idx}")
    except Exception:
        logger.warning("   ⚠️ 기본 쌀밥을 찾지 못했습니다. 인덱스 0 사용")
        default_rice_idx = 0

    # 컨텍스트 생성
    from datetime import datetime

    load_timestamp = datetime.now().isoformat()

    _CTX = FoodContext(
        ready=True,
        pools=pools,
        pool_matrices=pool_matrices,
        pool_display_names=pool_display_names,
        pool_cats=pool_cats,
        pool_allergies=pool_allergies,
        default_rice_idx=default_rice_idx,
        gene_space=gene_space,
        source=SPRING_FOOD_API,
        dessert_pool=dessert_pool,
        dessert_allergies=dessert_allergies,
        last_error=None,
        load_timestamp=load_timestamp,
        memory_size_mb=0.0,  # 임시값
    )

    # 메모리 크기 계산
    _CTX.memory_size_mb = _calculate_memory_size(_CTX)

    load_elapsed = time.time() - load_start_time

    logger.info("=" * 80)
    logger.info("✅ 음식 DB 컨텍스트 빌드 완료!")
    logger.info(f"   - 총 메뉴 수: {sum(len(p) for p in pools.values())}개")
    logger.info(f"   - 디저트/음료: {len(dessert_pool)}개")
    logger.info(f"   - 메모리 사용량: {_CTX.memory_size_mb:.2f} MB")
    logger.info(f"   - 소요 시간: {load_elapsed:.2f}초")
    logger.info(f"   - 로드 시각: {load_timestamp}")
    logger.info("=" * 80)


def get_valid_menu_names() -> List[str]:
    """
    DB에서 유효한 메뉴명 목록 조회

    Returns:
        메뉴명 리스트
    """
    ctx = get_context()

    # 모든 역할의 메뉴명을 합침
    valid_names = []
    for role in ["밥", "국", "주찬", "부찬", "김치", "디저트"]:
        if role in ctx.pool_display_names:
            valid_names.extend(ctx.pool_display_names[role].tolist())

    # 디저트도 추가
    if ctx.dessert_pool:
        valid_names.extend(ctx.dessert_pool)

    # 중복 제거
    valid_names = list(set(valid_names))

    return valid_names


def build_context_with_new_menus(new_menus: List[Dict[str, Any]]) -> FoodContext:
    """
    기존 컨텍스트에 신메뉴를 병합한 새 컨텍스트 생성

    Args:
        new_menus: 신메뉴 리스트 (NewMenuInput 형태의 dict)

    Returns:
        신메뉴가 병합된 FoodContext
    """
    import copy
    from app.utils.text import get_role

    base_ctx = get_context()

    if not new_menus:
        return base_ctx

    logger.info("=" * 60)
    logger.info(f"🆕 신메뉴 {len(new_menus)}개 병합 시작")
    logger.info("=" * 60)

    # 깊은 복사로 기존 컨텍스트 보존
    new_pools = {role: pool.copy() for role, pool in base_ctx.pools.items()}
    new_dessert_pool = list(base_ctx.dessert_pool)
    new_dessert_allergies = dict(base_ctx.dessert_allergies)

    # 디저트 카테고리 목록
    from app.core.config import DESSERT_CATEGORIES, NUM_COLS

    added_count = {"dessert": 0, "meal": 0}

    for menu in new_menus:
        food_name = menu.get("food_name", "").strip()
        category = menu.get("category", "").strip()
        allergy = menu.get("allergy_info", "") or ""

        if not food_name:
            logger.warning(f"   ⚠️ 신메뉴 스킵 (이름 없음): {menu}")
            continue

        # 디저트/음료 카테고리인 경우
        if category in DESSERT_CATEGORIES:
            if food_name not in new_dessert_pool:
                new_dessert_pool.append(food_name)
                new_dessert_allergies[food_name] = str(allergy)
                added_count["dessert"] += 1
                logger.info(f"   🍰 디저트 추가: {food_name} ({category})")
        else:
            # 일반 메뉴 - 역할 결정
            role = get_role(category)
            if role is None:
                logger.warning(f"   ⚠️ 역할 매핑 실패: {food_name} ({category})")
                continue

            if role not in new_pools:
                logger.warning(f"   ⚠️ 존재하지 않는 역할: {role}")
                continue

            # 이미 존재하는지 확인
            existing_names = new_pools[role]["menuName"].tolist()
            if food_name in existing_names:
                logger.info(f"   ℹ️ 이미 존재하는 메뉴 스킵: {food_name}")
                continue

            # 새 행 추가
            new_row = {
                "menuName": food_name,
                "category": category,
                "allergy": str(allergy),
                "kcal": float(menu.get("kcal", 0)),
                "carbs": float(menu.get("carbs", 0)),
                "protein": float(menu.get("protein", 0)),
                "fat": float(menu.get("fat", 0)),
                "calcium": float(menu.get("calcium", 0)),
                "iron": float(menu.get("iron", 0)),
                "vitaminA": float(menu.get("vitamin_a", 0)),
                "vitaminC": float(menu.get("vitamin_c", 0)),
                "role": role,
            }

            new_pools[role] = pd.concat(
                [new_pools[role], pd.DataFrame([new_row])],
                ignore_index=True,
            )
            added_count["meal"] += 1
            logger.info(f"   🍽️ {role} 추가: {food_name} ({category})")

    # 행렬 및 배열 재생성
    pool_matrices = {r: new_pools[r][NUM_COLS].values for r in new_pools}
    pool_display_names = {r: new_pools[r]["menuName"].values for r in new_pools}
    pool_cats = {r: new_pools[r]["category"].values for r in new_pools}
    pool_allergies = {
        r: new_pools[r]["allergy"].fillna("").astype(str).values for r in new_pools
    }

    # gene_space 재생성
    gene_space = [list(range(len(new_pools[r]))) for r in ROLE_ORDER]

    # 기본 쌀밥 인덱스
    default_rice_idx = base_ctx.default_rice_idx

    logger.info("=" * 60)
    logger.info(f"✅ 신메뉴 병합 완료")
    logger.info(f"   - 디저트 추가: {added_count['dessert']}개")
    logger.info(f"   - 일반 메뉴 추가: {added_count['meal']}개")
    logger.info(f"   - 총 디저트 풀: {len(new_dessert_pool)}개")
    for role, pool in new_pools.items():
        logger.info(f"   - {role}: {len(pool)}개")
    logger.info("=" * 60)

    return FoodContext(
        ready=True,
        pools=new_pools,
        pool_matrices=pool_matrices,
        pool_display_names=pool_display_names,
        pool_cats=pool_cats,
        pool_allergies=pool_allergies,
        default_rice_idx=default_rice_idx,
        gene_space=gene_space,
        source=base_ctx.source + " + new_menus",
        dessert_pool=new_dessert_pool,
        dessert_allergies=new_dessert_allergies,
        last_error=None,
        load_timestamp=base_ctx.load_timestamp,
        memory_size_mb=base_ctx.memory_size_mb,
    )
