from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from app.services.food_loader import get_context, get_context_stats
from app.services.cost_loader import get_cost_stats, reload_cost_db


router = APIRouter()


@router.get("/health")
def health_check():
    """기본 헬스체크"""
    try:
        ctx = get_context()
        total_menus = sum(len(p) for p in ctx.pools.values())
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "dataset_loaded": True,
            "total_menus": total_menus,
            "memory_mb": round(ctx.memory_size_mb, 2) if ctx.memory_size_mb else None,
            "load_timestamp": ctx.load_timestamp,
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "dataset_loaded": False,
            "error": str(e),
        }


@router.get("/debug/memory")
def debug_memory():
    """메모리에 로드된 데이터 상세 정보"""
    try:
        stats = get_context_stats()
        return stats
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/debug/menus")
def debug_menus(
    role: Optional[str] = Query(
        default=None, description="역할 (밥, 국, 주찬, 부찬, 김치, 디저트)"
    ),
    limit: int = Query(default=10, ge=1, le=100, description="조회할 메뉴 수"),
):
    """특정 역할의 메뉴 목록 조회"""
    try:
        ctx = get_context()

        if role:
            # 특정 역할 조회
            if role not in ctx.pools:
                return {
                    "status": "error",
                    "error": f"'{role}' 역할을 찾을 수 없습니다",
                    "available_roles": list(ctx.pools.keys()),
                }

            pool = ctx.pools[role]
            menus = pool.head(limit).to_dict("records")

            return {
                "role": role,
                "total": len(pool),
                "showing": len(menus),
                "menus": menus,
            }
        else:
            # 전체 역할 요약
            summary = {}
            for r, pool in ctx.pools.items():
                summary[r] = {
                    "count": len(pool),
                    "sample": (
                        pool["menuName"].head(3).tolist() if not pool.empty else []
                    ),
                }

            return {
                "available_roles": list(ctx.pools.keys()),
                "role_summary": summary,
                "total_menus": sum(len(p) for p in ctx.pools.values()),
                "dessert_count": len(ctx.dessert_pool),
            }

    except RuntimeError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"예상치 못한 오류: {str(e)}"}


@router.get("/debug/search")
def debug_search(
    query: str = Query(..., min_length=1, description="검색할 메뉴명"),
    limit: int = Query(default=20, ge=1, le=100, description="최대 결과 수"),
):
    """메뉴명으로 검색"""
    try:
        ctx = get_context()

        results = []
        query_lower = query.lower()

        # 모든 풀에서 검색
        for role, pool in ctx.pools.items():
            if pool.empty:
                continue

            # menuName 컬럼에서 검색
            matches = pool[
                pool["menuName"].str.lower().str.contains(query_lower, na=False)
            ]

            for _, row in matches.iterrows():
                results.append(
                    {
                        "role": role,
                        "menuName": row.get("menuName", ""),
                        "category": row.get("category", ""),
                        "kcal": row.get("kcal", 0),
                        "protein": row.get("protein", 0),
                        "allergy": row.get("allergy", ""),
                    }
                )

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        # 디저트 풀에서도 검색
        dessert_matches = [
            {"role": "디저트", "menuName": d, "category": "디저트/음료"}
            for d in ctx.dessert_pool
            if query_lower in d.lower()
        ]
        results.extend(dessert_matches[: limit - len(results)])

        return {"query": query, "found": len(results), "results": results[:limit]}

    except RuntimeError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"검색 중 오류: {str(e)}"}


@router.get("/debug/dataset/stats")
def debug_dataset_stats():
    """데이터셋 통계 정보"""
    return get_context_stats()


@router.get("/debug/system")
def debug_system():
    """시스템 리소스 및 메모리 사용량"""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        return {
            "process_info": {
                "pid": os.getpid(),
                "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "memory_percent": round(process.memory_percent(), 2),
                "cpu_percent": process.cpu_percent(interval=0.1),
            },
            "system_info": {
                "total_memory_mb": round(
                    psutil.virtual_memory().total / 1024 / 1024, 2
                ),
                "available_memory_mb": round(
                    psutil.virtual_memory().available / 1024 / 1024, 2
                ),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_count": psutil.cpu_count(),
            },
        }
    except ImportError:
        return {
            "status": "error",
            "error": "psutil 패키지가 설치되지 않았습니다. pip install psutil",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/debug/costs")
def debug_costs():
    """단가 DB 정보"""
    return get_cost_stats()


@router.post("/debug/costs/reload")
def reload_costs():
    """단가 DB 재로드"""
    cost_db = reload_cost_db()
    return {"status": "reloaded", "total_menus": len(cost_db)}
