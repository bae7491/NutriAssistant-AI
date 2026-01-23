import os

# 내부 토큰(선택): Spring에서만 호출하게 하고 싶으면 사용
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")  # 비우면 검사 안 함

# Spring 내부 API (예: http://localhost:8080/internal/food)
# 통합 음식 DB 제공 API 엔드포인트
SPRING_FOOD_API = os.getenv("SPRING_FOOD_API", "http://localhost:8080/internal/food")

# 페이지 크기 (Spring API가 Page 형태일 때)
SPRING_PAGE_SIZE = int(os.getenv("SPRING_PAGE_SIZE", "2000"))

# 요청 타임아웃(초)
SPRING_TIMEOUT_SECONDS = int(os.getenv("SPRING_TIMEOUT_SECONDS", "60"))

# -------------------------
# 로컬 파일(선택): 단가/가중치 DB
# -------------------------
WEIGHT_DB_PATH = os.getenv("WEIGHT_DB_PATH", "./data/menu_weights.json")
COST_DB_PATH = os.getenv("COST_DB_PATH", "./data/menu_costs.json")

# -------------------------
# 디저트/음료 카테고리(주 2회 랜덤, 비용/영양 제외)
# -------------------------
DESSERT_CATEGORIES = [
    s.strip()
    for s in os.getenv("DESSERT_CATEGORIES", "디저트류,음료류").split(",")
    if s.strip()
]
DESSERT_FREQUENCY_PER_WEEK = int(os.getenv("DESSERT_FREQUENCY_PER_WEEK", "2"))

# -------------------------
# 영양 컬럼(표준 키)
# -------------------------
NUM_COLS = [
    "kcal",
    "carbs",
    "protein",
    "fat",
    "calcium",
    "iron",
    "vitaminA",
    "vitaminC",
]

ROLE_RULES = {
    "밥": ["밥류", "면류", "죽류"],
    "국": ["국 및 탕류", "찌개 및 전골류", "스프류"],
    "주찬": [
        "구이류",
        "볶음류",
        "찜류",
        "조림류",
        "튀김류",
        "전·적 및 부침류",
        "만두류",
    ],
    "부찬": ["나물·숙채류", "생채·무침류", "장아찌·절임류"],
    "김치": ["김치류"],
}

# 6칸 고정(주찬 2개)
ROLE_ORDER = ["밥", "국", "주찬", "주찬", "부찬", "김치"]

# 기본 영양 기준(필요하면 요청 옵션으로 바꿔도 됨)
STD_KCAL = float(os.getenv("STD_KCAL", "900.0"))
STD_PROT = float(os.getenv("STD_PROT", "25.0"))
