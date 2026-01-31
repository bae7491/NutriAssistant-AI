import os

# 내부 토큰(선택): Spring에서만 호출하게 하고 싶으면 사용
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")  # 비우면 검사 안 함

# Spring 내부 API (예: http://localhost:8080/internal/food)
# 통합 음식 DB 제공 API 엔드포인트
SPRING_FOOD_API = os.getenv("SPRING_FOOD_API", "http://localhost:8080/internal/food")

# Spring 게시판 API (신메뉴 요청 피드백 조회용)
SPRING_BOARD_API = os.getenv(
    "SPRING_BOARD_API", "http://localhost:8080/new-menu/internal/feedback"
)

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

# =========================================================
# 영양 기준 DB (학교급/성별별)
# =========================================================
NUTRITION_STANDARDS_DB = {
    "초등": {
        "kcal": 670,
        "prot": 16.7,
        "vitA": 137,
        "vitC": 18.4,
        "ca": 217,
        "fe": 2.7,
    },
    "중학_남": {
        "kcal": 840,
        "prot": 20.0,
        "vitA": 177,
        "vitC": 23.4,
        "ca": 267,
        "fe": 3.7,
    },
    "중학_여": {
        "kcal": 670,
        "prot": 18.4,
        "vitA": 160,
        "vitC": 23.4,
        "ca": 250,
        "fe": 4.0,
    },
    "고등_남": {
        "kcal": 900,
        "prot": 21.7,
        "vitA": 207,
        "vitC": 26.7,
        "ca": 250,
        "fe": 3.7,
    },
    "고등_여": {
        "kcal": 670,
        "prot": 18.4,
        "vitA": 150,
        "vitC": 26.7,
        "ca": 234,
        "fe": 3.7,
    },
}

# Spring ENUM → Python 영양 기준 키 매핑
NUTRITION_KEY_MAPPING = {
    # 초등학교
    "ELEMENTARY": "초등_평균",
    # 중학교
    "MIDDLE_MALE": "중학_남",
    "MIDDLE_FEMALE": "중학_여",
    "MIDDLE_COED": "중학_공학",
    # 고등학교
    "HIGH_MALE": "고등_남",
    "HIGH_FEMALE": "고등_여",
    "HIGH_COED": "고등_공학",
}


def get_nutrition_standard(nutrition_key: str = None) -> dict:
    """
    nutrition_key에 따른 영양 기준 반환

    Args:
        nutrition_key: Spring ENUM 값 (ELEMENTARY, MIDDLE_MALE, etc.)

    Returns:
        영양 기준 딕셔너리 {"kcal": ..., "prot": ..., ...}
    """
    if not nutrition_key:
        # 기본값: 고등_남
        return NUTRITION_STANDARDS_DB["고등_남"]

    key = nutrition_key.upper()

    # 초등학교: 남자 4~6학년 값 사용
    if key == "ELEMENTARY":
        return NUTRITION_STANDARDS_DB["초등"]

    # 중학교
    if key == "MIDDLE_MALE":
        return NUTRITION_STANDARDS_DB["중학_남"]

    if key == "MIDDLE_FEMALE":
        return NUTRITION_STANDARDS_DB["중학_여"]

    # 남녀공학 중학교: 남자 값 사용
    if key == "MIDDLE_COED":
        return NUTRITION_STANDARDS_DB["중학_남"]

    # 고등학교
    if key == "HIGH_MALE":
        return NUTRITION_STANDARDS_DB["고등_남"]

    if key == "HIGH_FEMALE":
        return NUTRITION_STANDARDS_DB["고등_여"]

    # 남녀공학 고등학교: 남자 값 사용
    if key == "HIGH_COED":
        return NUTRITION_STANDARDS_DB["고등_남"]

    # 알 수 없는 키: 기본값
    return NUTRITION_STANDARDS_DB["고등_남"]


# 기본 영양 기준 (하위 호환용)
STD_KCAL = float(os.getenv("STD_KCAL", "900.0"))
STD_PROT = float(os.getenv("STD_PROT", "25.0"))

# 칼로리 허용 오차 비율
KCAL_TOLERANCE_RATIO = 0.10
