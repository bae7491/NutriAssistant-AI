"""
신메뉴 생성 서비스

게시판 피드백과 네이버 트렌드 분석을 통해 신메뉴 생성
"""

from __future__ import annotations

import logging
import os
import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from app.services.trend_analyzer import TrendAnalyzer
from app.services.board_analyzer import BoardAnalyzer
from app.services.food_loader import get_context
from app.models.schemas import (
    NewMenuItem,
    NutritionInfo,
    AnalysisSummary,
    NewMenuGenerationResponse,
)

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)


# 알레르기 번호 매핑
ALLERGEN_MAP = {
    "난류": 1,
    "계란": 1,
    "우유": 2,
    "메밀": 3,
    "땅콩": 4,
    "대두": 5,
    "밀": 6,
    "고등어": 7,
    "게": 8,
    "새우": 9,
    "돼지고기": 10,
    "복숭아": 11,
    "토마토": 12,
    "아황산염": 13,
    "호두": 14,
    "닭고기": 15,
    "쇠고기": 16,
    "오징어": 17,
    "조개류": 18,
    "잣": 19,
}

# 유효한 카테고리 목록
VALID_CATEGORIES = [
    "밥류",
    "국 및 탕류",
    "스프류",
    "전·적 및 부침류",
    "나물·숙채류",
    "디저트류",
    "볶음류",
    "구이류",
    "생채·무침류",
    "튀김류",
    "조림류",
    "찜류",
    "면류",
    "찌개 및 전골류",
    "죽류",
    "장아찌·절임류",
    "김치류",
    "음료류",
    "만두류",
]

# 카테고리 키워드 매핑 (fallback용)
CATEGORY_KEYWORDS = {
    "디저트류": [
        "쿠키",
        "케이크",
        "빵",
        "와플",
        "크로플",
        "마카롱",
        "약과",
        "호떡",
        "붕어빵",
        "도넛",
        "초콜릿",
        "탕후루",
        "타르트",
        "푸딩",
    ],
    "음료류": ["주스", "스무디", "라떼", "에이드", "차", "음료"],
    "면류": ["파스타", "라면", "우동", "짜장", "짬뽕", "국수", "냉면", "소바"],
    "밥류": ["덮밥", "비빔밥", "볶음밥", "김밥", "초밥", "주먹밥"],
    "국 및 탕류": ["탕", "국", "마라탕", "곰탕", "설렁탕"],
    "찌개 및 전골류": ["찌개", "전골", "부대찌개"],
    "스프류": ["스프", "수프", "포타주"],
    "볶음류": ["볶음", "떡볶이", "마라샹궈"],
    "튀김류": ["튀김", "까스", "돈까스", "치킨", "텐동", "가라아게"],
    "구이류": ["구이", "스테이크", "불고기", "바베큐"],
    "찜류": ["찜", "갈비찜", "아구찜"],
    "조림류": ["조림", "장조림"],
    "전·적 및 부침류": ["전", "부침", "빈대떡", "파전"],
    "나물·숙채류": ["나물", "숙채"],
    "생채·무침류": ["무침", "샐러드", "생채"],
    "만두류": ["만두", "교자", "딤섬"],
    "죽류": ["죽", "미음"],
    "김치류": ["김치"],
    "장아찌·절임류": ["장아찌", "절임", "피클"],
}


class NewMenuService:
    """신메뉴 생성 오케스트레이터"""

    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.board_analyzer = BoardAnalyzer()

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.model = "gpt-5-mini"

    async def generate_new_menus(
        self,
        use_trend: bool = True,
        use_board: bool = True,
        trend_days: int = 30,
        count: int = 2,
    ) -> NewMenuGenerationResponse:
        """
        신메뉴 생성 메인 함수

        Args:
            use_trend: 트렌드 분석 사용 여부
            use_board: 게시판 분석 사용 여부
            trend_days: 트렌드 분석 기간 (일)
            count: 생성할 신메뉴 수

        Returns:
            NewMenuGenerationResponse
        """
        logger.info("=" * 60)
        logger.info("🍽️ 신메뉴 생성 시작")
        logger.info(f"   - 트렌드 분석: {use_trend}")
        logger.info(f"   - 게시판 분석: {use_board}")
        logger.info(f"   - 생성 개수: {count}")
        logger.info("=" * 60)

        trend_keywords: List[str] = []
        board_menus: List[str] = []
        board_votes: Dict[str, int] = {}

        # 1. 트렌드 분석
        if use_trend:
            trend_keywords = await self.trend_analyzer.analyze(days=trend_days)
            logger.info(f"📈 트렌드 키워드: {trend_keywords[:5]}...")

        # 2. 게시판 분석
        if use_board:
            board_menus, board_votes = await self.board_analyzer.analyze(days=30)
            logger.info(f"📋 게시판 메뉴: {board_menus[:5]}...")

        # 3. 후보 통합 및 중복 제거
        candidates = self._merge_candidates(trend_keywords, board_menus)
        logger.info(f"📝 통합 후보: {len(candidates)}개")

        # 4. 기존 메뉴 풀과 매칭하여 중복 제거
        candidates = self._filter_existing_menus(candidates)
        logger.info(f"🔍 기존 메뉴 제외 후: {len(candidates)}개")

        # 5. 상위 N개 선정
        final_candidates = candidates[:count]

        # 6. 일괄 처리로 레시피/영양정보 생성 (LLM 호출 1회로 최적화)
        candidate_sources = [
            (c, self._determine_source(c, trend_keywords, board_menus))
            for c in final_candidates
        ]
        new_menus = await self._generate_menu_details_batch(candidate_sources)

        logger.info(f"✅ 생성된 신메뉴: {len(new_menus)}개")

        # 7. 응답 생성
        analysis_summary = AnalysisSummary(
            trend_keywords=trend_keywords[:10],
            board_votes=board_votes,
            total_candidates=len(candidates),
            final_count=len(new_menus),
        )

        return NewMenuGenerationResponse(
            generated_at=datetime.now().isoformat(),
            new_menus=new_menus,
            analysis_summary=analysis_summary,
        )

    def _merge_candidates(
        self, trend_keywords: List[str], board_menus: List[str]
    ) -> List[str]:
        """트렌드와 게시판 후보 통합"""
        # 게시판 메뉴를 우선순위로 (사용자 직접 요청이므로)
        merged = list(board_menus)

        # 트렌드 키워드 중 중복되지 않는 것 추가
        for kw in trend_keywords:
            if kw not in merged:
                merged.append(kw)

        return merged

    def _filter_existing_menus(self, candidates: List[str]) -> List[str]:
        """기존 메뉴 풀에 있는 메뉴 제외"""
        try:
            ctx = get_context()
            existing_menus = set()

            # 모든 역할의 메뉴명 수집
            for names in ctx.pool_display_names.values():
                existing_menus.update(str(n).strip() for n in names)

            # 디저트 풀도 추가
            existing_menus.update(ctx.dessert_pool)

            # 기존 메뉴와 완전 일치하는 것만 제외 (부분 일치는 허용)
            filtered = []
            for candidate in candidates:
                candidate_clean = candidate.strip()
                if candidate_clean not in existing_menus:
                    filtered.append(candidate_clean)
                else:
                    logger.debug(f"기존 메뉴 제외: {candidate_clean}")

            return filtered

        except RuntimeError:
            logger.warning("FoodContext를 사용할 수 없습니다. 필터링 없이 진행합니다.")
            return candidates

    def _determine_source(
        self, candidate: str, trend_keywords: List[str], board_menus: List[str]
    ) -> Literal["trend", "board", "hybrid"]:
        """메뉴 출처 결정"""
        in_trend = candidate in trend_keywords
        in_board = candidate in board_menus

        if in_trend and in_board:
            return "hybrid"
        elif in_board:
            return "board"
        else:
            return "trend"

    def _infer_category(self, menu_name: str) -> str:
        """메뉴명으로 카테고리 추론 (유효한 카테고리만 반환)"""
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in menu_name:
                    return category
        # 기본값으로 디저트류 반환 (신메뉴는 대부분 디저트/트렌드 음식)
        return "디저트류"

    def _validate_category(self, category: str) -> str:
        """카테고리 유효성 검증 및 보정"""
        if category in VALID_CATEGORIES:
            return category
        # 유사 카테고리 찾기
        for valid_cat in VALID_CATEGORIES:
            if category in valid_cat or valid_cat in category:
                return valid_cat
        return "디저트류"

    def _clean_ingredients(self, ingredients: List[str]) -> List[str]:
        """재료 목록에서 수량/중량 제거"""
        import re

        cleaned = []
        for ing in ingredients:
            # 숫자+단위 패턴 제거 (예: 200g, 1컵, 2큰술 등)
            clean = re.sub(
                r"\d+(\.\d+)?\s*(g|kg|ml|L|컵|큰술|작은술|개|장|조각|줌|꼬집|적당량)?",
                "",
                str(ing),
            )
            clean = clean.strip().strip(",").strip()
            if clean:
                cleaned.append(clean)
        return cleaned

    async def _generate_menu_details_batch(
        self, candidates: List[tuple]
    ) -> List[NewMenuItem]:
        """
        LLM 일괄 호출로 여러 메뉴의 상세 정보 생성 (성능 최적화)

        Args:
            candidates: [(메뉴명, source), ...] 리스트

        Returns:
            NewMenuItem 리스트
        """
        if not candidates:
            return []

        if self.openai_client is None:
            return [
                self._generate_fallback_menu(name, source)
                for name, source in candidates
            ]

        menu_names = [name for name, _ in candidates]
        source_map = {name: source for name, source in candidates}

        try:
            prompt = f"""
다음 메뉴들에 대한 상세 정보를 한번에 생성해주세요.

메뉴 목록: {json.dumps(menu_names, ensure_ascii=False)}

아래 JSON 형식으로 응답해주세요:
{{
    "menus": [
        {{
            "menu_name": "메뉴명",
            "category": "카테고리",
            "ingredients": ["재료1", "재료2", ...],
            "recipe_steps": ["1. 단계1", "2. 단계2", ...],
            "allergens": [알레르기번호들],
            "nutrition": {{
                "kcal": 숫자,
                "carbs": 숫자,
                "protein": 숫자,
                "fat": 숫자,
                "calcium": 숫자,
                "iron": 숫자,
                "vitamin_a": 숫자,
                "thiamin": 숫자,
                "riboflavin": 숫자,
                "vitamin_c": 숫자,
                "serving_basis": "100ml",
                "food_weight": 숫자
            }},
            "confidence": 0.0~1.0
        }},
        ...
    ]
}}

[category 규칙]
반드시 아래 목록 중 하나만 선택:
밥류, 국 및 탕류, 스프류, 전·적 및 부침류, 나물·숙채류, 디저트류, 볶음류, 구이류, 생채·무침류, 튀김류, 조림류, 찜류, 면류, 찌개 및 전골류, 죽류, 장아찌·절임류, 김치류, 음료류, 만두류

[ingredients 규칙]
- 재료명만 포함 (예: "밀가루", "설탕")
- 수량/중량 제외

[recipe_steps 규칙]
- 단계 수는 반드시 5~6단계로 제한할 것
- 각 단계는 한 문장으로 간결하게 작성할 것
- 불필요하게 세분화하지 말 것
- 조리 핵심만 포함할 것
- 한 단계당 30자 이내로 작성할 것

[nutrition 규칙]
- kcal: 칼로리(kcal)
- carbs: 탄수화물(g)
- protein: 단백질(g)
- fat: 지방(g)
- calcium: 칼슘(mg)
- iron: 철분(mg)
- vitamin_a: 비타민A(μg RAE)
- thiamin: 티아민/비타민B1(mg)
- riboflavin: 리보플라빈/비타민B2(mg)
- vitamin_c: 비타민C(mg)
- serving_basis: 영양성분 기준량 (항상 100ml로 고정)
- food_weight: 1인분 식품 중량 (항상 ml를 붙일 것. 예: 150ml)

[allergens 번호]
1=난류, 2=우유, 3=메밀, 4=땅콩, 5=대두, 6=밀, 7=고등어, 8=게, 9=새우, 10=돼지고기,
11=복숭아, 12=토마토, 13=아황산염, 14=호두, 15=닭고기, 16=쇠고기, 17=오징어, 18=조개류, 19=잣
""".strip()

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            menus_data = result.get("menus", [])

            new_menus: List[NewMenuItem] = []
            for item in menus_data:
                menu_name = item.get("menu_name", "")
                if not menu_name:
                    continue

                validated_category = self._validate_category(item.get("category", ""))
                cleaned_ingredients = self._clean_ingredients(
                    item.get("ingredients", [])
                )
                matched_menu = self._find_similar_menu(menu_name, validated_category)
                source = source_map.get(menu_name, "board")

                # 영양 정보 기본값 설정
                default_nutrition = {
                    "kcal": 300,
                    "carbs": 40,
                    "protein": 10,
                    "fat": 10,
                    "calcium": 20,
                    "iron": 1.0,
                    "vitamin_a": 10,
                    "thiamin": 0.1,
                    "riboflavin": 0.1,
                    "vitamin_c": 5,
                    "serving_basis": "100ml",
                    "food_weight": "150ml",
                }
                nutrition_data = {**default_nutrition, **item.get("nutrition", {})}

                new_menus.append(
                    NewMenuItem(
                        menu_name=menu_name,
                        category=validated_category,
                        source=source,
                        ingredients=cleaned_ingredients,
                        recipe_steps=item.get("recipe_steps", []),
                        allergens=item.get("allergens", []),
                        nutrition=NutritionInfo(**nutrition_data),
                        matched_menu=matched_menu,
                        confidence=item.get("confidence", 0.7),
                    )
                )

            logger.info(f"일괄 처리 완료: {len(new_menus)}개 메뉴 생성")
            return new_menus

        except Exception as e:
            logger.error(f"일괄 메뉴 상세 정보 생성 실패: {e}")
            return [
                self._generate_fallback_menu(name, source)
                for name, source in candidates
            ]

    async def _generate_menu_details(
        self, menu_name: str, source: Literal["trend", "board", "hybrid"]
    ) -> Optional[NewMenuItem]:
        """
        LLM을 사용하여 메뉴 상세 정보 생성

        Args:
            menu_name: 메뉴명
            source: 출처

        Returns:
            NewMenuItem 또는 None
        """
        if self.openai_client is None:
            # OpenAI 없이 기본 정보로 생성
            return self._generate_fallback_menu(menu_name, source)

        try:
            prompt = f"""
다음 메뉴에 대한 상세 정보를 생성해주세요.

메뉴명: {menu_name}

아래 JSON 형식으로 응답해주세요:
{{
    "category": "카테고리",
    "ingredients": ["재료1", "재료2", "재료3", ...],
    "recipe_steps": ["1. 첫번째 단계", "2. 두번째 단계", ...],
    "allergens": [알레르기 번호들],
    "nutrition": {{
        "kcal": 칼로리(정수),
        "carbs": 탄수화물g(정수),
        "protein": 단백질g(정수),
        "fat": 지방g(정수)
    }},
    "confidence": 0.0~1.0 사이 신뢰도
}}

[category 규칙]
반드시 아래 목록 중 하나만 선택:
밥류, 국 및 탕류, 스프류, 전·적 및 부침류, 나물·숙채류, 디저트류, 볶음류, 구이류, 생채·무침류, 튀김류, 조림류, 찜류, 면류, 찌개 및 전골류, 죽류, 장아찌·절임류, 김치류, 음료류, 만두류

[ingredients 규칙]
- 재료명만 포함 (예: "밀가루", "설탕", "버터")
- 수량/중량 제외 (예: "밀가루 200g" → "밀가루")

[recipe_steps 규칙]
- 단계 수는 반드시 5~6단계로 제한할 것
- 각 단계는 한 문장으로 간결하게 작성할 것
- 불필요하게 세분화하지 말 것
- 조리 핵심만 포함할 것
- 한 단계당 30자 이내로 작성할 것

[allergens 번호]
1=난류, 2=우유, 3=메밀, 4=땅콩, 5=대두, 6=밀, 7=고등어, 8=게, 9=새우, 10=돼지고기,
11=복숭아, 12=토마토, 13=아황산염, 14=호두, 15=닭고기, 16=쇠고기, 17=오징어, 18=조개류, 19=잣
""".strip()

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # category 유효성 검증
            raw_category = result.get("category", "")
            validated_category = self._validate_category(raw_category)

            # ingredients 정제 (수량/중량 제거)
            raw_ingredients = result.get("ingredients", [])
            cleaned_ingredients = self._clean_ingredients(raw_ingredients)

            # 영양 매칭
            matched_menu = self._find_similar_menu(menu_name, validated_category)

            # 영양 정보 기본값 설정
            default_nutrition = {
                "kcal": 300,
                "carbs": 40,
                "protein": 10,
                "fat": 10,
                "calcium": 20,
                "iron": 1.0,
                "vitamin_a": 10,
                "thiamin": 0.1,
                "riboflavin": 0.1,
                "vitamin_c": 5,
                "serving_basis": "100ml",
                "food_weight": "150ml",
            }
            nutrition_data = {**default_nutrition, **result.get("nutrition", {})}

            return NewMenuItem(
                menu_name=menu_name,
                category=validated_category,
                source=source,
                ingredients=cleaned_ingredients,
                recipe_steps=result.get("recipe_steps", []),
                allergens=result.get("allergens", []),
                nutrition=NutritionInfo(**nutrition_data),
                matched_menu=matched_menu,
                confidence=result.get("confidence", 0.7),
            )

        except Exception as e:
            logger.error(f"메뉴 상세 정보 생성 실패 ({menu_name}): {e}")
            return self._generate_fallback_menu(menu_name, source)

    def _generate_fallback_menu(
        self, menu_name: str, source: Literal["trend", "board", "hybrid"]
    ) -> NewMenuItem:
        """OpenAI 없이 기본 정보로 메뉴 생성"""
        category = self._infer_category(menu_name)
        matched_menu = self._find_similar_menu(menu_name, category)

        return NewMenuItem(
            menu_name=menu_name,
            category=category,
            source=source,
            ingredients=["재료 정보 없음"],
            recipe_steps=["레시피 정보 없음"],
            allergens=[],
            nutrition=NutritionInfo(
                kcal=300,
                carbs=40,
                protein=10,
                fat=10,
                calcium=20,
                iron=1.0,
                vitamin_a=10,
                thiamin=0.1,
                riboflavin=0.1,
                vitamin_c=5,
                serving_basis="100ml",
                food_weight="150ml",
            ),
            matched_menu=matched_menu,
            confidence=0.5,
        )

    def _find_similar_menu(self, menu_name: str, category: str) -> Optional[str]:
        """
        FoodContext에서 유사 메뉴 찾기 (영양 정보 추정용)

        Args:
            menu_name: 신메뉴명
            category: 카테고리

        Returns:
            유사 메뉴명 또는 None
        """
        try:
            ctx = get_context()

            # 카테고리로 필터링하여 후보 찾기
            for role, cats in ctx.pool_cats.items():
                names = ctx.pool_display_names[role]

                for i, cat in enumerate(cats):
                    # 카테고리가 유사하면 첫 번째 메뉴 반환
                    if category in str(cat) or str(cat) in category:
                        return str(names[i])

            # 카테고리 매칭 실패 시 키워드로 검색
            for role, names in ctx.pool_display_names.items():
                for name in names:
                    name_str = str(name)
                    # 메뉴명에 공통 키워드가 있으면 반환
                    for kw in ["파스타", "볶음", "탕", "까스", "쿠키", "케이크"]:
                        if kw in menu_name and kw in name_str:
                            return name_str

            return None

        except RuntimeError:
            return None
