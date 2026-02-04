from __future__ import annotations
import json
import logging
from typing import Dict, List
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class ReportAnalyzer:
    """리포트 분석 및 가중치 생성 (일회성)"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않았습니다")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = "gpt-4o-mini"

    async def analyze_report_to_weights(
        self, report_data: Dict, valid_menu_names: List[str]
    ) -> Dict[str, float]:
        """
        리포트를 분석하여 메뉴별 가중치 생성 (일회성)

        Args:
            report_data: 리포트 JSON 데이터
            valid_menu_names: DB에 존재하는 유효한 메뉴명 리스트

        Returns:
            메뉴별 가중치 딕셔너리 {메뉴명: 가중치(-10.0 ~ 10.0)}
        """
        if not self.client:
            logger.warning("⚠️ OpenAI 클라이언트 없음. 빈 가중치 반환")
            return {}

        logger.info("🤖 리포트 AI 분석 시작 (일회성)")

        # 리포트 핵심 정보 추출
        try:
            # 데이터 경로 탐색 (다양한 구조 지원)
            # 1) report_data.payload.data 구조
            # 2) report_data.data 구조
            # 3) report_data 자체가 data인 경우
            data = report_data.get("payload", {}).get("data", {})
            if not data:
                data = report_data.get("data", {})
            if not data:
                data = report_data

            # periodicAnalysis (camelCase) 또는 periodic_analysis (snake_case) 지원
            periodic = data.get("periodicAnalysis", {}) or data.get("periodic_analysis", {})
            trend = data.get("trendAnalysis", {}) or data.get("trend_analysis", {})

            # 문제 영역과 상세 분석 추출
            problem_areas = periodic.get("problem_areas", []) or periodic.get("problemAreas", [])
            deepdives = periodic.get("deepdives", []) or periodic.get("deepDives", [])

            # 트렌드 분석에서 선호/문제 카테고리 추출
            preferred_cats = trend.get("preferred_categories", []) or trend.get("preferredCategories", [])
            problem_cats = trend.get("problem_categories", []) or trend.get("problemCategories", [])

            # 추가 데이터 추출 (카테고리별 감정 분포, 불만 태그 등)
            category_sentiments = trend.get("category_sentiment_distribution", []) or trend.get("categorySentimentDistribution", [])
            category_complaints = trend.get("category_complaints", {}) or trend.get("categoryComplaints", {})

            # 리뷰 집계 데이터 추출
            reviews_aggregate = periodic.get("reviews", {})
            kpis = periodic.get("kpis", {})

            logger.info(f"   문제 영역: {len(problem_areas)}개")
            logger.info(f"   상세 분석: {len(deepdives)}개")
            logger.info(f"   선호 카테고리: {len(preferred_cats)}개")
            logger.info(f"   문제 카테고리: {len(problem_cats)}개")
            logger.info(f"   카테고리 감정 분포: {len(category_sentiments)}개")

        except Exception as e:
            logger.error(f"❌ 리포트 파싱 실패: {e}")
            return {}

        # AI 프롬프트 생성
        prompt = self._build_prompt(
            problem_areas=problem_areas,
            deepdives=deepdives,
            preferred_cats=preferred_cats,
            problem_cats=problem_cats,
            valid_menu_names=valid_menu_names,
            category_sentiments=category_sentiments,
            category_complaints=category_complaints,
            reviews_aggregate=reviews_aggregate,
            kpis=kpis,
        )

        # AI 분석
        try:
            logger.info("🔄 AI 분석 중...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            weights = self._extract_weights(result, valid_menu_names)

            # 가중치가 충분하지 않으면 기본 가중치로 보완
            if len(weights) < 10:
                logger.warning(f"⚠️ AI 가중치 부족 ({len(weights)}개). 기본 가중치로 보완합니다.")
                default_weights = self._generate_default_weights(valid_menu_names)
                for menu, weight in default_weights.items():
                    if menu not in weights:
                        weights[menu] = weight

            logger.info(f"✅ 가중치 생성 완료: {len(weights)}개 메뉴")

            # 0이 아닌 가중치 개수 확인
            non_zero_count = sum(1 for w in weights.values() if w != 0.0)
            logger.info(f"   0이 아닌 가중치: {non_zero_count}개")

            # 상위/하위 3개만 로깅
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            if sorted_weights:
                logger.info("📈 가중치 상위 3개:")
                for name, weight in sorted_weights[:3]:
                    logger.info(f"      {name}: {weight:+.2f}")
                logger.info("📉 가중치 하위 3개:")
                for name, weight in sorted_weights[-3:]:
                    logger.info(f"      {name}: {weight:+.2f}")

            return weights

        except Exception as e:
            logger.error(f"❌ AI 분석 실패: {e}")
            logger.info("📌 기본 가중치 생성으로 대체합니다.")
            return self._generate_default_weights(valid_menu_names)

    def _build_prompt(
        self,
        problem_areas: List[Dict],
        deepdives: List[Dict],
        preferred_cats: List[Dict],
        problem_cats: List[Dict],
        valid_menu_names: List[str],
        category_sentiments: List[Dict] = None,
        category_complaints: Dict = None,
        reviews_aggregate: Dict = None,
        kpis: Dict = None,
    ) -> str:
        """AI 프롬프트 생성"""
        category_sentiments = category_sentiments or []
        category_complaints = category_complaints or {}
        reviews_aggregate = reviews_aggregate or {}
        kpis = kpis or {}

        # 유효 메뉴 샘플만 포함 (토큰 절약)
        menu_sample = (
            valid_menu_names[:100] if len(valid_menu_names) > 100 else valid_menu_names
        )

        # 데이터가 부족한 경우 기본 지침 추가
        has_enough_data = bool(problem_areas or deepdives or preferred_cats or problem_cats or category_sentiments)

        prompt = f"""
급식 리포트를 분석하여 메뉴별 가중치를 생성해주세요.

[유효 메뉴 목록] (총 {len(valid_menu_names)}개)
{', '.join(menu_sample)}

[문제 영역] ({len(problem_areas)}개)
{json.dumps(problem_areas, ensure_ascii=False) if problem_areas else "데이터 없음"}

[상세 분석] ({len(deepdives)}개)
{json.dumps(deepdives[:3], ensure_ascii=False) if deepdives else "데이터 없음"}

[선호 카테고리] ({len(preferred_cats)}개)
{json.dumps(preferred_cats[:5], ensure_ascii=False) if preferred_cats else "데이터 없음"}

[문제 카테고리] ({len(problem_cats)}개)
{json.dumps(problem_cats[:5], ensure_ascii=False) if problem_cats else "데이터 없음"}

[카테고리별 감정 분포] ({len(category_sentiments)}개)
{json.dumps(category_sentiments[:10], ensure_ascii=False) if category_sentiments else "데이터 없음"}

[카테고리별 불만 태그]
{json.dumps(dict(list(category_complaints.items())[:5]), ensure_ascii=False) if category_complaints else "데이터 없음"}

---

**가중치 부여 기준**:
1. 긍정 (+3 ~ +10): 선호 카테고리에 속하는 메뉴, 긍정 비율 높은 카테고리
2. 부정 (-3 ~ -10): 문제 카테고리에 속하는 메뉴, 부정 비율 높은 카테고리, 불만 태그 많은 카테고리
3. 약한 긍정/부정 (+1 ~ +3 / -1 ~ -3): 간접적 연관성
4. 중립 (0): 분석 데이터에 언급되지 않은 메뉴
"""

        if not has_enough_data:
            prompt += """
**참고**: 분석 데이터가 부족합니다. 메뉴명을 기반으로 일반적인 급식 선호도를 고려하여 가중치를 부여해주세요.
- 인기 메뉴 (치킨, 돈까스, 떡볶이, 피자 등): +5 ~ +8
- 건강 메뉴 (샐러드, 나물 등): +1 ~ +3
- 비선호 메뉴 (특정 생선, 콩류 등): -1 ~ -3
"""

        prompt += """
**출력 형식** (JSON):
{
  "weights": [
    {"menu": "메뉴명1", "weight": 7.5},
    {"menu": "메뉴명2", "weight": -3.0}
  ]
}

**중요**:
- 메뉴명은 반드시 [유효 메뉴 목록]에 있는 것만 사용
- 최소 30개 이상의 메뉴에 가중치 부여
- 0이 아닌 가중치를 적극적으로 부여
"""
        return prompt

    def _get_system_prompt(self) -> str:
        return """
당신은 급식 데이터 분석 전문가입니다. 리포트를 분석하여 메뉴별 가중치를 생성합니다.

**가중치 범위**: -10 ~ +10
- +10: 매우 선호 (치킨, 피자 등 인기 메뉴)
- +5 ~ +9: 선호 (돈까스, 떡볶이 등)
- +1 ~ +4: 약간 선호
- 0: 중립 (특별한 정보 없음)
- -1 ~ -4: 약간 비선호
- -5 ~ -9: 비선호 (불만 많은 메뉴)
- -10: 매우 비선호

**중요 지침**:
1. 가능한 많은 메뉴에 0이 아닌 가중치를 부여하세요
2. 분석 데이터가 없어도 메뉴명에서 유추 가능한 경우 가중치를 부여하세요
3. 일반적인 급식 선호도를 참고하세요

JSON 형식으로만 출력하세요.
"""

    def _extract_weights(
        self, ai_result: Dict, valid_menu_names: List[str]
    ) -> Dict[str, float]:
        """AI 결과에서 가중치 추출"""

        weights = {}
        valid_set = set(valid_menu_names)

        for item in ai_result.get("weights", []):
            menu = item.get("menu", "").strip()
            weight = float(item.get("weight", 0.0))

            if menu and menu in valid_set:
                weights[menu] = max(-10.0, min(10.0, weight))

        return weights

    def _generate_default_weights(self, valid_menu_names: List[str]) -> Dict[str, float]:
        """메뉴명 기반 기본 가중치 생성"""

        # 선호 키워드와 가중치
        positive_keywords = {
            # 매우 선호 (+7 ~ +9)
            "치킨": 8.0, "피자": 8.0, "햄버거": 7.5, "스테이크": 8.0, "삼겹살": 8.0,
            "갈비": 8.0, "불고기": 7.5, "돈까스": 7.0, "탕수육": 7.5, "짜장": 6.5,
            "짬뽕": 6.5, "떡볶이": 6.5, "라면": 6.0, "우동": 5.5, "냉면": 6.0,
            # 선호 (+4 ~ +6)
            "카레": 5.0, "스파게티": 5.5, "파스타": 5.5, "볶음밥": 5.0, "덮밥": 4.5,
            "김밥": 5.0, "비빔밥": 4.5, "제육": 5.0, "오삼": 5.0, "닭": 5.5,
            "소고기": 5.5, "돼지": 4.5, "쫄면": 5.0, "만두": 5.0, "튀김": 4.5,
            # 약한 선호 (+2 ~ +3)
            "찌개": 3.0, "국밥": 3.0, "볶음": 2.5, "조림": 2.0, "구이": 3.0,
            "전": 3.0, "부침": 2.5, "탕": 2.5, "죽": 2.0,
        }

        # 비선호 키워드와 가중치
        negative_keywords = {
            # 비선호 (-3 ~ -5)
            "콩나물": -2.0, "숙주": -2.0, "시금치": -1.5, "무침": -1.0,
            "나물": -1.5, "미역": -1.0, "두부": -1.0, "비빔": 0.0,
            # 약한 비선호 (-1 ~ -2)
            "고등어": -1.5, "갈치": -1.0, "생선": -0.5, "조기": -1.0,
        }

        weights = {}
        for menu in valid_menu_names:
            weight = 0.0
            menu_lower = menu.lower()

            # 선호 키워드 확인
            for keyword, kw_weight in positive_keywords.items():
                if keyword in menu_lower or keyword in menu:
                    weight = max(weight, kw_weight)

            # 비선호 키워드 확인 (선호보다 낮은 우선순위)
            if weight == 0.0:
                for keyword, kw_weight in negative_keywords.items():
                    if keyword in menu_lower or keyword in menu:
                        weight = min(weight, kw_weight)

            weights[menu] = weight

        return weights
