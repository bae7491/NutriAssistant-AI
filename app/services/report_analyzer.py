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

        self.model = "gpt-5-mini"

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
            payload = report_data.get("payload", {})
            data = payload.get("data", {})

            periodic = data.get("periodic_analysis", {})
            trend = data.get("trendAnalysis", {})

            problem_areas = periodic.get("problem_areas", [])
            deepdives = periodic.get("deepdives", [])
            preferred_cats = trend.get("preferredCategories", [])
            problem_cats = trend.get("problemCategories", [])

            logger.info(f"   문제 영역: {len(problem_areas)}개")
            logger.info(f"   상세 분석: {len(deepdives)}개")

        except Exception as e:
            logger.error(f"❌ 리포트 파싱 실패: {e}")
            return {}

        # AI 프롬프트 생성
        prompt = self._build_prompt(
            problem_areas, deepdives, preferred_cats, problem_cats, valid_menu_names
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

            logger.info(f"✅ 가중치 생성 완료: {len(weights)}개 메뉴")

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
            return {}

    def _build_prompt(
        self,
        problem_areas: List[Dict],
        deepdives: List[Dict],
        preferred_cats: List[Dict],
        problem_cats: List[Dict],
        valid_menu_names: List[str],
    ) -> str:
        """AI 프롬프트 생성"""

        # 유효 메뉴 샘플만 포함 (토큰 절약)
        menu_sample = (
            valid_menu_names[:100] if len(valid_menu_names) > 100 else valid_menu_names
        )

        return f"""
급식 리포트를 분석하여 메뉴별 가중치를 생성해주세요.

[유효 메뉴 샘플] (총 {len(valid_menu_names)}개)
{', '.join(menu_sample)}

[문제 영역]
{json.dumps(problem_areas, ensure_ascii=False)}

[상세 분석]
{json.dumps(deepdives[:3], ensure_ascii=False)}

[선호 카테고리]
{json.dumps(preferred_cats[:5], ensure_ascii=False)}

[문제 카테고리]
{json.dumps(problem_cats[:5], ensure_ascii=False)}

---

**가중치 부여 기준**:
1. 긍정 (+5 ~ +10): 선호 카테고리, 높은 만족도, 긍정 리뷰 많음
2. 부정 (-5 ~ -10): 문제 카테고리, 높은 불만, 부정 리뷰 많음
3. 중립 (0): 언급 없음

**출력**:
{{
  "weights": [
    {{"menu": "메뉴명", "weight": 7.5}},
    {{"menu": "메뉴명", "weight": -6.0}}
  ]
}}

**제약**: 메뉴명은 [유효 메뉴 샘플]에 있는 것만 사용
"""

    def _get_system_prompt(self) -> str:
        return """
급식 데이터 분석가입니다. 리포트 기반 메뉴 가중치 생성.

가중치 범위: -10 ~ +10
- +10: 매우 선호
- 0: 중립
- -10: 매우 비선호

JSON만 출력.
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
