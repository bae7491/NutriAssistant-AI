import os
import json
import logging
from typing import Dict, List
from datetime import datetime
from openai import OpenAI

logger = logging.getLogger(__name__)


class CostGenerator:
    """AI 기반 메뉴 단가 생성기"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않았습니다")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = "gpt-4o-mini"
        self.base_year = 2023

        # 물가 상승률 (연도별)
        self.inflation_rates = {
            2023: 0.036,  # +3.6%
            2024: 0.023,  # +2.3%
            2025: 0.021,  # +2.1%
        }
        self.annual_rate = 0.022  # 기본 연 2.2%

    def _check_client(self):
        """OpenAI 클라이언트 확인"""
        if self.client is None:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. "
                "환경변수 OPENAI_API_KEY를 설정하세요."
            )

    def calculate_inflation_multiplier(self, target_year: int) -> float:
        """
        물가상승률 계산 (복리)

        Args:
            target_year: 목표 연도

        Returns:
            물가상승 계수 (예: 1.0821 = 8.21% 상승)
        """
        multiplier = 1.0

        logger.info(f"📈 물가 계산: {self.base_year}년 → {target_year}년")

        for year in range(self.base_year, target_year):
            rate = self.inflation_rates.get(year, self.annual_rate)
            multiplier *= 1 + rate
            logger.info(f"   - {year}년: +{rate*100:.1f}% (누적 {multiplier:.4f}배)")

        logger.info(f"   👉 최종 계수: {multiplier:.4f}배")
        return multiplier

    def generate_costs_batch(
        self, menu_names: List[str], batch_size: int = 40
    ) -> Dict[str, int]:
        """
        여러 메뉴의 2023년 기준 단가를 AI로 생성

        Args:
            menu_names: 메뉴명 리스트
            batch_size: 한 번에 처리할 메뉴 수 (기본 40개)

        Returns:
            {메뉴명: 2023년 단가(원)} 딕셔너리
        """
        self._check_client()

        logger.info(f"🤖 AI 단가 생성 시작: {len(menu_names)}개 메뉴")

        all_costs = {}
        batches = [
            menu_names[i : i + batch_size]
            for i in range(0, len(menu_names), batch_size)
        ]

        for i, batch in enumerate(batches):
            logger.info(f"   ⏳ ({i+1}/{len(batches)}) AI 조회 중...")

            try:
                costs = self._generate_costs_single_batch(batch)
                all_costs.update(costs)
            except Exception as e:
                logger.error(f"❌ Batch {i+1} 실패: {e}")
                # 실패한 메뉴는 기본값 1000원 설정
                for menu in batch:
                    if menu not in all_costs:
                        all_costs[menu] = 1000

        logger.info(f"✅ AI 단가 생성 완료: {len(all_costs)}개")
        return all_costs

    def _generate_costs_single_batch(self, menu_names: List[str]) -> Dict[str, int]:
        """
        단일 배치 처리 (2023년 기준가 생성)

        Args:
            menu_names: 메뉴명 리스트 (40개 이하)

        Returns:
            {메뉴명: 2023년 단가} 딕셔너리
        """
        prompt = f"""
급식 메뉴의 '1인분 식재료 원가(KRW)'를 {self.base_year}년 기준으로 추정해서 JSON으로 줘.

목록: {", ".join(menu_names)}

출력 형식 (JSON만):
{{
  "쌀밥": 180,
  "김치찌개": 600,
  "돈까스": 1800,
  ...
}}

주의사항:
- 급식용 1인분 기준
- 식재료비만 계산 (인건비, 설비비 제외)
- {self.base_year}년 기준 가격
- 정수(원) 단위로 반환
        """.strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"당신은 급식 식재료 원가 전문가입니다. {self.base_year}년 기준 가격을 JSON 형식으로만 응답하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        data = json.loads(result_text)

        # 다양한 응답 형식 처리
        if isinstance(data, dict):
            costs = data.get("data", data.get("prices", data))
        else:
            costs = {}

        # 정수 변환 및 검증
        cleaned_costs = {}
        for key, value in costs.items():
            clean_key = str(key).strip()
            try:
                price = int(float(value))
                # 최소값 100원, 최대값 50000원으로 제한
                price = max(100, min(50000, price))
                cleaned_costs[clean_key] = price
            except (ValueError, TypeError):
                logger.warning(
                    f"⚠️ 가격 변환 실패: {clean_key}={value}, 기본값 1000원 사용"
                )
                cleaned_costs[clean_key] = 1000

        return cleaned_costs

    def apply_inflation(
        self, base_costs: Dict[str, int], target_year: int
    ) -> Dict[str, int]:
        """
        2023년 기준 단가에 물가상승률 적용

        Args:
            base_costs: {메뉴명: 2023년 단가}
            target_year: 목표 연도 (예: 2026)

        Returns:
            {메뉴명: 목표연도 단가}
        """
        multiplier = self.calculate_inflation_multiplier(target_year)

        return {menu: int(price * multiplier) for menu, price in base_costs.items()}
