"""
트렌드 분석 서비스

네이버 쇼핑인사이트 식품 카테고리 트렌드 분석
"""

from __future__ import annotations

import logging
import os
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

# 재료/원재료 키워드 (메뉴가 아닌 것들)
INGREDIENT_KEYWORDS = [
    "버터",
    "피스타치오",
    "아몬드",
    "호두",
    "캐슈넛",
    "땅콩",
    "밀가루",
    "설탕",
    "소금",
    "후추",
    "올리브오일",
    "참기름",
    "고춧가루",
    "마늘",
    "양파",
    "대파",
    "생강",
    "간장",
    "된장",
    "고추장",
    "카다이프",
    "생크림",
    "우유",
    "치즈",
    "계란",
]


class TrendAnalyzer:
    """네이버 쇼핑인사이트 트렌드 분석기"""

    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY가 설정되지 않았습니다.")

        self.model = "gpt-5-mini"

    async def fetch_naver_trends(self, days: int = 7) -> List[str]:
        """
        네이버 데이터랩 쇼핑인사이트에서 식품 카테고리 트렌드 키워드 수집

        Args:
            days: 분석 기간 (일)

        Returns:
            트렌드 키워드 리스트 (최대 20개)
        """
        try:
            import requests
        except ImportError:
            logger.error("requests 패키지가 필요합니다.")
            return []

        if not self.client_id or not self.client_secret:
            logger.warning("네이버 API 키가 설정되지 않았습니다. 샘플 데이터를 반환합니다.")
            return self._get_sample_trends()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        url = "https://openapi.naver.com/v1/datalab/shopping/categories"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json",
        }

        # 식품 카테고리 코드 (네이버 쇼핑인사이트 기준)
        body = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "timeUnit": "date",
            "category": [
                {"name": "식품", "param": ["50000000"]},  # 식품 대분류
            ],
            "device": "",
            "gender": "",
            "ages": [],
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                return self._extract_keywords_from_response(data)
            else:
                logger.warning(f"네이버 API 응답 오류: {resp.status_code}")
                return self._get_sample_trends()

        except Exception as e:
            logger.error(f"네이버 트렌드 API 호출 실패: {e}")
            return self._get_sample_trends()

    def _extract_keywords_from_response(self, data: Dict[str, Any]) -> List[str]:
        """API 응답에서 키워드 추출"""
        keywords = []

        results = data.get("results", [])
        for result in results:
            title = result.get("title", "")
            if title:
                keywords.append(title)

            # 세부 데이터가 있으면 추출
            sub_data = result.get("data", [])
            for item in sub_data:
                keyword = item.get("keyword", "")
                if keyword:
                    keywords.append(keyword)

        return list(set(keywords))[:20]

    def _get_sample_trends(self) -> List[str]:
        """샘플 트렌드 키워드 (API 실패 시 또는 테스트용)"""
        return [
            "두바이초콜릿",
            "로제떡볶이",
            "마라탕",
            "크로플",
            "약과",
            "탕후루",
            "소금빵",
            "대왕카스테라",
            "먹태",
            "로제파스타",
            "마라샹궈",
            "매콤닭갈비",
            "치즈볼",
            "붕어빵",
            "호떡",
        ]

    async def filter_menu_keywords(self, keywords: List[str]) -> List[str]:
        """
        재료/원재료를 필터링하고 "메뉴/제품"인 키워드만 선별

        Args:
            keywords: 트렌드 키워드 리스트

        Returns:
            필터링된 메뉴 키워드 리스트
        """
        # 1단계: 재료 키워드 제외
        filtered = [
            kw
            for kw in keywords
            if not any(ing in kw for ing in INGREDIENT_KEYWORDS)
        ]

        if not filtered:
            return []

        # 2단계: LLM으로 메뉴인지 판별
        if self.openai_client is None:
            logger.warning("OpenAI 클라이언트 없음. 필터링된 키워드 그대로 반환")
            return filtered

        try:
            prompt = f"""
다음 키워드 목록에서 "음식 메뉴/요리/제품"인 것만 선별해주세요.
재료(밀가루, 버터 등), 브랜드명, 일반 명사는 제외합니다.

키워드 목록:
{json.dumps(filtered, ensure_ascii=False)}

JSON 형식으로 메뉴 키워드만 배열로 반환하세요:
{{"menus": ["메뉴1", "메뉴2", ...]}}
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
            menus = result.get("menus", [])

            logger.info(f"트렌드 메뉴 필터링: {len(keywords)} → {len(menus)}개")
            return menus

        except Exception as e:
            logger.error(f"메뉴 키워드 필터링 실패: {e}")
            return filtered

    async def analyze(self, days: int = 7) -> List[str]:
        """
        트렌드 분석 메인 함수

        Args:
            days: 분석 기간 (일)

        Returns:
            트렌드 메뉴 키워드 리스트
        """
        logger.info(f"트렌드 분석 시작 (기간: {days}일)")

        # 1. 네이버 트렌드 키워드 수집
        raw_keywords = await self.fetch_naver_trends(days)
        logger.info(f"수집된 트렌드 키워드: {len(raw_keywords)}개")

        # 2. 메뉴 키워드만 필터링
        menu_keywords = await self.filter_menu_keywords(raw_keywords)
        logger.info(f"필터링된 메뉴 키워드: {len(menu_keywords)}개")

        return menu_keywords
