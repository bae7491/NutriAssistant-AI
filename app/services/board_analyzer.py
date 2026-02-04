"""
게시판 분석 서비스

Spring API 연동으로 신메뉴 요청 게시판 피드백 분석
"""

from __future__ import annotations

import logging
import os
import json
import re
from typing import List, Dict, Any, Tuple

from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from app.core.config import (
    INTERNAL_API_KEY,
    SPRING_BOARD_API,
    SPRING_TIMEOUT_SECONDS,
)

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None


# 메뉴명 추출용 패턴
MENU_PATTERNS = [
    r"([가-힣a-zA-Z]+(?:볶음|찌개|탕|국|전|구이|조림|무침|튀김|파스타|덮밥|비빔밥|커리|카레|라면|우동|짜장|짬뽕|피자|버거|샌드위치|스테이크|샐러드|수프|스프))",
    r"([가-힣]+(?:치킨|떡볶이|순대|어묵|김밥|초밥|롤|쿠키|케이크|빵|도넛|와플|크로플|마카롱|타르트|푸딩|젤리|아이스크림))",
    r"(마라[가-힣]+)",
    r"(로제[가-힣]+)",
]

# 메뉴명 약어 → 정식 명칭 매핑
ABBREVIATION_MAP = {
    "돈까": "돈까스",
    "돈가": "돈까스",
    "치까": "치킨까스",
    "생까": "생선까스",
    "함까": "함박스테이크",
    "치볶": "치즈볶음밥",
    "김볶": "김치볶음밥",
    "제볶": "제육볶음",
    "오삼": "오삼불고기",
    "짜파게티": "짜파게티",
    "짜게티": "짜파게티",
    "불닭": "불닭볶음면",
    "떡볶": "떡볶이",
    "순대볶": "순대볶음",
}


class BoardAnalyzer:
    """게시판 피드백 분석기"""

    def __init__(self):
        if requests is None:
            raise RuntimeError("requests 패키지가 필요합니다. `pip install requests`")

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            logger.warning(
                "OPENAI_API_KEY가 설정되지 않았습니다. 패턴 기반 추출만 사용합니다."
            )

        self.model = "gpt-5-mini"

    async def fetch_board_feedback(
        self, days: int = 30, size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Spring API에서 신메뉴 요청 게시판 데이터 가져오기

        Args:
            days: 조회 기간 (일)
            size: 조회 개수

        Returns:
            게시판 데이터 리스트
        """
        headers: Dict[str, str] = {}
        if INTERNAL_API_KEY:
            headers["X-Internal-API-Key"] = INTERNAL_API_KEY

        params = {"days": days, "size": size}

        try:
            resp = requests.get(
                SPRING_BOARD_API,
                params=params,
                headers=headers,
                timeout=SPRING_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()

            data = resp.json()

            # 응답 형식 파싱 (content가 있으면 Spring Page 형식)
            if isinstance(data, dict):
                content = data.get("content", data.get("data", data.get("items", [])))
                if isinstance(content, list):
                    return content
            elif isinstance(data, list):
                return data

            logger.warning(f"게시판 API 응답 형식 예상과 다름: {type(data)}")
            return []

        except requests.exceptions.ConnectionError:
            logger.warning("게시판 API 연결 실패. 샘플 데이터를 반환합니다.")
            return self._get_sample_feedback()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(
                    f"게시판 API 엔드포인트 미구현 (404). "
                    f"Spring 서버에 {SPRING_BOARD_API} 구현 필요. 샘플 데이터를 반환합니다."
                )
            else:
                logger.warning(
                    f"게시판 API HTTP 에러 ({e.response.status_code}). 샘플 데이터를 반환합니다."
                )
            return self._get_sample_feedback()
        except Exception as e:
            logger.error(f"게시판 API 호출 실패: {e}")
            return self._get_sample_feedback()

    def _get_sample_feedback(self) -> List[Dict[str, Any]]:
        """샘플 피드백 데이터 (API 실패 시 또는 테스트용) - Spring Board 엔티티 형식"""
        return [
            {
                "id": 1,
                "title": "로제파스타 요청",
                "content": "로제파스타 넣어주세요!",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T10:00:00",
            },
            {
                "id": 2,
                "title": "마라탕 먹고싶어요",
                "content": "마라탕 급식으로 나왔으면 좋겠어요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T11:00:00",
            },
            {
                "id": 3,
                "title": "로제파스타 추가 부탁",
                "content": "로제파스타 추가해주세요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T12:00:00",
            },
            {
                "id": 4,
                "title": "크로플 디저트 요청",
                "content": "크로플 디저트로 넣어주세요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T13:00:00",
            },
            {
                "id": 5,
                "title": "마라탕 제발요",
                "content": "마라탕 제발요!!",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T14:00:00",
            },
            {
                "id": 6,
                "title": "돈까스 자주 나왔으면",
                "content": "돈까스 더 자주 나왔으면 좋겠어요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T15:00:00",
            },
            {
                "id": 7,
                "title": "로제떡볶이 추가",
                "content": "로제떡볶이 메뉴 추가 부탁드려요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T16:00:00",
            },
            {
                "id": 8,
                "title": "마라샹궈 요청",
                "content": "마라샹궈 먹고싶습니다",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T17:00:00",
            },
            {
                "id": 9,
                "title": "약과 디저트",
                "content": "약과 디저트로 주세요",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T18:00:00",
            },
            {
                "id": 10,
                "title": "로제파스타!!!",
                "content": "로제파스타 먹고싶어요!!!",
                "category": "NEW_MENU",
                "createdAt": "2026-01-30T19:00:00",
            },
        ]

    def extract_menu_from_text_pattern(self, text: str) -> List[str]:
        """
        패턴 기반으로 텍스트에서 메뉴명 추출 (fallback용)

        Args:
            text: 게시글 내용

        Returns:
            추출된 메뉴명 리스트
        """
        menus = []

        # 패턴 매칭으로 메뉴명 추출
        for pattern in MENU_PATTERNS:
            matches = re.findall(pattern, text)
            menus.extend(matches)

        # 약어 변환
        standardized = []
        for menu in menus:
            # 약어가 있으면 정식 명칭으로 변환
            for abbr, full in ABBREVIATION_MAP.items():
                if abbr in menu:
                    menu = menu.replace(abbr, full)
            standardized.append(menu)

        # 단순 키워드 추출 (패턴에 매칭되지 않는 경우)
        if not standardized:
            # 간단한 메뉴 키워드 직접 추출
            simple_menus = [
                "로제파스타",
                "마라탕",
                "마라샹궈",
                "크로플",
                "약과",
                "탕후루",
                "소금빵",
                "떡볶이",
                "돈까스",
                "치킨",
                "두바이초콜릿",
                "로제떡볶이",
            ]
            for menu in simple_menus:
                if menu in text:
                    standardized.append(menu)

        return list(set(standardized))

    async def extract_menus_with_llm(
        self, feedback_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        LLM을 사용하여 게시글 목록에서 메뉴명 일괄 추출

        Args:
            feedback_list: 게시글 목록

        Returns:
            추출된 메뉴명 리스트
        """
        if not self.openai_client:
            logger.warning("OpenAI 클라이언트 없음. 패턴 기반 추출로 fallback")
            return []

        if not feedback_list:
            return []

        # 게시글 내용만 추출
        contents = []
        for fb in feedback_list:
            content = fb.get("content", "")
            title = fb.get("title", "")
            combined = f"{title} {content}".strip()
            if combined:
                contents.append(combined)

        if not contents:
            return []

        try:
            prompt = f"""
다음은 급식 신메뉴 요청 게시판의 게시글 목록입니다.
각 게시글에서 요청된 음식/메뉴 이름을 추출해주세요.

게시글 목록:
{json.dumps(contents, ensure_ascii=False, indent=2)}

규칙:
1. 음식/메뉴 이름만 추출 (재료명, 브랜드명 제외)
2. 약어/줄임말은 정확히 해석하여 원래 메뉴명으로 변환
3. 신조어/트렌드 메뉴도 포함
4. 중복 허용 (득표수 집계용)

[약어 해석 가이드]
- "두쫀쿠" = "두바이쫀득쿠키" (두바이+쫀득+쿠키)
- "돈까" = "돈까스"
- "치까" = "치킨까스"
- "제볶" = "제육볶음"
- "김볶" = "김치볶음밥"
- "떡볶" = "떡볶이"
- "마샹" = "마라샹궈"
- 약어 해석 시 글자 조합을 정확히 분석할 것 (예: 두+쫀+쿠 = 두바이+쫀득+쿠키)

JSON 형식으로 응답:
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

            logger.info(f"LLM 메뉴 추출 결과: {menus}")
            return menus

        except Exception as e:
            logger.error(f"LLM 메뉴 추출 실패: {e}")
            return []

    def extract_menu_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 메뉴명 추출 (패턴 기반, 단일 텍스트용)

        Args:
            text: 게시글 내용

        Returns:
            추출된 메뉴명 리스트
        """
        return self.extract_menu_from_text_pattern(text)

    def standardize_menu_name(self, menu: str) -> str:
        """
        메뉴명 표준화 (약어 해석 등)

        Args:
            menu: 원본 메뉴명

        Returns:
            표준화된 메뉴명
        """
        result = menu.strip()

        # 약어 변환
        for abbr, full in ABBREVIATION_MAP.items():
            if result == abbr or result.startswith(abbr):
                result = result.replace(abbr, full)
                break

        return result

    async def analyze(self, days: int = 30) -> Tuple[List[str], Dict[str, int]]:
        """
        게시판 분석 메인 함수

        Args:
            days: 분석 기간 (일)

        Returns:
            (상위 메뉴 리스트, 득표 현황 딕셔너리)
        """
        logger.info(f"게시판 분석 시작 (기간: {days}일)")

        # 1. 게시판 데이터 가져오기
        feedback_list = await self.fetch_board_feedback(days=days)
        logger.info(f"가져온 게시글 수: {len(feedback_list)}개")

        # 2. LLM으로 메뉴명 일괄 추출 시도
        all_menus: List[str] = []

        if self.openai_client and feedback_list:
            all_menus = await self.extract_menus_with_llm(feedback_list)

        # 3. LLM 추출 실패 시 패턴 기반 fallback
        if not all_menus:
            logger.info("LLM 추출 결과 없음. 패턴 기반 추출로 fallback")
            for feedback in feedback_list:
                content = feedback.get("content", "")
                title = feedback.get("title", "")
                combined = f"{title} {content}".strip()
                menus = self.extract_menu_from_text_pattern(combined)
                logger.info(f"게시글 ID {feedback.get('id')}에서 추출된 메뉴: {menus}")
                all_menus.extend(menus)

        # 4. 득표 집계
        vote_counter = Counter(all_menus)
        logger.info(f"추출된 고유 메뉴 수: {len(vote_counter)}개")

        # 5. 상위 N개 선정 (득표수 기준)
        top_menus = [menu for menu, _ in vote_counter.most_common(20)]
        votes_dict = dict(vote_counter.most_common(20))

        logger.info(f"상위 메뉴: {top_menus[:5]}")

        return top_menus, votes_dict
