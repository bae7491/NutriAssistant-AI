from openai import OpenAI
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)


class AIAnalyzer:
    def __init__(self):
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")

        if not api_key:
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. AI 기능이 비활성화됩니다.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI 클라이언트 초기화 완료")

        self.model = "gpt-5-mini"

    def _check_client(self):
        """AI 클라이언트가 활성화되어 있는지 확인"""
        if self.client is None:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 설정해주세요."
            )

    async def analyze_facility_condition(self, facility_text: str) -> Dict[str, bool]:
        """시설 현황 분석"""
        if not facility_text or facility_text.strip() == "":
            return {"has_oven": True, "has_fryer": True, "has_griddle": True}

        try:
            self._check_client()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 급식실 조리 시설 분석기다.\n"
                            "입력은 두 가지 형태일 수 있다:\n"
                            "1. 보유 장비 목록 (예: '오븐, 찜기, 전기밥솥')\n"
                            "2. 장비 이슈/변경 사항 (예: '오븐 고장', '튀김기 사용불가')\n\n"
                            "반드시 아래 3개 키를 모두 포함한 JSON으로 출력해라:\n"
                            "- has_oven: 오븐 사용 가능 여부\n"
                            "- has_fryer: 튀김기 사용 가능 여부\n"
                            "- has_griddle: 철판/그리들 사용 가능 여부\n\n"
                            "판단 규칙:\n"
                            "1. 보유 장비 목록 형태인 경우:\n"
                            "   - 오븐/컨벡션오븐이 언급됨 → has_oven=true, 아니면 false\n"
                            "   - 튀김기/프라이어가 언급됨 → has_fryer=true, 아니면 false\n"
                            "   - 철판/그리들/핫플레이트가 언급됨 → has_griddle=true, 아니면 false\n"
                            "   - 언급되지 않은 장비는 없는 것으로 판단 (false)\n"
                            "2. 이슈/변경 형태인 경우:\n"
                            "   - '고장/불가/사용불가/수리중'이 붙으면 해당 장비 false\n"
                            "   - 그 외 장비는 true (기본 보유 가정)\n"
                        ),
                    },
                    {"role": "user", "content": facility_text},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # 모든 키가 있는지 확인하고, 없으면 기본값 설정
            return {
                "has_oven": result.get("has_oven", False),
                "has_fryer": result.get("has_fryer", False),
                "has_griddle": result.get("has_griddle", False),
            }

        except ValueError as e:
            raise e
        except Exception as e:
            print(f"❌ AI 분석 중 오류 발생: {e}")
            return {"has_oven": True, "has_fryer": True, "has_griddle": True}

    async def analyze_reviews_and_generate_weights(
        self, report_data: dict, valid_menu_names: set
    ) -> Dict[str, float]:
        """리포트 분석 및 가중치 생성"""
        try:
            self._check_client()

            json_str = json.dumps(report_data, ensure_ascii=False)

            prompt = f"""
                너는 급식 데이터 분석가야. 리포트를 보고 메뉴별 평가 점수(0~100)를 매겨줘.
                칭찬/Best는 80점 이상, 불만/Worst는 40점 미만.
                
                출력 포맷:
                [{{"menu": "메뉴명", "score": 85, "reason": "이유"}}, ...]
                
                [데이터]
                {json_str}
            """.strip()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            parsed = json.loads(response.choices[0].message.content)

            # AI 응답 파싱
            ai_scores = []
            if isinstance(parsed, list):
                ai_scores = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        ai_scores = v
                        break

            # 가중치 계산
            weights = {}
            for item in ai_scores:
                menu = str(item.get("menu", "")).strip()
                if not menu or menu not in valid_menu_names:
                    continue

                ai_score = float(item.get("score", 60))
                delta = (ai_score - 60.0) / 50.0
                delta = max(-1.0, min(1.0, delta))
                weights[menu] = delta

            return weights

        except ValueError as e:
            raise e
        except Exception as e:
            print(f"❌ 리포트 분석 중 오류 발생: {e}")
            return {}
