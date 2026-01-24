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
                            "너는 조리 시설 분석기다.\n"
                            "입력은 '변경/이슈' 텍스트일 수 있다(예: '오븐 고장').\n"
                            "따라서 입력에 언급된 장비만 판단하고, 언급되지 않은 장비는 JSON에 포함하지 마라.\n"
                            "반드시 아래 키 중 필요한 것만 포함해서 JSON으로만 출력해라:\n"
                            "- has_oven\n"
                            "- has_fryer\n"
                            "- has_griddle\n"
                            "규칙:\n"
                            "- '오븐'이 언급되고 '고장/불가/사용불가/못씀'이면 has_oven=false\n"
                            "- '튀김기/튀김'이 언급되고 '고장/불가/사용불가/못씀'이면 has_fryer=false\n"
                            "- '철판/부침/전/그리들'이 언급되고 '고장/불가/사용불가/못씀'이면 has_griddle=false\n"
                            "- 언급되었지만 문제 없음이면 true\n"
                        ),
                    },
                    {"role": "user", "content": facility_text},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result

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
