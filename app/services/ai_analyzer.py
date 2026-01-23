from openai import OpenAI
import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class AIAnalyzer:
    def __init__(self):
        # 환경변수에서 직접 읽기
        api_key = os.getenv("OPENAI_API_KEY")

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
                            "당신은 조리 시설 분석 전문가입니다. "
                            "주방 기기 목록을 분석하여 JSON으로 반환하세요. "
                            '출력 예시: {"has_oven": false, "has_fryer": true, "has_griddle": true}'
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
