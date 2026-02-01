from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


issues_prompt = ChatPromptTemplate.from_messages([
("system", """
(Response Language: {response_lang})

당신은 급식 운영 이슈를 분석하는 전문가입니다.
"""),
("human", """
다음은 특정 기간 중 주요 이슈가 관측된 식사 사례들입니다.

{context}

각 사례별로:
- 어떤 요소가 문제로 나타났는지
- 메뉴 구성과 어떤 연관이 있는지
- 반복적 이슈인지 일시적 현상인지

를 중심으로 원인을 분석하세요.

<주의>
- Column 이름 혹은 Type 표기 혹은 결과값을 그대로 쓰는 대신 적절한 한국어 표현으로 대체하세요.
- '급식 만족도'에 대한 보고서용 문단 텍스트만 출력하세요.
- 포맷은 Plain Text로 상정합니다.
""")
])

def _is_negative_aspect(aspect) -> bool:
    if isinstance(aspect, dict):
        return aspect.get("neg_rate", 0) > 0
    return getattr(aspect, "neg_rate", 0) > 0

def _aspect_tag(aspect) -> str:
    if isinstance(aspect, dict):
        return aspect.get("tag", "")
    return getattr(aspect, "tag", "")

def generate_section_issues(analysis, llm, response_lang: str = "Korean") -> str:
    context = [
        {
            "date": d.date,
            "mealType": d.mealType,
            "menus": d.menus,
            "problem_tags": [
                _aspect_tag(a) for a in d.aspect_summary if _is_negative_aspect(a)
            ],
            "evidence": d.evidence_phrases,
        }
        for d in analysis.deepdives
    ]

    issues_message = issues_prompt.format_messages(
        response_lang=response_lang,
        context=context,
        streaming=True,
    )
    try:
        response = llm.invoke(issues_message).content.strip()
        return response
    except Exception as e:
        return ""
