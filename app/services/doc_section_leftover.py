from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


leftover_prompt = ChatPromptTemplate.from_messages([
("system", """
(Response Language: {response_lang})

당신은 급식 운영 데이터를 해석하는 전문가입니다.
"""),
("human", """
다음은 기간 내 잔반률 분석 결과입니다.

- 평균 잔반률: {avg_leftover_rate}
- 잔반률이 높았던 사례들: {worst_cases}

이 데이터를 바탕으로 잔반 발생 경향과 운영상 시사점을 설명하세요.
불필요한 추측은 하지 말고, 데이터에 근거해 서술하세요.

<주의>
- Column 이름 혹은 Type 표기 혹은 결과값을 그대로 쓰는 대신 적절한 한국어 표현으로 대체하세요.
- '잔반 추세'에 대한 보고서용 문단 텍스트만 출력하세요.
- 포맷은 Plain Text로 상정합니다.
""")
])

def generate_section_leftover(analysis, llm, response_lang: str = "Korean") -> str:
    leftover_message = leftover_prompt.format_messages(
        response_lang=response_lang,
        avg_leftover_rate=analysis.kpis["leftover"].avg_leftover_rate,
        worst_cases=analysis.kpis["leftover"].worst_cases,
        streaming=True,
    )
    try:
        response = llm.invoke(leftover_message).content.strip()
        return response
    except Exception as e:
        return ""
