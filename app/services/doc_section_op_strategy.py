from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


op_strategy_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
(Response Language: {response_lang})

당신은 학교 급식 영양사를 위한 월간 리포트를 작성하는 전문가입니다.
""",
        ),
        (
            "human",
            """
다음 분석 결과를 바탕으로 리포트 마지막에 들어갈 운영 전략을 작성합니다.

[운영 요약]
{section_summary} 

[운영 상 문제]
{section_issues}

[추세]
{section_trends}
 
[메뉴 전략]
{strategy_menu}

당신은 제시된 상황 및 문제를 해결하기 위한 운영 전략, 그 중에서도 메뉴에 연관된 전략들을 도출합니다.
불필요한 추측은 하지 말고, 데이터에 근거해 서술하세요.

<주의>
- 전략 하나에 3~5문장 정도의 길이가 허용됩니다.
- Column 이름 혹은 Type 표기 혹은 결과값을 그대로 쓰는 대신 적절한 한국어 표현으로 대체하세요.
- 포맷은 Plain Text로 상정합니다.
""",
        ),
    ]
)


def generate_section_op_strategy(
    section_summary,
    section_issues,
    section_trends,
    strategy_menu,
    llm,
    response_lang: str = "Korean",
) -> str:
    op_strategy_message = op_strategy_prompt.format_messages(
        response_lang=response_lang,
        section_summary=section_summary,
        section_issues=section_issues,
        section_trends=section_trends,
        strategy_menu=strategy_menu,
    )
    try:
        response = llm.invoke(op_strategy_message).content.strip()
        return response
    except Exception as e:
        return ""
