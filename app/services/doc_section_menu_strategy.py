from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate

from app.models.strategies import MenuStrategyItem


menu_strategies_prompt = ChatPromptTemplate.from_messages([
("system", """
(Response Language: {response_lang})

당신은 급식 운영 데이터를 해석하는 전문가입니다.
"""),
("human", """
다음은 기간 내 제기된 급식 운영 상의 문제 및 제안사항, 그 중에서도 메뉴에 연관된 사항들입니다.

[운영 요약]
{section_summary} 

[운영 상 문제 및 제안사항]
{strategies}

당신은 제시된 상황 및 문제를 해결하기 위한 운영 전략, 그 중에서도 메뉴에 연관된 전략들을 도출합니다.
불필요한 추측은 하지 말고, 데이터에 근거해 서술하세요.

<주의>
- 전략 하나에 3~5문장 정도의 길이가 허용됩니다.
- Column 이름 혹은 Type 표기 혹은 결과값을 그대로 쓰는 대신 적절한 한국어 표현으로 대체하세요.
- 포맷은 Plain Text로 상정합니다.
""")
])

def generate_section_menu_strategy(
        section_summary:str,
        strategies:List[MenuStrategyItem],
        llm,
        response_lang: str = "Korean"
    ) -> str:
    menu_strategies_message = menu_strategies_prompt.format_messages(
        section_summary=section_summary,
        strategies=strategies,
        response_lang=response_lang,
        streaming=True
    )
    try:
        response = llm.invoke(menu_strategies_message).content.strip()
        return response
    except Exception as e:
        return ""
