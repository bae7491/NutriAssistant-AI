from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


satisfaction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
(Response Language: {response_lang})

당신은 고객 만족도 분석 리포트를 작성하는 분석가입니다.
""",
        ),
        (
            "human",
            """
다음은 기간 내 급식 만족도 관련 데이터입니다.

- 평균 평점: {avg_rating}
- 감정 분포: {sentiment_distribution}
- 식사 유형 분포: {mealType_distribution}

이 데이터를 바탕으로 이용자 만족도 수준과 특징을 설명하세요.

<주의>
- Column 이름 혹은 Type 표기 혹은 결과값을 그대로 쓰는 대신 적절한 한국어 표현으로 대체하세요.
- '급식 만족도'에 대한 보고서용 문단 텍스트만 출력하세요.
- 포맷은 Plain Text로 상정합니다.
""",
        ),
    ]
)


def generate_section_satisfaction(analysis, llm, response_lang: str = "Korean") -> str:
    satisfaction_message = satisfaction_prompt.format_messages(
        response_lang=response_lang,
        avg_rating=analysis.kpis["rating"].avg_rating,
        sentiment_distribution=analysis.reviews.sentiment_distribution,
        mealType_distribution=analysis.reviews.mealType_distribution,
        streaming=True,
    )
    try:
        response = llm.invoke(satisfaction_message).content.strip()
        return response
    except Exception as e:
        return ""
