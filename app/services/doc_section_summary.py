from __future__ import annotations

from app.models.analysis_periodic import *

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


summary_prompt = ChatPromptTemplate.from_messages([
("system", """
(Response Language: {response_lang})

당신은 급식 운영 리포트를 작성하는 분석가입니다. 과장 없이, 운영자 관점에서 간결하게 요약하세요.
"""),
("human", """
다음은 특정 기간의 급식 운영 분석 결과 요약 데이터입니다.

- 평균 만족도(5점 만점): {avg_rating}
- 평균 잔반률: {avg_leftover_rate}
- 리뷰 수: {review_count}
- 게시물 수: {post_count}
- 주요 문제 영역: {top_problem_areas}

이 데이터를 바탕으로 전체 운영 상황을 3~4문장으로 요약하세요.
""")
])

def generate_section_summary(
    analysis:PeriodicAnalysisResult,
    llm:ChatOpenAI,
    response_lang:str="Korean"
) -> str:
    summary_message = summary_prompt.format_messages(
        response_lang=response_lang,
        avg_rating=analysis.kpis["rating"].avg_rating,
        avg_leftover_rate=analysis.kpis["leftover"].avg_leftover_rate,
        review_count=analysis.reviews.count,
        post_count=analysis.posts.count,
        top_problem_areas=[p.tag for p in analysis.problem_areas[:2]],
        streaming=True,
    )
    try:
        response = llm.invoke(summary_message).content.strip()
        return response
    except Exception as e:
        return ""
