import json
import os
import pandas as pd
from dotenv import load_dotenv

from app.services.gen_analysis_periodic import run_periodic_analysis
from app.services.gen_analysis_trends import run_trend_analysis
from app.services.gen_strategy_menu import run_menu_strategy_analysis
from app.services.doc_section_summary import generate_section_summary
from app.services.doc_section_leftover import generate_section_leftover
from app.services.doc_section_satisfaction import generate_section_satisfaction
from app.services.doc_section_issues import generate_section_issues
from app.services.doc_section_trends import generate_section_trend_analysis
from app.services.doc_section_menu_strategy import generate_section_menu_strategy
from app.services.doc_section_op_strategy import generate_section_op_strategy

from app.models.analysis_periodic import PeriodicAnalysisResult
from app.models.analysis_trends import TrendAnalysisResult
from app.models.strategies import MenuStrategyResponse
from app.models.report import *

from langchain_openai import ChatOpenAI

# .env 파일 로드
load_dotenv()

# LLM 1 설정 (메인)
llm_1_model_name = "gpt-4o"
llm_1_temperature = 0.5

# LLM 2 설정 (보조)
llm_2_model_name = "gpt-5-mini"  # gpt-4.1-mini -> gpt-4o-mini 수정
llm_2_temperature = 0.0


def read_api_keys() -> dict[str, str]:
    """환경 변수에서 API 키를 읽어옵니다."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    }


__g_api_keys: dict[str, str] = read_api_keys()


def generate_periodic_report(payload: MonthlyReportRequestPayload):
    print("[generate_periodic_report()] Request confirmed")

    # 음식DB
    df_food_db = pd.read_excel("_tmp_files/최종_통합_음식_DB.xlsx")

    # LLM 초기화 (메인 LLM)
    llm_1 = None
    try:
        llm_1 = ChatOpenAI(
            openai_api_key=__g_api_keys['OPENAI_API_KEY'],
            model=llm_1_model_name,
            temperature=llm_1_temperature,
            streaming=True
        )
    except Exception as e:
        print(f"LLM 1 초기화 실패: {e}")
        llm_1 = None

    # LLM 초기화 (보조 LLM)
    llm_2 = None
    try:
        llm_2 = ChatOpenAI(
            openai_api_key=__g_api_keys['OPENAI_API_KEY'],
            model=llm_2_model_name,
            temperature=llm_2_temperature,
            streaming=True
        )
    except Exception as e:
        print(f"LLM 2 초기화 실패: {e}")
        llm_2 = None

    # (Payload -> dict)
    payload_dict = payload.model_dump()

    # (기본 정보 확인)
    user_name: str = payload_dict['userName']
    year: int = payload_dict['year']
    month: int = payload_dict['month']
    target_group: str = payload_dict.get('targetGroup', "")

    # DataFrame 생성 - 식단표
    print("[generate_periodic_report()] Payload 해제 (mealPlan)")
    df_meal_plans = pd.DataFrame(payload_dict['mealPlan'])
    df_meal_plans['mealType'] = df_meal_plans['mealType'].map({"중식": "LUNCH", "석식": "DINNER"})
    for colname in ['rice', 'soup', 'main1', 'main2', 'side', 'dessert']:
        df_meal_plans[colname] = df_meal_plans[colname].fillna("-")
    df_meal_plans['date'] = pd.to_datetime(df_meal_plans["date"])

    # DataFrame 생성 - 일일 운영정보
    print("[generate_periodic_report()] Payload 해제 (dailyInfo)")
    df_daily_info = pd.DataFrame(payload_dict['dailyInfo'])
    df_daily_info['mealType'] = df_daily_info['mealType'].map({"중식": "LUNCH", "석식": "DINNER"})

    # DataFrame 생성 - 리뷰
    reviews = pd.DataFrame(payload_dict['reviews'])
    reviews['mealType'] = reviews['mealType'].map({"중식": "LUNCH", "석식": "DINNER"})
    reviews['date'] = pd.to_datetime(reviews['date'])
    reviews['review_id'] = (
            'R' + '-' +
            reviews['date'].dt.strftime('%Y%m%d') + '-' +
            reviews.groupby(reviews['date'].dt.normalize(), sort=False)
            .cumcount()
            .add(1)
            .astype(str)
            .str.zfill(4)
    )
    df_reviews = reviews[['review_id', 'date', 'mealType', 'rating', 'content']]

    # DataFrame 생성 - 일일 분석
    df_daily_report = pd.json_normalize(payload_dict['dailyAnalyses'])

    # DataFrame 생성 - 감정분석 (리뷰)
    df_sentiments_reviews = pd.DataFrame(payload_dict['reviewAnalyses'])
    df_sentiments_reviews['date'] = pd.to_datetime(
        df_sentiments_reviews['review_id'].str.extract(r'(\d{8})')[0],
        format='%Y%m%d',
        errors='coerce'
    )

    # DataFrame 생성 - 감정분석 (게시물)
    df_sentiments_posts = pd.DataFrame(payload_dict['postAnalyses'])
    df_sentiments_posts['date'] = pd.to_datetime(
        df_sentiments_posts['post_id'].str.extract(r'(\d{8})')[0],
        format='%Y%m%d',
        errors='coerce'
    )

    # 분석 - 기간에 대한 통계분석
    periodic_analysis_result: PeriodicAnalysisResult = run_periodic_analysis(
        df_meal_plans=df_meal_plans,
        df_daily_info=df_daily_info,
        df_sentiments_reviews=df_sentiments_reviews,
        df_sentiments_posts=df_sentiments_posts,
        df_daily_report=df_daily_report,
        df_reviews=df_reviews
    )
    # 분석 - 트렌드
    trend_analysis_result: TrendAnalysisResult = run_trend_analysis(
        df_sentiments_reviews=df_sentiments_reviews,
        df_meal_plans=df_meal_plans,
        df_food_db=df_food_db,
        period_analysis=periodic_analysis_result.model_dump(),
        target_month=month
    )
    # 분석 - 메뉴 전략
    strategy_menu: MenuStrategyResponse = run_menu_strategy_analysis(
        {
            "userName": user_name,
            "year": year,
            "month": month,
            "targetGroup": target_group,
            "mealPlan": payload_dict['mealPlan'],
            "reviews": payload_dict['reviews'],
            "feedbacks": payload_dict['posts'],
            "reviewAnalysis": payload_dict['reviewAnalyses'],
            "feedbackAnalysis": payload_dict['postAnalyses'],
            "dailyInfo": payload_dict['dailyInfo'],
            "dailyAnalysis": payload_dict['dailyAnalyses'],
            "trendAnalysis": trend_analysis_result.model_dump(),
            "llm": llm_2
        }
    )

    # 보고서 생성
    print("[generate_periodic_report()] Generating Reports (Summary)")
    section_summary: str = generate_section_summary(periodic_analysis_result, llm_1)
    print("[generate_periodic_report()] Generating Reports (Leftover)")
    section_leftover: str = generate_section_leftover(periodic_analysis_result, llm_1)
    print("[generate_periodic_report()] Generating Reports (Satisfaction)")
    section_satisfaction: str = generate_section_satisfaction(periodic_analysis_result, llm_1)
    print("[generate_periodic_report()] Generating Reports (Issues)")
    section_issues: str = generate_section_issues(periodic_analysis_result, llm_1)
    print("[generate_periodic_report()] Generating Reports (Trends)")
    section_trends: str = generate_section_trend_analysis(trend_analysis_result)
    print("[generate_periodic_report()] Generating Reports (Strategies - Menu)")
    menu_strategies: str = generate_section_menu_strategy(
        section_summary=section_summary,
        llm=llm_1,
        strategies=strategy_menu.menuStrategies
    )
    print("[generate_periodic_report()] Generating Reports (Strategies - Operational)")
    op_strategy: str = generate_section_op_strategy(
        section_summary=section_summary,
        section_issues=section_issues,
        section_trends=section_trends,
        strategy_menu=menu_strategies,
        llm=llm_1
    )

    # 리턴
    return MonthlyReport(
        userName=user_name,
        year=year,
        month=month,
        metadata=PeriodicReportMetadata(),
        data=PeriodicReportData(
            periodicAnalysis=periodic_analysis_result,
            trendAnalysis=trend_analysis_result,
            menuStrategy=strategy_menu
        ),
        doc=PeriodicReportDoc(
            summary=section_summary,
            leftover=section_leftover,
            satisfaction=section_satisfaction,
            issues=section_issues,
            trendAnalysis=section_trends,
            menuStrategies=menu_strategies,
            opStrategies=op_strategy
        )
    )