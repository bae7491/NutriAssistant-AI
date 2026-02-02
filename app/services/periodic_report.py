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
llm_2_model_name = "gpt-4o-mini"  # ✅ 수정: gpt-5-mini는 존재하지 않음
llm_2_temperature = 0.0


def read_api_keys() -> dict[str, str]:
    """환경 변수에서 API 키를 읽어옵니다."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    }


__g_api_keys: dict[str, str] = read_api_keys()


def generate_periodic_report(payload: MonthlyReportRequestPayload):
    """
    월간 리포트 생성 메인 함수

    Args:
        payload: 월간 리포트 요청 데이터

    Returns:
        MonthlyReport: 생성된 월간 리포트
    """
    print("=" * 80)
    print("[generate_periodic_report()] 월간 리포트 생성 시작")
    print("=" * 80)

    # ============================================================================
    # 1. 음식 DB 로드 (Spring Boot FoodInfo 사용)
    # ============================================================================
    print("📋 [1/10] 음식 DB 로드 중...")

    try:
        # ✅ Spring API에서 음식 DB 가져오기
        from app.services.food_loader import get_context

        ctx = get_context()

        # Context를 DataFrame으로 변환
        all_foods = []
        for role, pool in ctx.pools.items():
            foods = pool.copy()
            foods['role'] = role
            all_foods.append(foods)

        df_food_db = pd.concat(all_foods, ignore_index=True)
        print(f"   ✅ 음식 DB 로드 완료: {len(df_food_db)}개 메뉴")

    except Exception as e:
        print(f"   ⚠️  음식 DB 로드 실패: {e}")
        print("   📝 빈 DataFrame으로 계속 진행합니다")
        # 빈 DataFrame으로 대체 (트렌드 분석은 제한적으로 수행)
        df_food_db = pd.DataFrame()

    # ============================================================================
    # 2. LLM 초기화
    # ============================================================================
    print("🤖 [2/10] LLM 초기화 중...")

    # LLM 1 초기화 (메인 LLM - GPT-4o)
    llm_1 = None
    try:
        llm_1 = ChatOpenAI(
            openai_api_key=__g_api_keys["OPENAI_API_KEY"],
            model=llm_1_model_name,
            temperature=llm_1_temperature,
            streaming=True,
        )
        print(f"   ✅ 메인 LLM 초기화 완료: {llm_1_model_name}")
    except Exception as e:
        print(f"   ❌ 메인 LLM 초기화 실패: {e}")
        llm_1 = None

    # LLM 2 초기화 (보조 LLM - GPT-4o-mini)
    llm_2 = None
    try:
        llm_2 = ChatOpenAI(
            openai_api_key=__g_api_keys["OPENAI_API_KEY"],
            model=llm_2_model_name,
            temperature=llm_2_temperature,
            streaming=True,
        )
        print(f"   ✅ 보조 LLM 초기화 완료: {llm_2_model_name}")
    except Exception as e:
        print(f"   ❌ 보조 LLM 초기화 실패: {e}")
        llm_2 = None

    # ============================================================================
    # 3. Payload 데이터 추출
    # ============================================================================
    print("📦 [3/10] Payload 데이터 추출 중...")

    payload_dict = payload.model_dump()

    # 기본 정보 확인
    user_name: str = payload_dict.get("userName", "관리자")
    year: int = payload_dict["year"]
    month: int = payload_dict["month"]
    target_group: str = payload_dict.get("targetGroup", "")

    print(f"   📅 대상 기간: {year}년 {month}월")
    print(f"   👤 사용자: {user_name}")
    if target_group:
        print(f"   🎯 대상 그룹: {target_group}")

    # ============================================================================
    # 4. DataFrame 생성 - 식단표
    # ============================================================================
    print("🍽️  [4/10] 식단표 데이터 처리 중...")

    df_meal_plans = pd.DataFrame(payload_dict.get("mealPlan", []))

    if not df_meal_plans.empty:
        df_meal_plans["mealType"] = df_meal_plans["mealType"].map(
            {"중식": "LUNCH", "석식": "DINNER"}
        )
        for colname in ["rice", "soup", "main1", "main2", "side", "dessert", "kimchi"]:
            if colname in df_meal_plans.columns:
                df_meal_plans[colname] = df_meal_plans[colname].fillna("-")
        df_meal_plans["date"] = pd.to_datetime(df_meal_plans["date"])
        print(f"   ✅ 식단표 데이터: {len(df_meal_plans)}건")
    else:
        print("   ⚠️  식단표 데이터 없음")

    # ============================================================================
    # 5. DataFrame 생성 - 일일 운영정보
    # ============================================================================
    print("📊 [5/10] 일일 운영정보 처리 중...")

    df_daily_info = pd.DataFrame(payload_dict.get("dailyInfo", []))

    if not df_daily_info.empty:
        df_daily_info["mealType"] = df_daily_info["mealType"].map(
            {"중식": "LUNCH", "석식": "DINNER"}
        )
        print(f"   ✅ 일일 운영정보: {len(df_daily_info)}건")
    else:
        print("   ⚠️  일일 운영정보 없음")

    # ============================================================================
    # 6. DataFrame 생성 - 리뷰
    # ============================================================================
    print("💬 [6/10] 리뷰 데이터 처리 중...")

    reviews = pd.DataFrame(payload_dict.get("reviews", []))

    if not reviews.empty:
        reviews["mealType"] = reviews["mealType"].map({"중식": "LUNCH", "석식": "DINNER"})
        reviews["date"] = pd.to_datetime(reviews["date"])
        reviews["review_id"] = (
                "R"
                + "-"
                + reviews["date"].dt.strftime("%Y%m%d")
                + "-"
                + reviews.groupby(reviews["date"].dt.normalize(), sort=False)
                .cumcount()
                .add(1)
                .astype(str)
                .str.zfill(4)
        )
        df_reviews = reviews[["review_id", "date", "mealType", "rating", "content"]]
        print(f"   ✅ 리뷰 데이터: {len(df_reviews)}건")
    else:
        df_reviews = pd.DataFrame()
        print("   ⚠️  리뷰 데이터 없음")

    # ============================================================================
    # 7. DataFrame 생성 - 일일 분석
    # ============================================================================
    print("📈 [7/10] 일일 분석 데이터 처리 중...")

    daily_analyses = payload_dict.get("dailyAnalyses", [])
    if daily_analyses:
        df_daily_report = pd.json_normalize(daily_analyses)
        print(f"   ✅ 일일 분석 데이터: {len(df_daily_report)}건")
    else:
        df_daily_report = pd.DataFrame()
        print("   ⚠️  일일 분석 데이터 없음")

    # ============================================================================
    # 8. DataFrame 생성 - 감정분석 (리뷰)
    # ============================================================================
    print("😊 [8/10] 리뷰 감정분석 데이터 처리 중...")

    review_analyses = payload_dict.get("reviewAnalyses", [])
    if review_analyses:
        df_sentiments_reviews = pd.DataFrame(review_analyses)
        df_sentiments_reviews["date"] = pd.to_datetime(
            df_sentiments_reviews["review_id"].str.extract(r"(\d{8})")[0],
            format="%Y%m%d",
            errors="coerce",
        )
        print(f"   ✅ 리뷰 감정분석: {len(df_sentiments_reviews)}건")
    else:
        df_sentiments_reviews = pd.DataFrame()
        print("   ⚠️  리뷰 감정분석 데이터 없음")

    # ============================================================================
    # 9. DataFrame 생성 - 감정분석 (게시물)
    # ============================================================================
    print("📝 [9/10] 게시물 감정분석 데이터 처리 중...")

    post_analyses = payload_dict.get("postAnalyses", [])
    if post_analyses:
        df_sentiments_posts = pd.DataFrame(post_analyses)
        df_sentiments_posts["date"] = pd.to_datetime(
            df_sentiments_posts["post_id"].str.extract(r"(\d{8})")[0],
            format="%Y%m%d",
            errors="coerce",
        )
        print(f"   ✅ 게시물 감정분석: {len(df_sentiments_posts)}건")
    else:
        df_sentiments_posts = pd.DataFrame()
        print("   ⚠️  게시물 감정분석 데이터 없음")

    # ============================================================================
    # 10. 통계 분석 실행
    # ============================================================================
    print("=" * 80)
    print("🔬 [10/10] AI 분석 시작")
    print("=" * 80)

    # 분석 1: 기간 통계 분석
    print("📊 [분석 1/3] 주기 분석 (Periodic Analysis) 실행 중...")
    try:
        periodic_analysis_result: PeriodicAnalysisResult = run_periodic_analysis(
            df_meal_plans=df_meal_plans,
            df_daily_info=df_daily_info,
            df_sentiments_reviews=df_sentiments_reviews,
            df_sentiments_posts=df_sentiments_posts,
            df_daily_report=df_daily_report,
            df_reviews=df_reviews,
        )
        print("   ✅ 주기 분석 완료")
    except Exception as e:
        print(f"   ❌ 주기 분석 실패: {e}")
        raise

    # 분석 2: 트렌드 분석
    print("📈 [분석 2/3] 트렌드 분석 (Trend Analysis) 실행 중...")
    try:
        trend_analysis_result: TrendAnalysisResult = run_trend_analysis(
            df_sentiments_reviews=df_sentiments_reviews,
            df_meal_plans=df_meal_plans,
            df_food_db=df_food_db,
            period_analysis=periodic_analysis_result.model_dump(),
            target_month=month,
        )
        print("   ✅ 트렌드 분석 완료")
    except Exception as e:
        print(f"   ❌ 트렌드 분석 실패: {e}")
        raise

    # 분석 3: 메뉴 전략 분석
    print("🎯 [분석 3/3] 메뉴 전략 분석 (Menu Strategy) 실행 중...")
    try:
        strategy_menu: MenuStrategyResponse = run_menu_strategy_analysis(
            {
                "userName": user_name,
                "year": year,
                "month": month,
                "targetGroup": target_group,
                "mealPlan": payload_dict.get("mealPlan", []),
                "reviews": payload_dict.get("reviews", []),
                "feedbacks": payload_dict.get("posts", []),
                "reviewAnalysis": payload_dict.get("reviewAnalyses", []),
                "feedbackAnalysis": payload_dict.get("postAnalyses", []),
                "dailyInfo": payload_dict.get("dailyInfo", []),
                "dailyAnalysis": payload_dict.get("dailyAnalyses", []),
                "trendAnalysis": trend_analysis_result.model_dump(),
                "llm": llm_2,
            }
        )
        print("   ✅ 메뉴 전략 분석 완료")
    except Exception as e:
        print(f"   ❌ 메뉴 전략 분석 실패: {e}")
        raise

    # ============================================================================
    # 11. GPT 문서 생성
    # ============================================================================
    print("=" * 80)
    print("📝 문서 생성 시작 (GPT 활용)")
    print("=" * 80)

    print("✍️  [문서 1/7] 요약 (Summary) 생성 중...")
    section_summary: str = generate_section_summary(periodic_analysis_result, llm_1)

    print("✍️  [문서 2/7] 잔반률 (Leftover) 생성 중...")
    section_leftover: str = generate_section_leftover(periodic_analysis_result, llm_1)

    print("✍️  [문서 3/7] 만족도 (Satisfaction) 생성 중...")
    section_satisfaction: str = generate_section_satisfaction(
        periodic_analysis_result, llm_1
    )

    print("✍️  [문서 4/7] 이슈 (Issues) 생성 중...")
    section_issues: str = generate_section_issues(periodic_analysis_result, llm_1)

    print("✍️  [문서 5/7] 트렌드 분석 (Trends) 생성 중...")
    section_trends: str = generate_section_trend_analysis(trend_analysis_result)

    print("✍️  [문서 6/7] 메뉴 전략 (Menu Strategy) 생성 중...")
    menu_strategies: str = generate_section_menu_strategy(
        section_summary=section_summary,
        llm=llm_1,
        strategies=strategy_menu.menuStrategies,
    )

    print("✍️  [문서 7/7] 운영 전략 (Operational Strategy) 생성 중...")
    op_strategy: str = generate_section_op_strategy(
        section_summary=section_summary,
        section_issues=section_issues,
        section_trends=section_trends,
        strategy_menu=menu_strategies,
        llm=llm_1,
    )

    # ============================================================================
    # 12. 최종 리포트 생성
    # ============================================================================
    print("=" * 80)
    print("✅ 월간 리포트 생성 완료!")
    print("=" * 80)

    return MonthlyReport(
        userName=user_name,
        year=year,
        month=month,
        metadata=PeriodicReportMetadata(),
        data=PeriodicReportData(
            periodicAnalysis=periodic_analysis_result,
            trendAnalysis=trend_analysis_result,
            menuStrategy=strategy_menu,
        ),
        doc=PeriodicReportDoc(
            summary=section_summary,
            leftover=section_leftover,
            satisfaction=section_satisfaction,
            issues=section_issues,
            trendAnalysis=section_trends,
            menuStrategies=menu_strategies,
            opStrategies=op_strategy,
        ),
    )