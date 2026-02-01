from __future__ import annotations

from app.models.analysis_trends import TrendAnalysisResult


def generate_section_trend_analysis(result: TrendAnalysisResult) -> str:
    """Convert TrendAnalysisResult to doc.trendAnalysis string."""
    lines = []

    lines.append(
        f"분석 기간: {result.period_start} ~ {result.period_end} (총 {result.total_records:,}건)"
    )
    lines.append("")

    lines.append("[카테고리별 만족도 현황]")
    if result.preferred_categories:
        pref_str = ", ".join(
            [f"{c.category}({c.pos_ratio}%)" for c in result.preferred_categories]
        )
        lines.append(f"• 긍정 비율 상위: {pref_str}")
    if result.problem_categories:
        prob_str = ", ".join(
            [f"{c.category}({c.neg_ratio}%)" for c in result.problem_categories]
        )
        lines.append(f"• 부정 비율 상위: {prob_str}")
    lines.append("")

    if result.weekly_neg_trend:
        weeks = sorted(result.weekly_neg_trend.keys())
        first_week, last_week = weeks[0], weeks[-1]
        first_neg = result.weekly_neg_trend[first_week]
        last_neg = result.weekly_neg_trend[last_week]
        change = last_neg - first_neg

        lines.append("[주차별 부정 비율 추이]")
        if change > 2:
            lines.append(
                f"• {first_week}주차 {first_neg}% → {last_week}주차 {last_neg}% (+{change:.1f}%p 증가)"
            )
            lines.append("• 만족도 하락 추세 감지")
        elif change < -2:
            lines.append(
                f"• {first_week}주차 {first_neg}% → {last_week}주차 {last_neg}% ({change:.1f}%p 감소)"
            )
            lines.append("• 만족도 개선 추세")
        else:
            lines.append(
                f"• {first_week}주차 {first_neg}% → {last_week}주차 {last_neg}% ({change:+.1f}%p)"
            )
            lines.append("• 부정 비율 안정적 유지")
        lines.append("")

    if result.preference_changes:
        lines.append("[카테고리별 선호도 변화]")
        increases = [c for c in result.preference_changes if c.direction == "increase"]
        decreases = [c for c in result.preference_changes if c.direction == "decrease"]
        if increases:
            inc_str = ", ".join(
                [f"{c.category}(+{c.change_percent}%p)" for c in increases[:3]]
            )
            lines.append(f"• 선호도 상승: {inc_str}")
        if decreases:
            dec_str = ", ".join(
                [f"{c.category}({c.change_percent}%p)" for c in decreases[:3]]
            )
            lines.append(f"• 선호도 하락: {dec_str}")
        lines.append("")

    if result.complaint_tag_changes:
        lines.append("[불만 태그 증감 추이]")
        increases = [
            c for c in result.complaint_tag_changes if c.direction == "increase"
        ]
        decreases = [
            c for c in result.complaint_tag_changes if c.direction == "decrease"
        ]
        if increases:
            inc_str = ", ".join(
                [f"{c.tag}(+{c.change_percent:.0f}%)" for c in increases[:3]]
            )
            lines.append(f"• 증가: {inc_str}")
        if decreases:
            dec_str = ", ".join(
                [f"{c.tag}({c.change_percent:.0f}%)" for c in decreases[:3]]
            )
            lines.append(f"• 감소: {dec_str}")
        lines.append("")

    return "\n".join(lines)
