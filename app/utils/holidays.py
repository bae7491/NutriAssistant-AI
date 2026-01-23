from __future__ import annotations
from datetime import date
from typing import Set

try:
    from holidayskr import year_holidays
except Exception:

    def year_holidays(year):  # type: ignore
        return []


def get_holidays(year: int) -> Set[date]:
    # holidayskr는 (date, name) 튜플 리스트 형태
    return {d for d, _ in year_holidays(str(year))}
