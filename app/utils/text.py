from typing import Any, Optional
from app.core.config import ROLE_RULES


def get_role(cat: Any) -> Optional[str]:
    s = str(cat).strip()
    for role, cats in ROLE_RULES.items():
        if s in cats:
            return role
    return None
