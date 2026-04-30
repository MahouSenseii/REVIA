"""V4.1 Built-in skills — all zero-dep, offline-friendly.

* CalculatorSkill   — safe arithmetic via ``ast`` (no eval)
* ClockSkill        — current time / date / day of week
* EchoSkill         — debugging passthrough
* MemoryRecallSkill — pulls relevant episodes from V3 EpisodicMemoryStore

Each skill keeps its trigger patterns specific so the registry's match
score stays meaningful (false positives are expensive).
"""
from __future__ import annotations

import ast
import datetime as _dt
import operator
import time
from typing import Any

from .base import Skill, SkillRequest, SkillResponse


# ---------------------------------------------------------------------------
# Calculator — evaluate simple arithmetic expressions safely.
# ---------------------------------------------------------------------------

_CALC_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALC_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calc_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_BIN_OPS:
        return _CALC_BIN_OPS[type(node.op)](_calc_eval(node.left),
                                              _calc_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_UNARY_OPS:
        return _CALC_UNARY_OPS[type(node.op)](_calc_eval(node.operand))
    raise ValueError(f"unsupported_expression:{type(node).__name__}")


class CalculatorSkill(Skill):
    name = "calculator"
    description = "Evaluate basic arithmetic expressions (+,-,*,/,**,%,//)."
    cost_class = "free"
    triggers = (
        r"\b\d+\s*[+\-*/^%]\s*\d+",
        r"\bcalculate\b",
        r"\bcompute\b",
        r"what\s+is\s+\d",
        r"how\s+much\s+is\s+\d",
    )

    def _execute(self, req: SkillRequest) -> SkillResponse:
        expr = self._extract_expression(
            req.arguments.get("expression") or req.user_text
        )
        if not expr:
            return SkillResponse(success=False, error="no_expression_found")
        try:
            tree = ast.parse(expr, mode="eval")
            value = _calc_eval(tree)
        except Exception as exc:
            return SkillResponse(
                success=False,
                error=f"calc_error: {type(exc).__name__}: {exc}",
            )
        rendered = (
            str(int(value)) if value == int(value) and abs(value) < 1e15
            else f"{value:.6g}"
        )
        return SkillResponse(
            success=True,
            text=f"{expr} = {rendered}",
            data={"expression": expr, "value": value, "rendered": rendered},
        )

    @staticmethod
    def _extract_expression(text: str) -> str:
        if not text:
            return ""
        # Prefer the longest contiguous run of digits / operators / parens.
        import re
        matches = re.findall(r"[\d\.\s+\-*/^%()]+", text)
        # Replace "^" with "**" (common user shorthand for power).
        candidates = [m.strip().replace("^", "**") for m in matches if m.strip()]
        candidates = [c for c in candidates if any(ch.isdigit() for ch in c)]
        if not candidates:
            return ""
        # Pick the longest with at least one operator.
        candidates.sort(key=len, reverse=True)
        for c in candidates:
            if any(op in c for op in ("+", "-", "*", "/", "%", "**")):
                return c
        return candidates[0]


# ---------------------------------------------------------------------------
# Clock — current time / date.
# ---------------------------------------------------------------------------

class ClockSkill(Skill):
    name = "clock"
    description = "Report the current local date, time, weekday, or timestamp."
    cost_class = "free"
    triggers = (
        r"\bwhat\s+time\b",
        r"\bcurrent\s+time\b",
        r"\bwhat\s+date\b",
        r"\btoday['s]*\s+date\b",
        r"\bday\s+of\s+the\s+week\b",
        r"\bwhat\s+day\b",
        r"\btimestamp\b",
    )

    def _execute(self, req: SkillRequest) -> SkillResponse:
        now = _dt.datetime.now()
        utc = _dt.datetime.utcnow()
        text = now.strftime("%A, %B %d %Y, %H:%M:%S")
        return SkillResponse(
            success=True,
            text=text,
            data={
                "iso_local": now.isoformat(),
                "iso_utc": utc.isoformat() + "Z",
                "weekday": now.strftime("%A"),
                "year": now.year, "month": now.month, "day": now.day,
                "hour": now.hour, "minute": now.minute, "second": now.second,
                "epoch": int(time.time()),
            },
        )


# ---------------------------------------------------------------------------
# Echo — debugging passthrough.
# ---------------------------------------------------------------------------

class EchoSkill(Skill):
    name = "echo"
    description = "Return the user message verbatim (debug only)."
    cost_class = "free"
    triggers = (r"^\s*echo\b",)

    def _execute(self, req: SkillRequest) -> SkillResponse:
        text = req.user_text
        if text.lower().lstrip().startswith("echo"):
            text = text.lstrip()[4:].strip()
        return SkillResponse(
            success=True,
            text=text or "",
            data={"echoed_chars": len(text or "")},
        )


# ---------------------------------------------------------------------------
# MemoryRecall — query the V3 EpisodicMemoryStore.
# ---------------------------------------------------------------------------

class MemoryRecallSkill(Skill):
    name = "memory_recall"
    description = "Search past conversations for relevant prior turns."
    cost_class = "free"
    triggers = (
        r"\bremember\s+when\b",
        r"\bdid\s+(i|we|you)\s+(say|tell|talk|mention)\b",
        r"\bearlier\s+(today|i\s+said|we\s+talked)\b",
        r"\blast\s+time\s+(we|i|you)\b",
        r"\brecall\b",
    )

    def __init__(self, episode_store=None, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._store = episode_store

    def _execute(self, req: SkillRequest) -> SkillResponse:
        if self._store is None:
            return SkillResponse(success=False, error="no_memory_store")
        query = (req.arguments.get("query") or req.user_text or "").strip()
        limit = int(req.arguments.get("limit") or 3)
        try:
            results = self._store.search(query, limit=limit)
        except Exception as exc:
            return SkillResponse(success=False, error=f"search_error: {exc}")
        if not results:
            return SkillResponse(
                success=True,
                text="No matching prior conversation found.",
                data={"matches": []},
            )
        bullets = []
        for r in results:
            ep = r.episode
            bullets.append(
                f"- (score={r.score:.2f}) [{ep.intent_label}/{ep.emotion_label}] "
                f"you said: {ep.user_text[:80]!r}; "
                f"I said: {ep.reply_text[:80]!r}"
            )
        return SkillResponse(
            success=True,
            text="Relevant past turns:\n" + "\n".join(bullets),
            data={
                "matches": [r.to_dict() for r in results],
                "count": len(results),
            },
        )
