"""Parsing for the real Tripwire agent (T12).

Kept separate from scripts/t01_sanity_test.py (Manahil's merged, frozen T01 sanity check) per
her T03 reply: one owned copy of the parser here, instead of two implementations that could
drift apart.
"""

from __future__ import annotations

import re

THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
ACTION_RE = re.compile(r"Action:\s*(.*)", re.IGNORECASE)


def parse_reply(text: str) -> tuple[str, str, bool]:
    """Parse a raw LLM reply into (thought, action, parse_ok).

    `parse_ok` is False when no `Action:` line was found, in which case `action` falls back
    to "look" — this flag is what lets the logger tell a real "look" choice apart from a
    model formatting failure (docs/log_schema.md's `parse_ok` step field).
    """
    thought_match = THOUGHT_RE.search(text)
    action_match = ACTION_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    parse_ok = action_match is not None
    action = action_match.group(1).strip() if action_match else "look"
    return thought, action, parse_ok
