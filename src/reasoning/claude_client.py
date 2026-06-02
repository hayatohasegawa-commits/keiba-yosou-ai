"""Claude API クライアントラッパー。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていない。.env を確認"
        )
    return Anthropic(api_key=api_key)


def complete(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.3,
) -> str:
    """単発の補完。返答テキストを返す。"""
    client = get_client()
    msg = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)
