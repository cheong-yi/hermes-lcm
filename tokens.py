"""Token counting utilities for LCM.

Uses tiktoken when available, falls back to char-based estimate.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List

from .message_content import normalize_content_value

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_encoder = None
_encoder_checked = False


def _get_encoder():
    """Lazily load tiktoken cl100k_base encoder."""
    global _encoder, _encoder_checked
    if _encoder_checked:
        return _encoder
    _encoder_checked = True
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken not available, using char-based estimates")
    return _encoder


def _fallback_token_estimate(text: str) -> int:
    # Latin text is ~4 chars/token, but CJK and other non-Latin scripts
    # tokenize far denser (~1-2 tokens/char). A flat len//4 undercounts them
    # ~3-4x, so preflight under-triggers and assembly can overflow the real
    # budget. ASCII-only text is overwhelmingly common and can use the cheap
    # legacy estimate without scanning every character.
    length = len(text)
    if length == 0:
        return 0
    if text.isascii():
        return length // _CHARS_PER_TOKEN + 1
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = non_ascii / length
    if ratio >= 0.5:
        divisor = 1.5
    elif ratio >= 0.2:
        divisor = 2.5
    else:
        divisor = _CHARS_PER_TOKEN
    return int(length / divisor) + 1


def _count_tokens_core(text) -> int:
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    if isinstance(text, str):
        return _fallback_token_estimate(text)
    return len(text) // _CHARS_PER_TOKEN + 1


@lru_cache(maxsize=2048)
def _count_tokens_cached(text: str) -> int:
    # tiktoken encoding is the dominant per-turn cost: assembly and preflight
    # re-count the same content many times per turn. The encoder is decided
    # once per process (see _get_encoder), so a content-keyed cache is stable.
    return _count_tokens_core(text)


# Cap what the LRU may retain by reference. Very large strings are the ones
# least likely to recur identically, and caching them would still let the
# bounded LRU pin unnecessary memory; count those uncached (cost is
# proportional to size either way).
_MAX_CACHEABLE_TOKEN_TEXT_CHARS = 32_768


def count_tokens(text) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    # Only strings are memoized. Callers may pass non-string, unhashable values
    # (e.g. tool_call arguments as a dict); preserve the legacy tolerance by
    # counting those uncached rather than feeding them to the LRU.
    if isinstance(text, str) and len(text) <= _MAX_CACHEABLE_TOKEN_TEXT_CHARS:
        return _count_tokens_cached(text)
    return _count_tokens_core(text)


def truncate_text_to_tokens(text: str, max_tokens: int, *, from_end: bool = False) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens.

    Keeps the head, or the tail when ``from_end`` is set. Uses the tiktoken
    encoder for an exact cut when available, so the result honours the token
    budget even for CJK / dense scripts — a flat ``chars * 4`` budget overshoots
    those ~2-4x. Falls back to a script-density-aware char budget (the inverse of
    :func:`_fallback_token_estimate`) when tiktoken is unavailable.
    """
    if max_tokens <= 0 or not text:
        return ""
    enc = _get_encoder()
    if enc is not None:
        try:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            kept = tokens[-max_tokens:] if from_end else tokens[:max_tokens]
            return enc.decode(kept)
        except Exception:
            pass
    if count_tokens(text) <= max_tokens:
        return text
    length = len(text)
    non_ascii = 0 if text.isascii() else sum(1 for ch in text if ord(ch) > 127)
    ratio = (non_ascii / length) if length else 0.0
    if ratio >= 0.5:
        divisor = 1.5
    elif ratio >= 0.2:
        divisor = 2.5
    else:
        divisor = _CHARS_PER_TOKEN
    char_budget = max(1, int(max_tokens * divisor))
    # The estimate is approximate; correct any overshoot in a few bounded steps
    # so the returned slice never exceeds the token budget.
    for _ in range(8):
        candidate = text[-char_budget:] if from_end else text[:char_budget]
        estimated = count_tokens(candidate)
        if estimated <= max_tokens or char_budget <= 1:
            return candidate
        char_budget = max(1, int(char_budget * max_tokens / estimated) - 1)
    return text[-char_budget:] if from_end else text[:char_budget]


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens for a single OpenAI-format message."""
    total = 4  # role + overhead
    content = normalize_content_value(msg.get("content")) or ""
    total += count_tokens(content)
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            total += count_tokens(fn.get("name", ""))
            total += count_tokens(fn.get("arguments", ""))
        total += 3  # per-call overhead
    return total


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens for a message list."""
    return sum(count_message_tokens(m) for m in messages)
