"""Local privacy helpers for WhatsApp conversation data."""

from __future__ import annotations

import re

import pandas as pd


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def mask_sensitive_text(text: str, extra_terms=()) -> str:
    """Mask email addresses and phone-like number sequences."""
    masked = EMAIL_PATTERN.sub("[EMAIL]", str(text))

    def mask_phone(match):
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15 and ":" not in candidate and "/" not in candidate:
            return "[PHONE]"
        return candidate

    masked = PHONE_PATTERN.sub(mask_phone, masked)
    for term in extra_terms:
        if str(term).strip():
            masked = re.sub(
                re.escape(str(term).strip()),
                "[REDACTED]",
                masked,
                flags=re.IGNORECASE,
            )
    return masked


def anonymize_participants(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace participant names with stable labels in a copied DataFrame."""
    if "user" not in frame.columns:
        raise ValueError("The conversation data has no participant column named 'user'.")

    result = frame.copy()
    names = sorted(result["user"].dropna().astype(str).unique())
    aliases = {name: f"Participant {index}" for index, name in enumerate(names, 1)}
    result["user"] = result["user"].map(
        lambda value: aliases.get(str(value), value) if pd.notna(value) else value
    )
    return result


def sanitize_conversation(
    frame: pd.DataFrame,
    anonymize: bool = True,
    mask_contacts: bool = True,
    extra_terms=(),
) -> pd.DataFrame:
    """Return a privacy-sanitized copy while preserving the original in memory."""
    result = anonymize_participants(frame) if anonymize else frame.copy()
    if "message" in result.columns:
        result["message"] = result["message"].map(
            lambda value: (
                mask_sensitive_text(
                    value,
                    extra_terms=extra_terms,
                ) if mask_contacts else _redact_terms(value, extra_terms)
            ) if pd.notna(value) else value
        )
    if "Message" in result.columns:
        if {"user", "message"}.issubset(result.columns):
            result["Message"] = result.apply(
                lambda row: f"{row['user']}: {row['message']}", axis=1
            )
        else:
            result["Message"] = result["Message"].map(
                lambda value: (
                    mask_sensitive_text(value, extra_terms=extra_terms)
                    if mask_contacts
                    else _redact_terms(value, extra_terms)
                ) if pd.notna(value) else value
            )
    return result


def _redact_terms(text: str, extra_terms=()) -> str:
    redacted = str(text)
    for term in extra_terms:
        if str(term).strip():
            redacted = re.sub(
                re.escape(str(term).strip()),
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )
    return redacted
