from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from .models import Correction, Segment


@dataclass(frozen=True)
class GlossaryEntry:
    canonical: str
    aliases: tuple[str, ...]
    auto_apply: bool = False
    case_sensitive: bool = False
    context: str | None = None


def load_glossary(path: Path | None) -> list[GlossaryEntry]:
    if path is None:
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        raise ValueError("В словаре ожидается список entries")
    entries: list[GlossaryEntry] = []
    for index, item in enumerate(raw_entries, start=1):
        if not isinstance(item, dict) or not str(item.get("canonical", "")).strip():
            raise ValueError(f"Некорректная запись словаря #{index}")
        canonical = str(item["canonical"]).strip()
        aliases_raw = item.get("aliases", [])
        if not isinstance(aliases_raw, list):
            raise ValueError(f"aliases записи #{index} должен быть списком")
        aliases = tuple(str(alias).strip() for alias in aliases_raw if str(alias).strip())
        entries.append(
            GlossaryEntry(
                canonical=canonical,
                aliases=aliases,
                auto_apply=bool(item.get("auto_apply", False)),
                case_sensitive=bool(item.get("case_sensitive", False)),
                context=str(item["context"]).strip() if item.get("context") else None,
            )
        )
    return entries


def _pattern(alias: str) -> str:
    return rf"(?<!\w){re.escape(alias)}(?!\w)"


def apply_glossary(
    segments: list[Segment], entries: list[GlossaryEntry]
) -> tuple[list[Segment], list[Correction]]:
    corrected: list[Segment] = []
    audit: list[Correction] = []
    for segment in segments:
        original = segment.text
        replacements: list[tuple[int, int, GlossaryEntry, str]] = []
        for entry in entries:
            if not entry.auto_apply:
                continue
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            for alias in entry.aliases:
                for match in re.finditer(_pattern(alias), original, flags=flags):
                    replacements.append((match.start(), match.end(), entry, match.group(0)))
        selected: list[tuple[int, int, GlossaryEntry, str]] = []
        for candidate in sorted(replacements, key=lambda item: (item[0], -(item[1] - item[0]))):
            if any(candidate[0] < existing[1] and candidate[1] > existing[0] for existing in selected):
                continue
            selected.append(candidate)
        text = original
        for start, end, entry, before in sorted(
            selected, key=lambda item: (item[0], item[1]), reverse=True
        ):
            text = text[:start] + entry.canonical + text[end:]
            audit.append(
                Correction(
                    start=segment.start,
                    end=segment.end,
                    before=before,
                    after=entry.canonical,
                    canonical=entry.canonical,
                    rule="exact_alias",
                    automatic=True,
                )
            )
        corrected.append(
            Segment(
                segment.start,
                segment.end,
                text,
                confidence=segment.confidence,
                speakers=segment.speakers,
            )
        )
    return corrected, audit


def suggest_glossary_matches(
    segments: list[Segment],
    entries: list[GlossaryEntry],
    *,
    threshold: float = 0.86,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.text.split()
        for entry in entries:
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            if re.search(_pattern(entry.canonical), segment.text, flags=flags):
                continue
            for alias in entry.aliases:
                width = max(1, len(alias.split()))
                for start in range(max(1, len(words) - width + 1)):
                    candidate = " ".join(words[start : start + width])
                    if candidate.casefold() == alias.casefold():
                        continue
                    matcher = SequenceMatcher(None, candidate.casefold(), alias.casefold())
                    # Верхние оценки дешевле ratio(): отсеиваем заведомо далёкие n-граммы.
                    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                        continue
                    score = matcher.ratio()
                    if score >= threshold:
                        suggestions.append(
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "heard": candidate,
                                "alias": alias,
                                "canonical": entry.canonical,
                                "similarity": round(score, 3),
                                "context": entry.context,
                            }
                        )
    return suggestions
