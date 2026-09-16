from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from .models import Correction, Segment
from .phonetic import phonetic_similarity


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


def _trim(word: str) -> str:
    """Убирает пунктуацию по краям: «трец.» должно сравниваться как «трец»."""
    return word.strip(".,!?;:()[]{}«»\"'…-—–")


# Короче этого фонетический код вырождается: у «CFO» и «свою», у «SMM» и «SMS»
# совпадает и костяк согласных, и огрублённые гласные. Такие термины ловятся
# только точным алиасом — лучше промолчать, чем засорять очередь проверки.
MIN_PHONETIC_LETTERS = 4


def _phonetically_comparable(term: str) -> bool:
    return sum(1 for char in term if char.isalpha()) >= MIN_PHONETIC_LETTERS


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
    phonetic_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Ищет в тексте места, похожие на термины словаря, но не совпавшие точно.

    Две независимые меры похожести, и вторая появилась не для симметрии.
    Буквенная (`SequenceMatcher`) ловит опечатку внутри одного алфавита, но
    на главном классе ошибок она бесполезна: русская ASR слышит английский
    термин и пишет его кириллицей, а у «тэгэстат» и «TGStat» нет ни одной
    общей буквы — замер на примерах словаря дал 0 попаданий из 17.

    Фонетическая мера сводит оба алфавита к произношению и те же примеры
    ловит в 13 случаях из 17, давая 2 ложных срабатывания на 553 словах
    обычного текста. Поэтому она работает только на предложения в очередь —
    текст не правится, решение остаётся за человеком.
    """
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[float, str, str]] = set()
    for segment in segments:
        words = segment.text.split()
        for entry in entries:
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            if re.search(_pattern(entry.canonical), segment.text, flags=flags):
                continue
            # Фонетика сравнивает услышанное с каноном: алиасы — это уже
            # записанные ослышки, а ищем мы как раз ещё не записанные.
            targets = [(alias, "alias") for alias in entry.aliases]
            targets.append((entry.canonical, "canonical"))
            for target, kind in targets:
                width = max(1, len(target.split()))
                for start in range(max(1, len(words) - width + 1)):
                    candidate = _trim(" ".join(words[start : start + width]))
                    if not candidate or candidate.casefold() == target.casefold():
                        continue
                    score = 0.0
                    rule = ""
                    if kind == "alias":
                        matcher = SequenceMatcher(
                            None, candidate.casefold(), target.casefold()
                        )
                        # Верхние оценки дешевле ratio(): отсеиваем заведомо далёкие.
                        if (
                            matcher.real_quick_ratio() >= threshold
                            and matcher.quick_ratio() >= threshold
                            and matcher.ratio() >= threshold
                        ):
                            score, rule = matcher.ratio(), "fuzzy_alias"
                    if not rule and _phonetically_comparable(target):
                        phonetic = phonetic_similarity(candidate, target)
                        if phonetic >= phonetic_threshold:
                            score, rule = phonetic, "phonetic"
                    if not rule:
                        continue
                    key = (segment.start, candidate.casefold(), entry.canonical)
                    if key in seen:
                        continue
                    seen.add(key)
                    suggestions.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "heard": candidate,
                            "alias": target,
                            "canonical": entry.canonical,
                            "similarity": round(score, 3),
                            "rule": rule,
                            "context": entry.context,
                        }
                    )
    return suggestions
