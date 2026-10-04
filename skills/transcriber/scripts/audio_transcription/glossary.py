from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from .models import Correction, ReviewItem, Segment
from .phonetic import phonetic_similarity
from .reconcile import normalize_text

# libyaml разбирает словарь на 127 КБ за 0,04 с против 0,26 с у чистого Python,
# а читается он дважды за прогон. Без libyaml остаётся безопасный загрузчик.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


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
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
    return parse_entries(payload)


def parse_entries(payload: Any) -> list[GlossaryEntry]:
    """Разбор записей словаря отдельно от чтения файла.

    Тем же разбором пользуется `glossary_store`: словарь, который скилл ведёт
    сам, держит записи в том же формате, но живёт в документе с лишними
    ключами (`pending`, `learned`) и уже разобранным YAML.
    """
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


def merge_glossaries(
    curated: list[GlossaryEntry], learned: list[GlossaryEntry]
) -> list[GlossaryEntry]:
    """Словарь пользователя поверх того, что скилл выучил сам.

    Выученное не должно спорить с выверенным: если алиас или канон уже заняты
    записью пользователя, из автоматической записи они вычёркиваются, а
    оставшаяся без вариантов запись не попадает в работу вовсе. Иначе один
    и тот же кусок текста претендовали бы исправить две записи, и победила
    бы та, что оказалась длиннее.
    """
    taken = {entry.canonical.casefold() for entry in curated}
    taken.update(alias.casefold() for entry in curated for alias in entry.aliases)
    result = list(curated)
    for entry in learned:
        if entry.canonical.casefold() in taken:
            continue
        aliases = tuple(alias for alias in entry.aliases if alias.casefold() not in taken)
        if not aliases:
            continue
        result.append(
            GlossaryEntry(
                canonical=entry.canonical,
                aliases=aliases,
                auto_apply=entry.auto_apply,
                case_sensitive=entry.case_sensitive,
                context=entry.context,
            )
        )
    return result


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
            # Канон обычно стоит и среди алиасов: «MCP» в тексте совпадает с
            # алиасом «mcp», но править нечего. В журнале такие места были
            # двумя третями «исправлений» (95 из 151 за 2026-09-24..10-04).
            if before == entry.canonical:
                continue
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
    only_words: set[str] | None = None,
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

    `only_words` сужает поиск до слов, о которых модели спорили. Без него
    фонетика перебирает весь текст против всего словаря: на словаре из 168
    записей это 11 с на пятиминутном ролике и 28 с на 17-минутной встрече, а
    на ролике 18 подсказок из 19 оказались ложными («так → Docs»). Слово, в
    котором обе модели сошлись, ослышкой термина почти не бывает.
    """
    suggestions: list[dict[str, Any]] = []
    seen: set[tuple[float, str, str]] = set()
    for segment in segments:
        words = segment.text.split()
        disputed = [
            only_words is None or bool(set(normalize_text(word).split()) & only_words)
            for word in words
        ]
        if not any(disputed):
            continue
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
                    if not any(disputed[start : start + width]):
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


# Порог по звучанию между тем, что написала основная модель, и каноном. На
# 72 записях (2026-10-02) все верные места лежали от 0,67 («клот» → Claude,
# «MC» → MCP) до 1,0, ложные — не выше 0,50 («СберАстра» → «Астра»,
# «Dots» → «Docs», где права как раз основная).
VERIFIER_CANONICAL_SIMILARITY = 0.6
VERIFIER_CANONICAL_KINDS = frozenset({"substantive", "spelling"})


def _best_piece(words: list[str], canonical: str) -> tuple[int, int, float]:
    """Кусок фрагмента основной модели, сильнее всего похожий на канон.

    Расхождение часто шире термина («э-э-э в ВКрософт» против «пока как
    Microsoft»): заменять надо одно слово, а не всю фразу вокруг.
    """
    best = (0, 0, 0.0)
    # Короткие куски первыми, длинный побеждает только строго большим
    # сходством: у «ВКрософт» и «в ВКрософт» оно одинаковое (0,78), и
    # предлог иначе уходил вместе с ослышкой.
    for size in range(1, len(canonical.split()) + 2):
        for start in range(len(words) - size + 1):
            piece = " ".join(_trim(word) for word in words[start : start + size]).strip()
            if len(piece) < 2:
                continue
            score = phonetic_similarity(piece, canonical)
            if score > best[2]:
                best = (start, start + size, score)
    return best


def take_verifier_canonicals(
    segments: list[Segment],
    review_items: list[ReviewItem],
    entries: list[GlossaryEntry],
    *,
    threshold: float = VERIFIER_CANONICAL_SIMILARITY,
) -> tuple[list[Segment], list[ReviewItem], list[Correction]]:
    """Берёт термин у проверяющей, когда она написала канон доверенной записи.

    Подставлять текст проверяющей вообще нельзя: на eval40 это почти удвоило
    ошибку (5,9% → 10,7%), расхождение значит «послушай», а не «вторая права».
    Здесь уже не голос против голоса: проверяющая написала ровно то, что
    человек или автоматика в словаре утвердили как канон, а основная на этом
    месте написала похожее по звучанию («Макросов» → Microsoft, «MC» → MCP).
    Поэтому доверие — только записям с `auto_apply`, кусок основной модели не
    должен быть другим термином словаря («SMM» против «SMS»), и меняется
    только он, а не весь фрагмент. Закрытое место остаётся в очереди с видом
    `verifier_canonical`: из отчёта уходит, след и обучение словаря — нет.
    """
    trusted = [entry for entry in entries if entry.auto_apply]
    known = {
        normalize_text(term)
        for entry in entries
        for term in (entry.canonical, *entry.aliases)
    }
    texts = [segment.text for segment in segments]
    taken: list[Correction] = []
    items: list[ReviewItem] = []
    for item in review_items:
        fix = None
        if item.kind in VERIFIER_CANONICAL_KINDS and item.primary_span and item.verifier_span:
            words = item.primary_span.split()
            for entry in trusted:
                # Фрагмент шире термина — выравнивание рыхлое, и термин
                # проверяющей может стоять против другого слова: «истории к
                # Dots» против «Docs и», где права основная (2026-10-02).
                if len(words) > len(entry.canonical.split()) + 1:
                    continue
                flags = 0 if entry.case_sensitive else re.IGNORECASE
                pattern = _pattern(entry.canonical)
                if not re.search(pattern, item.verifier_span, flags=flags):
                    continue
                if re.search(pattern, item.primary_span, flags=flags):
                    continue
                start, end, score = _best_piece(words, entry.canonical)
                if score < threshold:
                    continue
                piece = " ".join(_trim(word) for word in words[start:end]).strip()
                if normalize_text(piece) in known:
                    continue
                if fix is None or score > fix[2]:
                    fix = (entry, piece, score)
        index = None
        if fix is not None:
            entry, piece, _ = fix
            index = next(
                (
                    i
                    for i, segment in enumerate(segments)
                    if segment.start <= item.start < segment.end
                    and re.search(_pattern(piece), texts[i])
                ),
                None,
            )
        if fix is None or index is None:
            items.append(item)
            continue
        texts[index] = re.sub(_pattern(piece), lambda _: entry.canonical, texts[index], count=1)
        taken.append(
            Correction(
                start=segments[index].start,
                end=segments[index].end,
                before=piece,
                after=entry.canonical,
                canonical=entry.canonical,
                rule="verifier_canonical",
                automatic=True,
            )
        )
        items.append(replace(item, kind="verifier_canonical"))
    corrected = [
        segment if text == segment.text else replace(segment, text=text)
        for segment, text in zip(segments, texts)
    ]
    return corrected, items, taken


_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_FOREIGN_LETTER = re.compile(r"[^\W\dA-Za-zА-Яа-яЁё_]")
VERIFIER_ONLY_MAX_WORDS = 4


def mark_verifier_only_terms(
    review_items: list[ReviewItem],
    entries: list[GlossaryEntry],
    *,
    latin_is_signal: bool,
) -> list[ReviewItem]:
    """Отмечает места, где термин есть только у проверяющей, а у основной пусто.

    GigaAM на быстрой диктовке роняет как раз английские названия: «OpenAI
    понижает лимиты» → «понижает лимиты» (Handy, 2026-10-02). Вставлять текст
    проверяющей нельзя: на записях с эталоном 13 таких вставок дали больше
    ошибок, чем сняли, — Whisper на пустом месте дописывает «Продолжение
    следует...» (`experiments/verifier-insertions/`). Поэтому место только
    поднимается в отдельный раздел отчёта, `verifier_only`: текст не меняется,
    а читатель видит, какое название могло выпасть.

    Сигнал — латиница на русском маршруте или термин словаря; длинный фрагмент
    и чужая письменность («바instant») — признак срыва проверяющей, а не слова.
    """
    known = {
        normalize_text(term)
        for entry in entries
        for term in (entry.canonical, *entry.aliases)
    }
    known.discard("")

    def has_term(span: str) -> bool:
        words = normalize_text(span).split()
        return any(
            " ".join(words[start : start + size]) in known
            for size in (1, 2, 3)
            for start in range(len(words) - size + 1)
        )

    marked: list[ReviewItem] = []
    for item in review_items:
        span = item.verifier_span.strip()
        if (
            item.kind == "substantive"
            and not item.primary_span.strip()
            and span
            and len(span.split()) <= VERIFIER_ONLY_MAX_WORDS
            and not _FOREIGN_LETTER.search(span)
            and ((latin_is_signal and _LATIN_WORD.search(span)) or has_term(span))
        ):
            item = replace(item, kind="verifier_only")
        marked.append(item)
    return marked
