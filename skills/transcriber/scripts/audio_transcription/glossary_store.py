"""Словарь, который скилл ведёт сам: заводит при первом прогоне и пополняет.

Зачем: выигрыш словаря измерим (WER 6,3% → 4,9% на голосовых), но он достаётся
только тому, кто сядет и заведёт файл руками. Здесь тот же цикл — снять
кандидатов из очереди проверки, накопить доказательства, включить замену —
выполняется без участия человека.

Чего автоматика принципиально не может: назвать каноническое написание термина,
который обе модели переврали («Treds» против «трэдс»). Такие места копятся в
разделе `pending` и ждут человека — выдумывать написание скилл не станет.

Устройство файла (`~/.transcriber/glossary.yaml` по умолчанию):

    version: 1
    entries:            # то же, что в обычном словаре: canonical/aliases/auto_apply
      - canonical: Threads
        aliases: [трэц, трэдс]
        auto_apply: true
        learned:        # служебный блок: на чём основано решение
          runs: 2
          windows: 4
          first_seen: 2026-09-17
          last_seen: 2026-09-18
          auto_apply_written: true
    pending:            # термин слышно, правильного написания нет ни у кого
      - aliases: [трэдс, treds]
        runs: 1
    pruned:             # снято уборкой (`prune`) с причиной; заново не заводится
      - canonical: Zon
        aliases: [там]
        reason: вариант встречается как обычное слово

`entries` читает обычный `load_glossary`, `pending` и `learned` он игнорирует.

Файл машинный: скилл его перезаписывает целиком, комментарии в нём не выживают.
Свои выверенные термины держите отдельным словарём и передавайте `--glossary` —
его скилл только читает.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .glossary import GlossaryEntry, parse_entries
from .mining import collect_candidates
from .phonetic import phonetic_similarity
from .reconcile import normalize_text

# libyaml разбирает словарь на 127 КБ за 0,04 с против 0,26 с у чистого Python,
# а читается он дважды за прогон. Без libyaml остаётся безопасный загрузчик.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

DEFAULT_STORE = Path.home() / ".transcriber" / "glossary.yaml"
# Именованные словари: рабочие встречи и личные голосовые не должны учить друг
# друга. `--glossary-profile work` ведёт ~/.transcriber/glossaries/work.yaml.
PROFILES_DIR = Path.home() / ".transcriber" / "glossaries"
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

# Порог, заменяющий человеческое подтверждение. Смысл не в числах самих по
# себе, а в том, что случайная ослышка до него не доживает: она приходит из
# одной записи и одного окна, а термин повторяется в разных записях.
PROMOTE_MIN_RUNS = 3
PROMOTE_MIN_WINDOWS = 3
# Насколько алиас обязан звучать как канон. Ослышка всегда похожа на то, что
# сказали, поэтому пары, звучанием не объяснимые, — не термин, а совпадение:
# «значит» против «GPT» даёт 0,25, «слзов» против «sales» — 0,20.
#
# Верхней границы нет намеренно, хотя соблазн есть: ровно 1,0 означает, что
# модели услышали одно и то же и разошлись только алфавитом. В этом классе
# сидят и вредные пары («алиса» → «ALISA», «клод» → «CLOT»), и главный
# полезный («обсидиан» → «Obsidian»). Замер по 66 записям показал, чем они
# различаются на деле: вредные приходят из одной записи — так проверяющая
# модель капитализирует то, в чём не уверена, — а название продукта
# повторяется. Поэтому их разделяет требование трёх разных записей, а не
# похожесть; граница по похожести отрезала бы ровно ту замену, ради которой
# словарь и заводят.
PROMOTE_MIN_PHONETIC = 0.6
# Короткий алиас не повышаем: «SMM» и «SMS» на слух неразличимы, а литеральная
# замена трёхбуквенного куска бьёт по слишком многим словам. Та же граница, что
# у фонетического поиска в glossary.MIN_PHONETIC_LETTERS.
PROMOTE_MIN_ALIAS_LETTERS = 4

HEADER = """\
# Словарь терминов, который скилл ведёт сам.
#
# Пополняется после каждого прогона из расхождений двух моделей. Запись
# начинает править текст (auto_apply: true), когда термин повторился в разных
# записях: до тех пор он только подсвечивается в review-needed.md.
#
# Файл машинный — скилл перезаписывает его целиком, комментарии не сохраняются.
# Свои выверенные термины держите отдельным файлом и передавайте --glossary:
# его скилл только читает и никогда не меняет.
#
# Правки здесь уважаются: если поменять auto_apply руками, скилл больше не
# трогает это поле. Чтобы убрать термин навсегда, удалите запись и запустите
# прогон с --no-learn, иначе он вернётся из следующего расхождения.
#
# auto_promote: false в корне файла — скилл копит кандидатов и доказательства,
# но автозамену не включает никогда: только вы, полем auto_apply.
"""


@dataclass
class LearnReport:
    """Что изменилось в словаре за прогон — для stderr и glossary-audit.json."""

    added: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    reinforced: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    total_entries: int = 0
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "promoted": self.promoted,
            "reinforced": self.reinforced,
            "pending": self.pending,
            "blocked": self.blocked,
            "demoted": self.demoted,
            "pruned": self.pruned,
            "total_entries": self.total_entries,
            "path": self.path,
        }

    def summary(self) -> str | None:
        """Одна строка в stderr. None — если рассказывать не о чем."""
        parts = []
        if self.added:
            parts.append(f"новых терминов {len(self.added)}")
        if self.promoted:
            parts.append(f"включены в автозамену: {', '.join(self.promoted)}")
        if self.demoted:
            parts.append(f"сняты с автозамены: {', '.join(self.demoted)}")
        if self.pending:
            parts.append(f"ждут написания {len(self.pending)}")
        if self.pruned:
            parts.append(f"убрано мусорных {len(self.pruned)}")
        if not parts:
            return None
        return f"Словарь: {'; '.join(parts)} (всего записей {self.total_entries})"


def profile_path(name: str) -> Path:
    if not _PROFILE_NAME.fullmatch(name):
        raise ValueError(
            f"Имя профиля словаря «{name}»: только строчные латинские буквы, цифры, - и _"
        )
    return PROFILES_DIR / f"{name}.yaml"


def store_path(explicit: str | Path | None = None, profile: str | None = None) -> Path:
    """Где лежит авто-словарь.

    Порядок: явный путь → профиль → TRANSCRIBER_GLOSSARY_STORE →
    TRANSCRIBER_GLOSSARY_PROFILE → общий домашний.
    """
    if explicit:
        return Path(explicit).expanduser()
    if profile:
        return profile_path(profile)
    from_env = os.environ.get("TRANSCRIBER_GLOSSARY_STORE")
    if from_env:
        return Path(from_env).expanduser()
    profile_env = os.environ.get("TRANSCRIBER_GLOSSARY_PROFILE")
    if profile_env:
        return profile_path(profile_env.strip())
    return DEFAULT_STORE


def load_document(path: Path) -> dict[str, Any]:
    """Читает файл словаря. Нет файла — пустой документ, а не ошибка."""
    if not path.is_file():
        return {"version": 1, "entries": [], "pending": []}
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Словарь {path} должен быть отображением с ключом entries")
    payload.setdefault("version", 1)
    payload.setdefault("entries", [])
    payload.setdefault("pending", [])
    payload.setdefault("pruned", [])
    if not all(isinstance(payload[key], list) for key in ("entries", "pending", "pruned")):
        raise ValueError(f"В словаре {path} entries, pending и pruned должны быть списками")
    return payload


def entries_of(document: dict[str, Any]) -> list[GlossaryEntry]:
    """Записи документа в том же виде, что у обычного словаря."""
    return parse_entries(document.get("entries", []))


# Сколько отпечатков записей помнить на термин. Нужны они только чтобы отличить
# новую запись от повторного прогона старой, поэтому храним хвост, а не всё.
SOURCE_MEMORY = 12


def _mark_source(learned: dict[str, Any], source: str | None) -> bool:
    """Отмечает, что термин встретился в этой записи. True — запись новая.

    Считать надо именно разные записи, а не запуски: повторный прогон того же
    файла — обычное дело (кэш гипотез делает его дешёвым), и если считать
    запуски, случайная ослышка доберёт порог простым повтором команды.
    """
    sources = [str(item) for item in learned.get("sources", [])]
    fresh = source not in sources if source else not sources
    if fresh:
        sources.append(source or "unknown")
        learned["sources"] = sources[-SOURCE_MEMORY:]
    learned["runs"] = len(learned.get("sources", sources))
    return fresh


def _alias_letters(alias: str) -> int:
    return sum(1 for char in alias if char.isalpha())


def _alias_problem(alias: str, canonical: str, blocked: set[str]) -> str | None:
    """Почему этот вариант нельзя заменять молча. None — можно."""
    if _alias_letters(alias) < PROMOTE_MIN_ALIAS_LETTERS:
        return "слишком короткий вариант"
    if normalize_text(alias) in blocked:
        return "встречается как обычное слово"
    score = phonetic_similarity(alias, canonical)
    if score < PROMOTE_MIN_PHONETIC:
        return f"звучит не как «{canonical}» ({score:.2f})"
    return None


def _hold_unsafe_aliases(
    entry: dict[str, Any], blocked: set[str], report: LearnReport
) -> None:
    """Держит автозамену записи на тех же тормозах, что и её включение.

    Порог проверяет варианты один раз, в момент повышения, а записи копят их
    и дальше. 2026-09-23 на повторе по 161 записи `Bittrex` после повышения
    подобрал «в» и «вот»: литеральная замена переписала бы каждое «в» в тексте.
    Поэтому каждый прогон варианты включённой записи проверяются заново;
    непрошедшие уходят в `learned.held_aliases` — видно, но не заменяется.
    Не осталось ни одного — запись снимается с автозамены.
    """
    learned = entry.setdefault("learned", {})
    canonical = str(entry["canonical"])
    safe, held = [], [str(item) for item in learned.get("held_aliases", [])]
    for alias in entry.get("aliases", []):
        if _alias_problem(str(alias), canonical, blocked):
            if alias not in held:
                held.append(alias)
        else:
            safe.append(alias)
    if len(safe) == len(entry.get("aliases", [])):
        return
    entry["aliases"] = safe
    learned["held_aliases"] = held
    if not safe:
        entry["auto_apply"] = False
        learned["auto_apply_written"] = False
        learned["held_back"] = "не осталось вариантов, которые можно заменять молча"
        report.demoted.append(canonical)


def _promotable(entry: dict[str, Any], blocked: set[str]) -> tuple[bool, str | None]:
    """Пора ли записи начать править текст. Вторым значением — что помешало."""
    learned = entry.get("learned", {})
    if not entry.get("canonical"):
        return False, "нет канонического написания"
    if not learned.get("latin"):
        # Канон подтверждён только там, где его кто-то написал латиницей.
        # Кириллическое расхождение могло повториться и просто потому, что
        # модель одинаково ослышалась в похожих местах.
        return False, "написание не подтверждено латиницей"
    aliases = [str(alias) for alias in entry.get("aliases", [])]
    if not aliases:
        return False, "нет вариантов распознавания"
    short = [alias for alias in aliases if _alias_letters(alias) < PROMOTE_MIN_ALIAS_LETTERS]
    if short:
        return False, f"слишком короткий вариант: {', '.join(short)}"
    collisions = [alias for alias in aliases if normalize_text(alias) in blocked]
    if collisions or learned.get("seen_as_word"):
        # Слово встретилось там, где обе модели согласны, — значит это обычное
        # слово языка, а не ослышка. Замена «трек» → «Трекер» испортила бы
        # текст молча, и заметить это можно было бы только сверкой с raw.json.
        return False, f"встречается как обычное слово: {', '.join(collisions) or '—'}"
    for alias in aliases:
        score = phonetic_similarity(alias, str(entry["canonical"]))
        if score < PROMOTE_MIN_PHONETIC:
            return False, f"«{alias}» звучит не как «{entry['canonical']}» ({score:.2f})"
    if int(learned.get("runs", 0)) < PROMOTE_MIN_RUNS:
        return False, "мало записей"
    if int(learned.get("windows", 0)) < PROMOTE_MIN_WINDOWS:
        return False, "мало окон"
    return True, None


def _edited_by_hand(entry: dict[str, Any]) -> bool:
    """Правил ли человек auto_apply после нас.

    Сравниваем текущее значение с тем, которое записали сами. Разошлись —
    дальше это поле не наше: автоматика не должна переспоривать человека.
    """
    learned = entry.get("learned")
    if not isinstance(learned, dict) or "auto_apply_written" not in learned:
        # Запись пришла не от нас (например, человек добавил её руками).
        return True
    return bool(entry.get("auto_apply", False)) != bool(learned["auto_apply_written"])


def learn(
    document: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    blocked: set[str] | None = None,
    source: str | None = None,
    today: date | None = None,
) -> tuple[dict[str, Any], LearnReport]:
    """Вносит кандидатов прогона в документ словаря. Ввода-вывода не делает."""
    today = today or date.today()
    blocked = blocked or set()
    report = LearnReport()
    entries: list[dict[str, Any]] = [dict(item) for item in document.get("entries", [])]
    pending: list[dict[str, Any]] = [dict(item) for item in document.get("pending", [])]
    by_canonical = {
        normalize_text(str(item.get("canonical", ""))): item for item in entries
    }
    # Человек мог назвать термину другое написание: «Астра» пишется
    # по-русски, а проверяющая модель снова и снова пишет «Astra». Если
    # чужой канон уже стоит вариантом в записи, которую правил человек, это
    # та же запись, иначе скилл вырастил бы рядом прежнюю.
    for item in entries:
        if _edited_by_hand(item):
            for alias in item.get("aliases", []):
                by_canonical.setdefault(normalize_text(str(alias)), item)
    # Снятое уборкой не возвращается из следующего расхождения: иначе «E ← и»
    # вырастало бы заново после каждого прогона.
    pruned_canonicals = {
        normalize_text(str(item.get("canonical", "")))
        for item in document.get("pruned", [])
        if isinstance(item, dict)
    }
    pending_by_alias: dict[str, dict[str, Any]] = {}
    for item in pending:
        for alias in item.get("aliases", []):
            pending_by_alias.setdefault(normalize_text(str(alias)), item)

    for candidate in candidates:
        canonical = str(candidate.get("canonical", "")).strip()
        aliases = [str(alias) for alias in candidate.get("aliases", []) if str(alias).strip()]
        windows = int(candidate.get("count", 1))
        known_entry = next(
            (by_canonical[normalize_text(alias)] for alias in aliases
             if normalize_text(alias) in by_canonical
             and _edited_by_hand(by_canonical[normalize_text(alias)])),
            None,
        )
        if known_entry is not None and (
            not canonical or normalize_text(canonical) not in by_canonical
        ):
            # Написание этому месту человек уже назвал — ждать больше нечего,
            # а своё написание проверяющей («BITREX» против ручного
            # «Битрикс24») — не повод растить рядом вторую запись.
            canonical = str(known_entry["canonical"])
        if canonical and normalize_text(canonical) in pruned_canonicals:
            continue
        if not canonical:
            # Написания нет ни у одной модели — назвать его может только
            # человек. Копим наблюдения, чтобы он увидел, насколько часто.
            existing = next(
                (pending_by_alias[normalize_text(alias)] for alias in aliases
                 if normalize_text(alias) in pending_by_alias),
                None,
            )
            if existing is None:
                item = {
                    "aliases": aliases,
                    "windows": windows,
                    "first_seen": today.isoformat(),
                    "last_seen": today.isoformat(),
                }
                _mark_source(item, source)
                pending.append(item)
                for alias in aliases:
                    pending_by_alias[normalize_text(alias)] = item
                report.pending.append(aliases[0] if aliases else "?")
            else:
                for alias in aliases:
                    if alias not in existing["aliases"]:
                        existing["aliases"].append(alias)
                        pending_by_alias[normalize_text(alias)] = existing
                existing["last_seen"] = today.isoformat()
                if _mark_source(existing, source):
                    existing["windows"] = int(existing.get("windows", 0)) + windows
            continue

        key = normalize_text(canonical)
        entry = by_canonical.get(key)
        if entry is None:
            entry = {
                "canonical": canonical,
                "aliases": aliases,
                "context": f"выучено скиллом, первое окно {float(candidate.get('first_at', 0.0)):.1f} с",
                "auto_apply": False,
                "case_sensitive": False,
                "learned": {
                    "windows": windows,
                    "latin": bool(candidate.get("latin", False)),
                    "first_seen": today.isoformat(),
                    "last_seen": today.isoformat(),
                    "auto_apply_written": False,
                },
            }
            _mark_source(entry["learned"], source)
            entries.append(entry)
            by_canonical[key] = entry
            report.added.append(canonical)
        else:
            learned = entry.setdefault("learned", {})
            dropped = {normalize_text(str(item)) for item in learned.get("pruned_aliases", [])}
            for alias in aliases:
                if normalize_text(alias) in dropped:
                    continue
                if not _edited_by_hand(entry) and alias not in entry.get("aliases", []):
                    entry.setdefault("aliases", []).append(alias)
            learned["latin"] = bool(learned.get("latin", False)) or bool(
                candidate.get("latin", False)
            )
            learned["last_seen"] = today.isoformat()
            learned.setdefault("first_seen", today.isoformat())
            if _mark_source(learned, source):
                learned["windows"] = int(learned.get("windows", 0)) + windows
            report.reinforced.append(canonical)

    for entry in entries:
        learned = entry.get("learned")
        if not isinstance(learned, dict):
            continue
        if any(normalize_text(str(alias)) in blocked for alias in entry.get("aliases", [])):
            # Флаг вечный: слово, однажды встреченное вне спора, остаётся
            # обычным словом языка и в тех записях, где спор о нём был.
            learned["seen_as_word"] = True

    for entry in entries:
        if _edited_by_hand(entry):
            continue
        if entry.get("auto_apply"):
            _hold_unsafe_aliases(entry, blocked, report)
            continue
        ready, reason = _promotable(entry, blocked)
        if ready and document.get("auto_promote") is False:
            # Хозяин словаря включает замены сам: доказательства копятся, но
            # текст не правится, пока он не поставит auto_apply руками.
            ready, reason = False, "автозамену включает человек (auto_promote: false)"
        if ready:
            entry["auto_apply"] = True
            entry["learned"]["auto_apply_written"] = True
            report.promoted.append(str(entry["canonical"]))
        elif reason and entry.get("canonical") in report.added + report.reinforced:
            entry.setdefault("learned", {})["held_back"] = reason
            if "встречается как обычное слово" in reason:
                report.blocked.append(str(entry["canonical"]))

    document = dict(document)
    document["version"] = document.get("version", 1)
    document["entries"] = entries
    document["pending"] = pending
    document, pruned = prune(document, today=today)
    report.pruned = [str(item.get("canonical")) for item in pruned.removed]
    report.total_entries = len(document["entries"])
    return document, report


# Вариант короче трёх букв — «и», «в», «с», «НА». Такой запись не сделает
# термином никогда: литеральная замена двухбуквенного куска бьёт по тексту.
PRUNE_MAX_ALIAS_LETTERS = 2


@dataclass
class PruneReport:
    removed: list[dict[str, Any]] = field(default_factory=list)
    trimmed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"removed": self.removed, "trimmed": self.trimmed}


def _junk_alias(alias: str, canonical: str, seen_as_word: bool) -> str | None:
    """Почему вариант — совпадение, а не ослышка термина. None — ослышка.

    Одного флага «встречается как обычное слово» мало: «антропик» обе модели
    пишут кириллицей, и Anthropic от этого не перестаёт быть термином. Мусор
    отличает второе условие — вариант не звучит как канон («там» → «Zon»).
    """
    if sum(char.isalnum() for char in alias) <= PRUNE_MAX_ALIAS_LETTERS:
        return "короче трёх знаков"
    if seen_as_word:
        score = phonetic_similarity(alias, canonical)
        if score < PROMOTE_MIN_PHONETIC:
            return f"обычное слово, не похожее на «{canonical}» ({score:.2f})"
    return None


def prune(
    document: dict[str, Any], *, today: date | None = None
) -> tuple[dict[str, Any], PruneReport]:
    """Уборка словаря: мусор, дубли ручных записей и спорные варианты.

    Ревью 2026-09-27: из 195 записей треть не могла стать термином никогда
    («E ← и», «Zon ← там», «M ← угу»), а подсветку в review-needed.md давала.
    Ручные записи не трогаются. Снятое не удаляется, а уезжает в `pruned` с
    причиной — и оттуда же `learn` узнаёт, что заводить это заново не надо.
    """
    today = today or date.today()
    report = PruneReport()
    entries = [dict(item) for item in document.get("entries", [])]
    hand = [item for item in entries if _edited_by_hand(item)]
    hand_terms: dict[str, str] = {}
    for item in hand:
        for term in (item.get("canonical", ""), *item.get("aliases", [])):
            hand_terms.setdefault(normalize_text(str(term)), str(item.get("canonical")))

    def drop(item: dict[str, Any], reason: str) -> None:
        report.removed.append(
            {
                "canonical": item.get("canonical"),
                "aliases": list(item.get("aliases", [])),
                "reason": reason,
                "pruned_on": today.isoformat(),
            }
        )

    kept: list[dict[str, Any]] = []
    for item in entries:
        if item in hand:
            kept.append(item)
            continue
        owner = next(
            (
                hand_terms[normalize_text(str(term))]
                for term in (item.get("canonical", ""), *item.get("aliases", []))
                if normalize_text(str(term)) in hand_terms
            ),
            None,
        )
        if owner is not None:
            drop(item, f"поглощена ручной записью «{owner}»")
            continue
        if item.get("auto_apply") or (item.get("learned") or {}).get("held_aliases"):
            # Включённую запись держит `_hold_unsafe_aliases` на своих порогах,
            # а снятая им с автозамены остаётся на виду: она была термином.
            kept.append(item)
            continue
        learned = item.setdefault("learned", {})
        canonical = str(item.get("canonical", ""))
        junk = {
            str(alias): reason
            for alias in item.get("aliases", [])
            if (reason := _junk_alias(str(alias), canonical, bool(learned.get("seen_as_word"))))
        }
        if len(junk) == len(item.get("aliases", [])):
            drop(item, "нет вариантов" if not junk else f"все варианты — мусор: {'; '.join(f'{a} — {r}' for a, r in junk.items())}")
            continue
        if junk:
            item["aliases"] = [alias for alias in item["aliases"] if str(alias) not in junk]
            learned["pruned_aliases"] = sorted({*map(str, learned.get("pruned_aliases", [])), *junk})
            report.trimmed.append({"canonical": canonical, "aliases": sorted(junk)})
        kept.append(item)

    # Один вариант — у одной записи: иначе замена зависит от порядка в файле.
    # Хозяин — ручная запись, затем включённая, затем набравшая больше окон.
    def rank(item: dict[str, Any]) -> tuple[int, int, int]:
        learned = item.get("learned") or {}
        return (
            item in hand,
            bool(item.get("auto_apply")),
            int(learned.get("windows", 0)),
        )

    owners: dict[str, dict[str, Any]] = {}
    for item in sorted(kept, key=rank, reverse=True):
        for alias in item.get("aliases", []):
            owners.setdefault(normalize_text(str(alias)), item)
    result: list[dict[str, Any]] = []
    for item in kept:
        if item in hand:
            result.append(item)
            continue
        foreign = [
            str(alias)
            for alias in item.get("aliases", [])
            if owners[normalize_text(str(alias))] is not item
        ]
        if foreign:
            learned = item.setdefault("learned", {})
            item["aliases"] = [alias for alias in item.get("aliases", []) if str(alias) not in foreign]
            learned["pruned_aliases"] = sorted(
                {*map(str, learned.get("pruned_aliases", [])), *foreign}
            )
            report.trimmed.append({"canonical": item.get("canonical"), "aliases": foreign})
            if not item["aliases"]:
                drop(item, "все варианты принадлежат другим записям")
                continue
        result.append(item)

    document = dict(document)
    document["entries"] = result
    document["pruned"] = [*document.get("pruned", []), *report.removed]
    return document, report


def save_document(path: Path, document: dict[str, Any]) -> None:
    """Пишет словарь атомарно: прогон могут прервать, файл переживёт.

    Каталог создаётся при первом прогоне — это и есть «завести словарь».
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        document, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(HEADER)
            handle.write(body)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def record_fingerprint(sha256: str, origin: str | None = None) -> str:
    """Чем одна запись отличается от другой для порога «три разных записи».

    По умолчанию — звук. Но у ролика по ссылке звук не единственный: его
    качают заново, режут на клипы, берут левый канал, и каждый такой файл даёт
    новый отпечаток. Три клипа одного спикера добирали бы порог в одиночку,
    поэтому у записи по ссылке отпечаток — адрес.
    """
    if origin:
        return hashlib.sha256(origin.strip().encode("utf-8")).hexdigest()[:12]
    return sha256[:12]


def update_from_run(
    path: Path,
    review_items: list[dict[str, Any]],
    *,
    blocked: set[str] | None = None,
    curated: set[str] | None = None,
    source: str | None = None,
    today: date | None = None,
) -> tuple[list[GlossaryEntry], LearnReport]:
    """Полный цикл после прогона: снять кандидатов, дополнить словарь, сохранить.

    Возвращает записи уже обновлённого словаря — чтобы выученное применилось
    к текущей расшифровке, а не только к следующей.

    Из отбора исключается только выверенный словарь пользователя: там решение
    уже принято. Свои записи, наоборот, должны приходить снова и снова —
    повторная встреча термина и есть то доказательство, по которому запись
    когда-нибудь начнёт править текст.
    """
    document = load_document(path)
    candidates = collect_candidates(review_items, curated or set())
    document, report = learn(
        document, candidates, blocked=blocked, source=source, today=today
    )
    report.path = str(path)
    save_document(path, document)
    return entries_of(document), report
