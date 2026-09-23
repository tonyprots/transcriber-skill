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

`entries` читает обычный `load_glossary`, `pending` и `learned` он игнорирует.

Файл машинный: скилл его перезаписывает целиком, комментарии в нём не выживают.
Свои выверенные термины держите отдельным словарём и передавайте `--glossary` —
его скилл только читает.
"""
from __future__ import annotations

import hashlib
import os
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

DEFAULT_STORE = Path.home() / ".transcriber" / "glossary.yaml"

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
        if not parts:
            return None
        return f"Словарь: {'; '.join(parts)} (всего записей {self.total_entries})"


def store_path(explicit: str | Path | None = None) -> Path:
    """Где лежит авто-словарь: явный путь → TRANSCRIBER_GLOSSARY_STORE → домашний."""
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("TRANSCRIBER_GLOSSARY_STORE")
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_STORE


def load_document(path: Path) -> dict[str, Any]:
    """Читает файл словаря. Нет файла — пустой документ, а не ошибка."""
    if not path.is_file():
        return {"version": 1, "entries": [], "pending": []}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Словарь {path} должен быть отображением с ключом entries")
    payload.setdefault("version", 1)
    payload.setdefault("entries", [])
    payload.setdefault("pending", [])
    if not isinstance(payload["entries"], list) or not isinstance(payload["pending"], list):
        raise ValueError(f"В словаре {path} entries и pending должны быть списками")
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
        if not canonical and known_entry is not None:
            # Написание этому месту человек уже назвал — ждать больше нечего.
            canonical = str(known_entry["canonical"])
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
            for alias in aliases:
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
    report.total_entries = len(entries)
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
