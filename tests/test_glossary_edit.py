"""Правка своего словаря скриптом, а не руками по машинному YAML.

Агент правил glossary.yaml руками, и одна опечатка роняла прогоны. Скрипт
пишет то же самое атомарно, переносит занятый вариант и не пускает пары,
которые литеральная замена испортит.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import yaml

from audio_transcription.glossary_store import (
    FORGOTTEN_REASON,
    entries_of,
    load_document,
    parse_pair,
    remember,
    remember_problem,
    update_from_run,
)

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts" / "glossary_edit.py"
spec = importlib.util.spec_from_file_location("glossary_edit", SCRIPT)
glossary_edit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glossary_edit)  # type: ignore[union-attr]


def _run(heard: str, verifier: str) -> list[dict]:
    item = {
        "start": 1.0, "end": 3.0, "readable_text": heard, "comparison_text": verifier,
        "similarity": 0.5, "differing_tokens": [f"{heard} / {verifier}"], "reason": "term",
    }
    return [item, {**item, "start": 30.0, "end": 32.0}]


def _edit(argv: list[str], capsys, stdin: str | None = None) -> tuple[int, dict]:
    code = glossary_edit.main(argv, stdin=io.StringIO(stdin) if stdin is not None else None)
    return code, json.loads(capsys.readouterr().out)


def _learned(canonical: str, aliases: list[str], **learned) -> dict:
    return {
        "canonical": canonical,
        "aliases": aliases,
        "auto_apply": learned.pop("auto_apply", False),
        "case_sensitive": False,
        "learned": {"auto_apply_written": False, "windows": 2, **learned},
    }


def test_pair_separators() -> None:
    assert parse_pair("клод → Claude") == ("клод", "Claude")
    assert parse_pair(" клод -> Claude ") == ("клод", "Claude")
    assert parse_pair("клод => Claude") == ("клод", "Claude")
    assert parse_pair("Битрикс24") == (None, "Битрикс24")


def test_remember_writes_the_file_of_the_language(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_STORE", str(tmp_path / "glossary.yaml"))

    code, report = _edit(["remember", "Cloud Code → Claude Code", "--language", "en"], capsys)

    assert code == 0 and report["applied"] is True
    assert report["store"] == str(tmp_path / "glossary.en.yaml")
    assert not (tmp_path / "glossary.yaml").exists(), "русский словарь не трогается"
    (entry,) = entries_of(load_document(tmp_path / "glossary.en.yaml"))
    assert (entry.canonical, entry.aliases, entry.auto_apply) == ("Claude Code", ("Cloud Code",), True)


def test_taken_alias_moves_and_learning_does_not_bring_it_back(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    store.write_text(
        yaml.safe_dump({"version": 1, "entries": [_learned("CLOT", ["клод"])], "pending": []},
                       allow_unicode=True),
        encoding="utf-8",
    )

    code, _ = _edit(["remember", "клод → Claude", "--store", str(store)], capsys)

    assert code == 0
    document = load_document(store)
    clot, claude = document["entries"]
    assert clot["aliases"] == [] and clot["learned"]["pruned_aliases"] == ["клод"]
    assert claude["aliases"] == ["клод"] and claude["auto_apply"] is True
    assert "learned" not in claude, "запись человека — ручная"
    # Проверяющая снова пишет «CLOT» на том же месте — вариант к ней не вернётся.
    entries, _ = update_from_run(store, _run("клод", "CLOT"), source="rec9")
    assert [(entry.canonical, entry.aliases) for entry in entries if "клод" in entry.aliases] == [
        ("Claude", ("клод",))
    ]


def test_guard_rejects_case_forms_and_force_overrides(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"

    code, report = _edit(["remember", "Яндексу → Яндексе", "--store", str(store)], capsys)

    assert code == 1 and report["applied"] is False
    assert "падеж" in report["results"][0]["reason"]
    assert not store.exists()

    code, report = _edit(["remember", "Яндексу → Яндексе", "--store", str(store), "--force"], capsys)
    assert code == 0 and "падеж" in report["results"][0]["forced_over"]
    assert entries_of(load_document(store))[0].aliases == ("Яндексу",)


def test_guard_catches_human_mistakes() -> None:
    empty: dict = {"entries": []}
    assert "регистр" in remember_problem(empty, "notion", "Notion")
    assert "перепутаны" in remember_problem(empty, "Claude", "клод")
    assert "короткий" in remember_problem(empty, "сео", "SEO")
    assert "звучит не как" in remember_problem(empty, "стакан", "GPT")
    assert remember_problem(empty, "клод", "Claude") is None
    assert remember_problem(empty, "страцессия", "стратсессия") is None
    assert remember_problem(empty, None, "Битрикс24") is None
    taken = {"entries": [{"canonical": "Астра", "aliases": ["Astra"], "auto_apply": True}]}
    assert "вариантом у «Астра»" in remember_problem(taken, "астра", "Astra")


def test_named_learned_entry_becomes_hand_and_keeps_only_safe_aliases() -> None:
    document = {
        "entries": [_learned("Threads", ["трэц", "тре"], held_aliases=["трэдс"], pruned_aliases=["трэдс"])],
        "pending": [{"aliases": ["тредс"], "windows": 1}],
        "pruned": [{"canonical": "Threads", "aliases": [], "reason": "x"}],
    }

    result, notes = remember(document, "трэдс", "Threads")

    (entry,) = result["entries"]
    assert entry["auto_apply"] is True
    assert "auto_apply_written" not in entry["learned"], "человек назвал канон — запись ручная"
    # «тре» короче порога: человек его не называл, в замену он не идёт.
    assert entry["aliases"] == ["трэц", "трэдс"]
    assert entry["learned"]["held_aliases"] == ["тре"]
    assert entry["learned"]["pruned_aliases"] == []
    assert result["pruned"] == [], "названный термин снимается из pruned"
    assert result["pending"] == [{"aliases": ["тредс"], "windows": 1}]
    assert document["entries"][0]["aliases"] == ["трэц", "тре"], "исходный документ не меняется"
    assert notes


def test_named_pending_place_stops_waiting() -> None:
    document = {"entries": [], "pending": [{"aliases": ["тэгэстат"], "windows": 3}]}
    result, _ = remember(document, "тэгэстат", "TGStat")
    assert result["pending"] == []
    assert result["entries"][0]["aliases"] == ["тэгэстат"]


def test_forget_moves_entry_to_pruned_and_learning_respects_it(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    update_from_run(store, _run("макросов", "Microsoft"), source="rec1")
    assert [entry.canonical for entry in entries_of(load_document(store))] == ["Microsoft"]

    code, report = _edit(["forget", "microsoft", "--store", str(store)], capsys)

    assert code == 0
    document = load_document(store)
    assert document["entries"] == []
    assert document["pruned"][0]["canonical"] == "Microsoft"
    assert document["pruned"][0]["reason"] == FORGOTTEN_REASON
    entries, learned = update_from_run(store, _run("макросов", "Microsoft"), source="rec2")
    assert entries == [] and learned.added == []


def test_add_lines_reads_stdin_and_reports_rejected(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    lines = "# термины встречи\n\nклод → Claude\nЯндексу -> Яндексе\nБитрикс24\n"

    code, report = _edit(["add-lines", "--store", str(store)], capsys, stdin=lines)

    assert code == 1
    assert [item["status"] for item in report["results"]] == ["ok", "rejected", "ok"]
    assert [entry.canonical for entry in entries_of(load_document(store))] == ["Claude", "Битрикс24"]


def test_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    code, report = _edit(["remember", "клод → Claude", "--store", str(store), "--dry-run"], capsys)
    assert code == 0 and report["applied"] is False
    assert not store.exists()


def test_unreadable_store_is_reported_not_overwritten(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    store.write_text("entries: [\n", encoding="utf-8")
    assert glossary_edit.main(["remember", "клод → Claude", "--store", str(store)]) == 2
    assert store.read_text(encoding="utf-8") == "entries: [\n"


def test_repeated_remember_changes_nothing(tmp_path: Path, capsys) -> None:
    store = tmp_path / "glossary.yaml"
    _edit(["remember", "клод → Claude", "--store", str(store)], capsys)
    before = store.read_text(encoding="utf-8")

    code, report = _edit(["remember", "клод → Claude", "--store", str(store)], capsys)

    assert code == 0 and report["applied"] is False
    assert report["results"][0]["notes"] == ["уже записано так"]
    assert store.read_text(encoding="utf-8") == before
