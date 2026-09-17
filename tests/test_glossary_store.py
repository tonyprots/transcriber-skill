"""Словарь, который скилл ведёт сам.

Здесь проверяется не столько накопление, сколько его тормоза: включить замену
без человека можно только там, где ошибиться почти невозможно.
"""
from pathlib import Path

import yaml

from audio_transcription.glossary import GlossaryEntry, merge_glossaries
from audio_transcription.glossary_store import (
    load_document,
    save_document,
    store_path,
    update_from_run,
)
from audio_transcription.mining import undisputed_words
from audio_transcription.models import ReviewItem, Segment


def _item(heard: str, verifier: str, start: float = 1.0, comparison: str | None = None) -> dict:
    return {
        "start": start,
        "end": start + 2,
        "readable_text": heard,
        "comparison_text": comparison if comparison is not None else verifier,
        "similarity": 0.5,
        "differing_tokens": [f"{heard} / {verifier}"],
        "reason": "term",
    }


def _run(heard: str, verifier: str) -> list[dict]:
    return [_item(heard, verifier, 1.0), _item(heard, verifier, 30.0)]


def test_store_is_created_on_the_first_run(tmp_path: Path) -> None:
    store = tmp_path / "nested" / "glossary.yaml"
    entries, report = update_from_run(store, _run("трэц", "Threads"), source="rec1")

    assert store.is_file(), "словарь должен заводиться сам, без участия пользователя"
    assert [entry.canonical for entry in entries] == ["Threads"]
    assert report.added == ["Threads"]
    # Новая запись только подсвечивает: одной записи мало, чтобы править текст.
    assert entries[0].auto_apply is False


def test_same_recording_processed_again_is_not_new_evidence(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    for _ in range(5):
        update_from_run(store, _run("трэц", "Threads"), source="one-and-the-same")

    learned = load_document(store)["entries"][0]["learned"]
    assert learned["runs"] == 1, "перепрогон того же файла не должен копить доказательства"
    assert learned["windows"] == 2


def test_term_from_three_recordings_starts_correcting_the_text(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    for source in ("rec1", "rec2"):
        entries, _ = update_from_run(store, _run("трэц", "Threads"), source=source)
        assert entries[0].auto_apply is False

    entries, report = update_from_run(store, _run("трэц", "Threads"), source="rec3")
    assert entries[0].auto_apply is True
    assert report.promoted == ["Threads"]


def test_ordinary_word_never_starts_being_replaced(tmp_path: Path) -> None:
    """«трек» в «фитнес-трек» не должен превращаться в термин."""
    store = tmp_path / "glossary.yaml"
    for source in ("rec1", "rec2", "rec3", "rec4"):
        # В первой записи слово встретилось там, где моделям спорить было не о чем.
        blocked = {"трек"} if source == "rec1" else set()
        entries, _ = update_from_run(store, _run("трек", "Track"), blocked=blocked, source=source)

    assert entries[0].auto_apply is False
    learned = load_document(store)["entries"][0]["learned"]
    assert learned["seen_as_word"] is True, "признак обычного слова должен быть вечным"


def test_pair_that_does_not_sound_alike_is_not_promoted(tmp_path: Path) -> None:
    """«значит» против «GPT» — совпадение в очереди, а не ослышка."""
    store = tmp_path / "glossary.yaml"
    for source in ("rec1", "rec2", "rec3", "rec4"):
        entries, _ = update_from_run(store, _run("значит", "GPT"), source=source)

    assert entries[0].auto_apply is False
    assert "звучит не как" in load_document(store)["entries"][0]["learned"]["held_back"]


def test_short_alias_is_not_promoted(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    for source in ("rec1", "rec2", "rec3"):
        entries, _ = update_from_run(store, _run("смб", "SMB"), source=source)

    assert entries[0].auto_apply is False


def test_manual_switch_off_survives_further_runs(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    for source in ("rec1", "rec2", "rec3"):
        update_from_run(store, _run("трэц", "Threads"), source=source)
    document = load_document(store)
    assert document["entries"][0]["auto_apply"] is True

    document["entries"][0]["auto_apply"] = False
    save_document(store, document)
    for source in ("rec4", "rec5", "rec6"):
        entries, _ = update_from_run(store, _run("трэц", "Threads"), source=source)

    assert entries[0].auto_apply is False, "ручная правка сильнее автоматики"


def test_term_nobody_spelled_right_waits_for_a_human(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    # Латиница у основной модели: правильного написания нет ни у одной стороны.
    entries, report = update_from_run(store, [_item("treds", "трэдс")], source="rec1")

    document = load_document(store)
    assert document["entries"] == []
    assert document["pending"], "такие места должны копиться, а не пропадать"
    assert report.pending
    assert entries == [], "выдумывать написание скилл не должен"


def test_curated_glossary_wins_over_the_learned_one() -> None:
    curated = [GlossaryEntry("Threads", ("трэц",), auto_apply=True)]
    learned = [
        GlossaryEntry("Threads", ("трэц", "трэдс"), auto_apply=True),
        GlossaryEntry("Obsidian", ("обсидиан",), auto_apply=True),
    ]
    merged = merge_glossaries(curated, learned)

    assert [entry.canonical for entry in merged] == ["Threads", "Obsidian"]
    assert merged[0].aliases == ("трэц",), "запись пользователя не должна подменяться"


def test_undisputed_words_are_counted_word_by_word() -> None:
    segments = [Segment(0, 20, "мы обсуждали трек и фитнес-трекер весь день")]
    items = [
        ReviewItem(
            start=0,
            end=20,
            readable_text="трек",
            comparison_text="Track",
            similarity=0.5,
            differing_tokens=("трек / Track",),
            reason="term",
        )
    ]
    words = undisputed_words(segments, items)

    assert "день" in words, "спорное окно не должно вычёркивать всё предложение"
    assert "трек" not in words, "спорное слово обычным не считается"


def test_store_path_follows_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_STORE", str(tmp_path / "custom.yaml"))
    assert store_path() == tmp_path / "custom.yaml"
    assert store_path(tmp_path / "explicit.yaml") == tmp_path / "explicit.yaml"


def test_saved_document_stays_readable_yaml(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    update_from_run(store, _run("трэц", "Threads"), source="rec1")

    text = store.read_text(encoding="utf-8")
    assert text.startswith("# Словарь терминов"), "файл читает человек, а не только скилл"
    assert yaml.safe_load(text)["entries"][0]["canonical"] == "Threads"
