"""Словарь, который скилл ведёт сам.

Здесь проверяется не столько накопление, сколько его тормоза: включить замену
без человека можно только там, где ошибиться почти невозможно.
"""
from pathlib import Path

import pytest
import yaml

from audio_transcription.glossary import GlossaryEntry, merge_glossaries
from audio_transcription.glossary_store import (
    load_document,
    record_fingerprint,
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


def _promote_threads(store: Path) -> None:
    for source in ("rec1", "rec2", "rec3"):
        entries, _ = update_from_run(store, _run("трэц", "Threads"), source=source)
    assert entries[0].auto_apply is True


def test_switched_on_entry_does_not_pick_up_an_unsafe_alias(tmp_path: Path) -> None:
    """После повышения запись копит варианты дальше; «в» в них попасть не должно."""
    store = tmp_path / "glossary.yaml"
    _promote_threads(store)

    entries, _ = update_from_run(store, _run("в", "Threads"), source="rec4")

    assert entries[0].auto_apply is True
    assert entries[0].aliases == ("трэц",)
    assert load_document(store)["entries"][0]["learned"]["held_aliases"] == ["в"]


def test_switched_on_entry_is_switched_off_when_its_alias_turns_out_ordinary(
    tmp_path: Path,
) -> None:
    store = tmp_path / "glossary.yaml"
    _promote_threads(store)

    entries, report = update_from_run(store, [], blocked={"трэц"}, source="rec4")

    assert entries[0].auto_apply is False
    assert report.demoted == ["Threads"]
    assert "сняты с автозамены: Threads" in report.summary()


def test_phrase_punctuation_does_not_become_part_of_the_term(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    entries, _ = update_from_run(
        store, [_item("вкс, <unk>", "VKS, Байкал", comparison="VKS, Байкал")], source="rec1"
    )

    assert [(entry.canonical, list(entry.aliases)) for entry in entries] == [("VKS", ["вкс"])]


def test_clips_of_one_video_are_one_recording() -> None:
    """Три клипа одного ролика — одна запись, а не три доказательства."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert record_fingerprint("a" * 64, url) == record_fingerprint("b" * 64, url)
    assert record_fingerprint("a" * 64) != record_fingerprint("b" * 64)


def test_spelling_named_by_a_human_is_not_relearned(tmp_path: Path) -> None:
    """«Астра» пишется по-русски; модель, пишущая «Astra», не растит запись заново."""
    store = tmp_path / "glossary.yaml"
    save_document(
        store,
        {
            "version": 1,
            "entries": [
                {
                    "canonical": "Астра",
                    "aliases": ["Astra"],
                    "auto_apply": True,
                    "learned": {"auto_apply_written": False, "sources": ["rec1"]},
                }
            ],
            "pending": [],
        },
    )

    entries, _ = update_from_run(store, _run("астра", "Astra"), source="rec2")
    # Латиница у основной модели: раньше это место ушло бы ждать человека.
    update_from_run(store, _run("Astra", "астро"), source="rec3")

    document = load_document(store)
    assert [entry.canonical for entry in entries] == ["Астра"]
    assert document["entries"][0]["aliases"] == ["Astra"], "ручную запись автоматика не дополняет"
    assert document["pending"] == []


def _learned(canonical: str, aliases: list[str], **learned) -> dict:
    return {
        "canonical": canonical,
        "aliases": aliases,
        "auto_apply": False,
        "learned": {"auto_apply_written": False, "windows": 1, **learned},
    }


def test_prune_removes_coincidences_and_keeps_terms() -> None:
    from audio_transcription.glossary_store import prune

    document = {
        "entries": [
            _learned("Zon", ["там"], seen_as_word=True),
            _learned("D", ["и"]),
            # Кириллическая запись термина — не мусор, хоть и «обычное слово».
            _learned("Anthropic", ["антропик"], seen_as_word=True),
            _learned("SEO", ["там", "сео"], seen_as_word=True),
            _learned("Q4", ["ку-4"]),
        ],
        "pending": [],
    }
    cleaned, report = prune(document)
    assert [item["canonical"] for item in cleaned["entries"]] == ["Anthropic", "SEO", "Q4"]
    seo = cleaned["entries"][1]
    assert seo["aliases"] == ["сео"] and seo["learned"]["pruned_aliases"] == ["там"]
    assert {item["canonical"] for item in cleaned["pruned"]} == {"Zon", "D"}
    assert all(item["reason"] for item in cleaned["pruned"])


def test_prune_never_touches_hand_and_switched_on_entries() -> None:
    from audio_transcription.glossary_store import prune

    hand = {"canonical": "Битрикс24", "aliases": ["битрикс"], "auto_apply": True}
    switched_on = {
        "canonical": "Threads",
        "aliases": ["трэц"],
        "auto_apply": True,
        "learned": {"auto_apply_written": True, "seen_as_word": True},
    }
    duplicate = _learned("BITREX", ["битрикс"])
    cleaned, report = prune({"entries": [hand, switched_on, duplicate], "pending": []})
    assert cleaned["entries"] == [hand, switched_on]
    assert "поглощена ручной записью «Битрикс24»" in report.removed[0]["reason"]


def test_shared_alias_stays_with_the_stronger_entry() -> None:
    from audio_transcription.glossary_store import prune

    weak = _learned("OPNN", ["опен"], windows=1)
    strong = _learned("Open", ["опен", "оупен"], windows=5)
    cleaned, report = prune({"entries": [weak, strong], "pending": []})
    assert [item["canonical"] for item in cleaned["entries"]] == ["Open"]
    assert report.trimmed == [{"canonical": "OPNN", "aliases": ["опен"]}]


def test_pruned_term_is_not_learned_again(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    store.write_text(
        yaml.safe_dump(
            {"version": 1, "entries": [], "pending": [], "pruned": [{"canonical": "Zon", "aliases": ["там"], "reason": "x"}]},
            allow_unicode=True,
        )
    )
    entries, report = update_from_run(store, _run("зон", "Zon"), source="rec1")
    assert not any(entry.canonical == "Zon" for entry in entries)


def test_verifier_spelling_of_a_hand_term_reinforces_it(tmp_path: Path) -> None:
    store = tmp_path / "glossary.yaml"
    store.write_text(
        yaml.safe_dump(
            {"version": 1, "entries": [{"canonical": "Битрикс24", "aliases": ["битрикс"], "auto_apply": True}]},
            allow_unicode=True,
        )
    )
    entries, report = update_from_run(store, _run("битрикс", "BITREX"), source="rec1")
    assert [entry.canonical for entry in entries] == ["Битрикс24"]
    # Не завели и тут же убрали, а сразу засчитали ручной записи.
    assert report.added == [] and report.reinforced == ["Битрикс24"]


def test_profile_keeps_dictionaries_apart(monkeypatch, tmp_path) -> None:
    from audio_transcription import glossary_store

    monkeypatch.setattr(glossary_store, "PROFILES_DIR", tmp_path / "glossaries")
    monkeypatch.delenv("TRANSCRIBER_GLOSSARY_STORE", raising=False)
    monkeypatch.delenv("TRANSCRIBER_GLOSSARY_PROFILE", raising=False)
    assert glossary_store.store_path(profile="work") == tmp_path / "glossaries" / "work.yaml"
    # Явный путь сильнее профиля, профиль сильнее переменных окружения.
    assert glossary_store.store_path(tmp_path / "x.yaml", "work") == tmp_path / "x.yaml"
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_STORE", str(tmp_path / "env.yaml"))
    assert glossary_store.store_path(profile="home") == tmp_path / "glossaries" / "home.yaml"
    monkeypatch.delenv("TRANSCRIBER_GLOSSARY_STORE")
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_PROFILE", "home")
    assert glossary_store.store_path() == tmp_path / "glossaries" / "home.yaml"
    with pytest.raises(ValueError):
        glossary_store.profile_path("../etc")


def test_owner_can_switch_off_automatic_promotion(tmp_path: Path) -> None:
    """Кто хочет решать сам, ставит auto_promote: false — доказательства копятся, текст не правится."""
    store = tmp_path / "glossary.yaml"
    save_document(store, {"version": 1, "auto_promote": False, "entries": []})
    for source in ("rec1", "rec2", "rec3", "rec4"):
        entries, report = update_from_run(store, _run("трэц", "Threads"), source=source)
    assert entries[0].auto_apply is False
    assert report.promoted == []
    document = load_document(store)
    assert document["auto_promote"] is False, "настройка не должна теряться при перезаписи"
    assert "человек" in document["entries"][0]["learned"]["held_back"]
