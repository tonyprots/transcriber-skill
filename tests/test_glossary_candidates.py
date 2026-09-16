import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts"))

from glossary_candidates import collect_candidates, main, render_yaml  # noqa: E402


def _item(heard: str, verifier: str, start: float = 1.0) -> dict:
    return {
        "start": start,
        "end": start + 3,
        "readable_text": f"про {heard} сегодня",
        "comparison_text": f"про {verifier} сегодня",
        "differing_tokens": [f"{heard.lower()} / {verifier.lower()}"],
        "reason": "тест",
    }


def test_latin_on_verifier_side_becomes_candidate_with_original_case() -> None:
    payload = {"review_items": [_item("нотион", "Notion"), _item("который", "которые")]}
    candidates = collect_candidates([payload], known=set())
    assert [c["canonical"] for c in candidates] == ["Notion"]
    assert candidates[0]["heard"] == "нотион"
    assert "auto_apply: false" in render_yaml(candidates)


def test_repeated_cyrillic_disagreement_needs_two_windows_and_known_terms_are_skipped() -> None:
    payload = {"review_items": [_item("ретро", "ретра", 1.0), _item("ретро", "ретра", 9.0), _item("бэклог", "Backlog")]}
    assert collect_candidates([payload], known={"backlog"}) == []
    candidates = collect_candidates([payload], known={"backlog"}, min_count=2)
    assert [(c["heard"], c["count"]) for c in candidates] == [("ретро", 2)]


def test_uneven_spans_do_not_widen_the_alias() -> None:
    # «трец когда / threads» — два слова против одного: выравнивания нет.
    # Раньше алиасом становилось «трец когда», и замена работала только
    # рядом с этим соседом.
    payload = {"review_items": [_item("трец когда", "threads")]}
    assert collect_candidates([payload], known=set()) == []


def test_latin_on_the_primary_side_is_a_place_to_name() -> None:
    # Основная написала латиницей, проверяющая кириллицей: термин виден, но
    # правильного написания нет ни у одной — готовой записи не собрать.
    payload = {"review_items": [_item("treds", "трэдс")]}
    candidates = collect_candidates([payload], known=set())
    assert [c["heard"] for c in candidates] == ["treds"]
    assert candidates[0]["canonical"] == ""
    rendered = render_yaml(candidates)
    assert "entries:\n  []" in rendered
    assert "правильного написания нет ни у одной модели" in rendered
    # Незаполненная запись не должна попадать в entries: пустой canonical
    # роняет load_glossary.
    assert yaml.safe_load(rendered)["entries"] == []


def test_cli_writes_valid_glossary_yaml(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "segments.json").write_text(
        json.dumps({"review_items": [_item("фигма", "Figma")]}, ensure_ascii=False), encoding="utf-8"
    )
    target = tmp_path / "candidates.yaml"
    assert main([str(out_dir), "--output", str(target)]) == 0
    entries = yaml.safe_load(target.read_text(encoding="utf-8"))["entries"]
    assert entries[0]["canonical"] == "Figma"
    assert entries[0]["aliases"] == ["фигма"]
    assert entries[0]["auto_apply"] is False


def test_variants_of_one_term_collapse_into_a_single_entry() -> None:
    """«трэц», «treds», «трэдс» — одно слово, а не три кандидата.

    Раньше пользователь получал запись «Threads» и рядом два безымянных
    кандидата с «?» вместо канона — и должен был сам сообразить, что это тот
    же термин, по написаниям без единой общей буквы.
    """
    payload = {
        "review_items": [
            _item("трэц", "Threads", 11.4),
            _item("treds", "трэдс", 24.9),
            _item("treds", "трэдсе", 41.6),
        ]
    }
    candidates = collect_candidates([payload], known=set())
    assert [c["canonical"] for c in candidates] == ["Threads"]
    assert set(candidates[0]["aliases"]) == {"трэц", "treds", "трэдс", "трэдсе"}


def test_different_terms_are_not_glued_together() -> None:
    """Склейка идёт по звучанию, поэтому разные термины обязаны остаться врозь."""
    payload = {
        "review_items": [
            _item("инстаграме", "instagram", 5.0),
            _item("фейсбуке", "facebook", 6.0),
        ]
    }
    candidates = collect_candidates([payload], known=set())
    assert sorted(c["canonical"] for c in candidates) == ["facebook", "instagram"]


def test_merged_aliases_reach_the_rendered_yaml() -> None:
    payload = {
        "review_items": [_item("трэц", "Threads", 11.4), _item("treds", "трэдс", 24.9)]
    }
    document = yaml.safe_load(render_yaml(collect_candidates([payload], known=set())))
    (entry,) = document["entries"]
    assert entry["canonical"] == "Threads"
    assert "treds" in entry["aliases"] and "трэц" in entry["aliases"]
