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
