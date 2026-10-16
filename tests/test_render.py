import json
from pathlib import Path

from audio_transcription.models import Diarization, Hypothesis, MediaInfo, Segment, SpeakerTurn
from audio_transcription.render import format_timestamp, write_bundle


def test_timestamp_formats_srt() -> None:
    assert format_timestamp(65.432, srt=True) == "00:01:05,432"


def test_write_bundle_creates_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    media = MediaInfo(source, 1.0, "pcm_s16le", 16000, 1)
    segments = [Segment(0, 1, "Привет, мир!")]
    hypothesis = Hypothesis("model", "ru", 0.2, segments)
    output = write_bundle(
        tmp_path / "result",
        media=media,
        hypotheses=[hypothesis],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="fast",
        offline=True,
    )
    expected = {
        "manifest.json",
        "raw.json",
        "verbatim.md",
        "readable.md",
        "subtitles.srt",
        "subtitles.vtt",
        "segments.json",
        "glossary-audit.json",
        "review-needed.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 8
    assert manifest["model_revisions"] == {}
    assert "Привет, мир!" in (output / "readable.md").read_text()


def test_manifest_keeps_model_revisions(tmp_path: Path) -> None:
    """Ревизия весов — часть аудит-следа: без неё «тот же прогон» непроверяем."""
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    media = MediaInfo(source, 1.0, "pcm_s16le", 16000, 1)
    segments = [Segment(0, 1, "Текст")]
    output = write_bundle(
        tmp_path / "result",
        media=media,
        hypotheses=[Hypothesis("gigaam-v3-e2e-rnnt", "ru", 0.1, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="max",
        offline=True,
        model_revisions={"gigaam-v3-e2e-rnnt": "322c3b294926", "silero": None},
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model_revisions"]["gigaam-v3-e2e-rnnt"] == "322c3b294926"
    assert manifest["model_revisions"]["silero"] is None


def test_write_bundle_accepts_precreated_empty_output(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    media = MediaInfo(source, 1.0, "pcm_s16le", 16000, 1)
    segments = [Segment(0, 1, "Текст")]
    output = tmp_path / "reserved-output"
    output.mkdir()
    write_bundle(
        output,
        media=media,
        hypotheses=[Hypothesis("model", "ru", 0.1, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="fast",
        offline=True,
    )
    assert (output / "manifest.json").is_file()
    # Пустой заранее созданный каталог не архивируется: архив нужен только
    # прежнему результату, а не мусорной папке рядом.
    assert not list(tmp_path.glob("reserved-output.archive-*"))


def test_write_bundle_renders_speakers_and_diarization_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    media = MediaInfo(source, 2.0, "pcm_s16le", 16000, 1)
    segments = [Segment(0, 2, "Привет", speakers=("speaker_1",))]
    diarization = Diarization(
        "sortformer",
        0.1,
        [SpeakerTurn(0, 2, ("speaker_1",))],
    )
    output = write_bundle(
        tmp_path / "speaker-result",
        media=media,
        hypotheses=[Hypothesis("model", "ru", 0.2, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="fast",
        offline=True,
        diarization=diarization,
    )
    assert "Спикер 1: Привет" in (output / "readable.md").read_text()
    assert (output / "speakers.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["speaker_count"] == 1


def _bundle(tmp_path: Path, review_items, segments=None) -> Path:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    segments = segments or [Segment(0, 1, "Текст")]
    return write_bundle(
        tmp_path / "result",
        media=MediaInfo(source, 60.0, "pcm_s16le", 16000, 1),
        hypotheses=[Hypothesis("model", "ru", 0.1, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=review_items,
        corrections=[],
        suggestions=[],
        mode="max",
        offline=True,
    )


def test_unavailable_verifier_is_one_status_line_not_a_queue(tmp_path: Path) -> None:
    """На диктовке Handy 2026-10-10 все 39 мест «Слушать» были «verifier недоступен».

    Окно без сверки — статус прогона, а не место, которое надо слушать.
    """
    from audio_transcription.cli import verifier_failure_items
    from audio_transcription.models import AudioChunk

    chunks = [AudioChunk(index, Path(f"{index}.wav"), index * 2.0, index * 2.0 + 2.0) for index in range(39)]
    primary = Hypothesis(
        "gigaam", "ru", 0.1, [],
        {"chunks": [{"sequence": chunk.sequence, "text": f"фраза {chunk.sequence}"} for chunk in chunks]},
    )
    items = verifier_failure_items(chunks, primary, "Изолированный Whisper завершился с ошибкой: OOM\nдлинный хвост")
    assert {item.kind for item in items} == {"unverified"}

    output = _bundle(tmp_path, items)
    review = (output / "review-needed.md").read_text(encoding="utf-8")
    status = [line for line in review.splitlines() if line.startswith("Не сверено")]
    assert len(status) == 1
    assert "OOM" in status[0] and "39" in status[0] and "длинный хвост" not in review
    assert "## Слушать" not in review
    assert "не найдено" not in review
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_counts"]["unverified"] == 39
    assert manifest["review_counts"]["listen"] == 0
    # review_count не меняет смысла: в него входит всё, кроме взятого у проверяющей.
    assert manifest["review_count"] == 39


def test_primary_gap_has_its_own_section_above_a_long_queue(tmp_path: Path) -> None:
    """Провал основной модели стоял на 27–172-м месте «Слушать», ниже 25-й строки."""
    from audio_transcription.cli import primary_retry_items
    from audio_transcription.models import ReviewItem

    heavy = [
        ReviewItem(float(i), float(i) + 1, f"контекст {i}", "", 0.2, (f"а{i} б{i} / в{i} г{i}",), "r", 5)
        for i in range(30)
    ]
    primary = Hypothesis(
        "gigaam", "ru", 0.1, [],
        {"chunks": [
            {"sequence": 0, "start": 100.0, "end": 104.0, "text": "начало фразы", "status": "partial"},
            {"sequence": 1, "start": 110.0, "end": 112.0, "text": "", "status": "failed"},
        ]},
    )
    gaps = primary_retry_items(primary)
    assert [item.kind for item in gaps] == ["primary_gap", "primary_gap"]

    output = _bundle(tmp_path, heavy + gaps)
    review = (output / "review-needed.md").read_text(encoding="utf-8")
    assert "## Окна без текста основной модели — 2" in review
    gap_section = review.split("## Окна без текста основной модели")[1].split("## Слушать")[0]
    assert "[00:01:40.000–00:01:44.000]" in gap_section and "начало фразы" in gap_section
    assert "[00:01:50.000–00:01:52.000] не распознано" in gap_section
    assert "## Слушать — 30, здесь первые 25" in review
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_counts"] == {
        "listen": 30, "spelling": 0, "dropped": 0, "minor": 0,
        "taken": 0, "unverified": 0, "primary_gap": 2,
    }
    assert manifest["review_count"] == 32


def test_rest_of_a_long_queue_is_named_by_minutes(tmp_path: Path) -> None:
    from audio_transcription.models import ReviewItem

    items = [
        ReviewItem(float(i), float(i) + 1, f"контекст {i}", "", 0.2, (f"а{i} б{i} / в{i}",), "r", 3)
        for i in range(25)
    ] + [
        ReviewItem(start, start + 1, "хвост", "", 0.5, (f"х{start:g} / ц",), "r", 1)
        for start in (130.0, 245.0, 3700.0, 3720.0)
    ]
    review = (_bundle(tmp_path, items) / "review-needed.md").read_text(encoding="utf-8")
    assert "Ещё 4 — в `segments.json` → `review_items`. Они на минутах: 2, 4, 61–62." in review
