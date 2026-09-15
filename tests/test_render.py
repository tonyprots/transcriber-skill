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
    assert manifest["schema_version"] == 6
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
