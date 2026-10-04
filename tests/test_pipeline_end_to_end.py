"""Сквозной прогон: настоящий ffmpeg, настоящий VAD, настоящая сборка каталога.

Остальные тесты проверяют детали по отдельности и проходят за полсекунды, но
между ними остаётся зазор: склейка окон, распаковка по таймкодам, сверка,
словарь и запись артефактов ни разу не встречаются в одном прогоне. Регрессия
в стыке видна только здесь.

Быстрый тест подменяет ASR фальшивыми бэкендами — весов не нужно, а весь путь
от файла до `manifest.json` проходится целиком. Помеченный `slow` гоняет тот же
путь на настоящих моделях, если они уже скачаны.

Речь берём у системного синтезатора macOS: он даёт живой сигнал, который проходит
VAD, и не тащит в публичный репозиторий чужой голос.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.backends import language_hypothesis
from audio_transcription.catalog import GIGAAM_RUSSIAN, hf_cache_dir
from audio_transcription.models import Hypothesis, Segment

PRIMARY_TEXT = "сегодня смотрим отчёт в PostgreSQL и считаем бюджет"
VERIFIER_TEXT = "сегодня смотрим отчёт в постгрес и считаем бюджет"


def _say_available() -> bool:
    return bool(shutil.which("say") and shutil.which("ffmpeg") and shutil.which("ffprobe"))


requires_speech = pytest.mark.skipif(
    not _say_available(), reason="нужны say, ffmpeg и ffprobe (macOS)"
)


@pytest.fixture
def spoken_audio(tmp_path: Path) -> Path:
    source = tmp_path / "речь.aiff"
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(source), "Сегодня смотрим отчёт и считаем бюджет"],
        check=True,
        capture_output=True,
    )
    return source


def fake_backend(backend: str, chunks, language: str, work_dir: Path, **kwargs) -> Hypothesis:
    """Гипотеза по тем же окнам, что пришли: расхождение ровно одно, на термине.

    `metadata["chunks"]` обязателен: по `sequence` результат проверяющей
    раскладывается обратно на исходные окна, и без него распаковка вернёт пусто.
    """
    if backend.endswith("-lid"):
        return language_hypothesis(backend, [{"ru": 0.97, "en": 0.03}] * len(chunks), 0.1)
    text = PRIMARY_TEXT if backend == "gigaam" else VERIFIER_TEXT
    chunks = list(chunks)
    segments = [Segment(chunk.start, chunk.end, text) for chunk in chunks]
    metadata = {
        "chunks": [
            {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end, "text": text}
            for chunk in chunks
        ]
    }
    return Hypothesis(f"{backend}-fake", language, len(chunks), segments, metadata=metadata)


@requires_speech
def test_pipeline_writes_full_contract(tmp_path: Path, spoken_audio: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    output = tmp_path / "result"

    result = cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "max",
                "--language",
                "ru",
                "--no-cache",
                "--quiet",
                # Склейку проверяет test_packing; здесь нужен прямой путь
                # «окно к окну», чтобы фальшивый бэкенд отвечал за то же окно.
                "--verifier-window-seconds",
                "0",
            ]
        )
    )

    assert result == output
    produced = {path.name for path in output.iterdir()}
    assert produced == {
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

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 8
    assert manifest["mode"] == "max"
    assert manifest["glossary"]["learning"] is True
    assert manifest["glossary"]["store"].endswith("glossary.yaml")
    assert manifest["language_route"]["verifier_used"] is not None
    assert manifest["timings_seconds"]["total"] > 0
    # VAD настоящий: окна пришли из речи, а не из пустого списка.
    assert manifest["source"]["duration_seconds"] > 0

    # Расхождение моделей дошло до очереди вместе с обеими формулировками.
    review = (output / "review-needed.md").read_text(encoding="utf-8")
    assert "PostgreSQL" in review and "постгрес" in review
    assert manifest["review_count"] >= 1

    # Текст основной модели не подменён вариантом проверяющей.
    assert "PostgreSQL" in (output / "verbatim.md").read_text(encoding="utf-8")
    assert "постгрес" not in (output / "readable.md").read_text(encoding="utf-8")

    srt = (output / "subtitles.srt").read_text(encoding="utf-8")
    assert "-->" in srt and "PostgreSQL" in srt


@requires_speech
def test_pipeline_applies_glossary_and_keeps_raw_intact(
    tmp_path: Path, spoken_audio: Path, monkeypatch
) -> None:
    """Словарь правит читаемый текст, но не аудит-след сырых гипотез."""
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text(
        "version: 1\nlanguage: ru\nentries:\n"
        "  - canonical: PostgreSQL\n    aliases: [постгрес]\n    auto_apply: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--glossary",
                str(glossary),
                "--no-cache",
                "--quiet",
            ]
        )
    )

    raw = json.loads((output / "raw.json").read_text(encoding="utf-8"))
    assert any(PRIMARY_TEXT in segment["text"] for h in raw["hypotheses"] for segment in h["segments"])
    audit = json.loads((output / "glossary-audit.json").read_text(encoding="utf-8"))
    assert isinstance(audit["corrections"], list)


@pytest.mark.slow
@requires_speech
def test_pipeline_on_real_models(tmp_path: Path, spoken_audio: Path) -> None:
    """То же самое, но настоящими моделями: запускать вручную `pytest -m slow`."""
    if not (hf_cache_dir() / (GIGAAM_RUSSIAN.cache_dir or "")).is_dir():
        pytest.skip("модели русского маршрута не скачаны")
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--no-cache",
                "--quiet",
            ]
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    # Ревизии весов проставлены настоящие — именно они делают прогон воспроизводимым.
    assert manifest["model_revisions"][GIGAAM_RUSSIAN.model] == GIGAAM_RUSSIAN.revision
    # И это те веса, которые воркер загрузил на самом деле, а не запись каталога.
    raw = json.loads((output / "raw.json").read_text(encoding="utf-8"))
    primary = next(h for h in raw["hypotheses"] if h["model"] == GIGAAM_RUSSIAN.model)
    assert primary["metadata"]["weights"] == {"revision": GIGAAM_RUSSIAN.revision, "pinned": True}
    text = (output / "verbatim.md").read_text(encoding="utf-8")
    assert "бюджет" in text.lower()


@requires_speech
def test_speech_window_flag_reaches_vad(tmp_path: Path, spoken_audio: Path, monkeypatch) -> None:
    """Флаг длины окна должен доезжать до нарезки, а не жить в справке.

    Замер 2026-09-20 отверг 28 секунд, но настройка осталась ради следующей
    проверки — а настройка, которая никуда не приходит, хуже отсутствующей.
    """
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    original = cli.split_speech_windows
    seen: dict[str, float] = {}

    def recording(*args, **kwargs):
        seen["max_seconds"] = kwargs["max_seconds"]
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "split_speech_windows", recording)
    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(tmp_path / "result"),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--speech-window-seconds",
                "28",
                "--no-cache",
                "--no-glossary",
                "--quiet",
            ]
        )
    )
    assert seen["max_seconds"] == 28.0


@pytest.fixture
def audio_after_pause(tmp_path: Path, spoken_audio: Path) -> Path:
    """Шесть секунд тишины, потом речь: фрагмент с 5-й секунды ловит только её."""
    target = tmp_path / "пауза-и-речь.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", "6", "-i", "anullsrc=r=16000:cl=mono",
            "-i", str(spoken_audio),
            "-filter_complex", "[1:a]aresample=16000,aformat=channel_layouts=mono[s];[0:a][s]concat=n=2:v=0:a=1",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@requires_speech
def test_section_keeps_source_timecodes(tmp_path: Path, audio_after_pause: Path, monkeypatch) -> None:
    """Фрагмент расшифрован отдельно, а время в нём — по исходнику.

    Иначе цитату с 00:10:50 в субтитрах пришлось бы искать в расшифровке
    на 00:00:00 и пересчитывать руками.
    """
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    output = tmp_path / "result"
    cli.run(
        cli.build_parser().parse_args(
            [str(audio_after_pause), "--section", "0:05-1:00", "--output", str(output),
             "--mode", "fast", "--no-cache", "--no-glossary", "--quiet"]
        )
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    section = manifest["source"]["section"]
    assert section["start"] == 5.0
    # Конец за пределами записи обрезан по её длине, а не оставлен минутой.
    assert section["end"] < 60
    assert manifest["source"]["duration_seconds"] == pytest.approx(section["end"] - 5.0, abs=0.1)
    segments = json.loads((output / "segments.json").read_text(encoding="utf-8"))
    assert segments["readable"] and all(item["start"] >= 5.0 for item in segments["readable"])
    readable = (output / "readable.md").read_text(encoding="utf-8")
    assert "фрагмент 00:00:05–" in readable
    assert "[00:00:0" in readable and "[00:00:00" not in readable


@requires_speech
def test_batch_downloads_once_and_writes_each_section(
    tmp_path: Path, audio_after_pause: Path, monkeypatch, capsys
) -> None:
    """Пакет: два фрагмента одной записи — два каталога и один JSON с путями."""
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    code = cli.main(
        [str(audio_after_pause), str(tmp_path / "нет-такого.wav"),
         "--section", "0-4", "--section", "5-60", "--output", str(tmp_path / "batch"),
         "--mode", "fast", "--no-cache", "--no-glossary", "--quiet"]
    )
    payload = json.loads(capsys.readouterr().out)
    results = payload["results"]
    assert code == 2  # одна запись не нашлась — пакет сообщает об этом кодом
    done = [item for item in results if "error" not in item]
    failed = [item for item in results if "error" in item]
    assert len(done) == 2 and len(failed) == 2
    assert {Path(item["output"]).name for item in done} == {
        "пауза-и-речь-000000-000004",
        "пауза-и-речь-000005-000100",
    }
    for item in done:
        assert Path(item["readable"]).is_file()
        assert item["review_count"] is not None


@requires_speech
def test_two_sections_of_one_file_do_not_share_cache(
    tmp_path: Path, audio_after_pause: Path, monkeypatch
) -> None:
    """SHA-256 у кусков один — ключ кэша обязан различать границы."""
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    keys: list[str] = []
    original = cli.cache_key

    def recording(kind, payload):
        keys.append(payload["source_sha256"]) if kind == "primary" else None
        return original(kind, payload)

    monkeypatch.setattr(cli, "cache_key", recording)
    for number, section in enumerate(("0-4", "5-60")):
        cli.run(
            cli.build_parser().parse_args(
                [str(audio_after_pause), "--section", section,
                 "--output", str(tmp_path / f"r{number}"), "--mode", "fast",
                 "--cache-dir", str(tmp_path / "cache"), "--no-glossary", "--quiet"]
            )
        )
    assert len(keys) == 2 and keys[0] != keys[1]


@requires_speech
def test_crashed_primary_warns_and_is_recomputed_next_run(
    tmp_path: Path, spoken_audio: Path, monkeypatch, capsys
) -> None:
    """Сценарий 2026-09-23: выключение системы уронило все окна основной модели.

    Тогда провал ушёл в кэш, а повтор выдал пустую расшифровку без единого
    предупреждения. Теперь первый прогон кричит, второй считает заново.
    """
    def crashing_backend(backend, chunks, language, work_dir, **kwargs):
        if backend != "gigaam" or kwargs["config"]["model_name"] != GIGAAM_RUSSIAN.model:
            return fake_backend(backend, chunks, language, work_dir, **kwargs)
        chunks = list(chunks)
        metadata = {"chunks": [
            {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end,
             "text": "", "status": "failed", "retry_attempts": 1, "retry_depth": 0,
             "error": "dlopen(libonnxruntime.dylib): system is shutting down"}
            for chunk in chunks
        ]}
        return Hypothesis(f"{backend}-fake", language, 0.1, [], metadata=metadata)

    def argv(output: Path) -> list[str]:
        return [str(spoken_audio), "--output", str(output), "--mode", "fast",
                "--language", "ru", "--cache-dir", str(tmp_path / "cache"),
                "--no-glossary", "--quiet"]

    monkeypatch.setattr(cli, "run_isolated_backend", crashing_backend)
    assert cli.main(argv(tmp_path / "broken")) == 0
    captured = capsys.readouterr()
    stdout = json.loads(captured.out)
    assert any("не распознала" in item and "system is shutting down" in item
               for item in stdout["warnings"])
    assert "не распознала" in captured.err
    manifest = json.loads((tmp_path / "broken" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["warnings"] == stdout["warnings"]

    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    assert cli.main(argv(tmp_path / "healed")) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["warnings"] == []
    assert "не распознала" not in captured.err
    assert "PostgreSQL" in (tmp_path / "healed" / "readable.md").read_text(encoding="utf-8")


@requires_speech
def test_run_prunes_old_cache_and_tags_new_entries(tmp_path: Path, spoken_audio: Path, monkeypatch) -> None:
    """Прогон убирает давно не нужное и помечает своё исходником — иначе `forget` слеп."""
    import os
    import time

    from audio_transcription import cache

    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    cache_dir = tmp_path / "cache"
    stale = cache_dir / "hypotheses" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    long_ago = time.time() - 200 * 86400
    os.utime(stale, (long_ago, long_ago))

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(tmp_path / "result"),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--no-learn",
                "--cache-dir",
                str(cache_dir),
                "--quiet",
            ]
        )
    )

    assert not stale.exists()
    manifest = json.loads((tmp_path / "result" / "manifest.json").read_text(encoding="utf-8"))
    swept = cache.forget(cache_dir, manifest["source"]["sha256"])
    assert swept.files >= 2  # основная и проверяющая
    assert cache.cache_stats(cache_dir)["files"] == 0


ENGLISH_PRIMARY = "we launched Chachapiti with Astra today"
ENGLISH_VERIFIER = "we launched GPT with Astra today"


def fake_english_backend(backend: str, chunks, language: str, work_dir: Path, **kwargs) -> Hypothesis:
    """Английский маршрут: Whisper основная, Parakeet через тот же onnx-воркер."""
    text = ENGLISH_PRIMARY if "whisper" in backend else ENGLISH_VERIFIER
    chunks = list(chunks)
    segments = [Segment(chunk.start, chunk.end, text) for chunk in chunks]
    metadata = {
        "chunks": [
            {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end, "text": text}
            for chunk in chunks
        ]
    }
    return Hypothesis(f"{backend}-fake", language, len(chunks), segments, metadata=metadata)


@requires_speech
def test_russian_store_does_not_touch_english_route(
    tmp_path: Path, spoken_audio: Path, monkeypatch
) -> None:
    """Подкаст Lenny 2026-10-04: «Astra» → «Астра», «Chachapiti» → «GPT».

    Авто-словарь копится на русской речи: канон в нём — русское написание,
    алиасы — ослышки GigaAM. На английской записи он не читается вовсе, у неё
    свой файл рядом.
    """
    monkeypatch.setattr(cli, "run_isolated_backend", fake_english_backend)
    russian_store = tmp_path / "glossary.yaml"
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_STORE", str(russian_store))
    russian_store.write_text(
        "version: 1\nentries:\n"
        "  - canonical: Астра\n    aliases: [Astra, астра]\n    auto_apply: true\n"
        "  - canonical: GPT\n    aliases: [джипити, gpt]\n    auto_apply: true\n",
        encoding="utf-8",
    )
    before = russian_store.read_text(encoding="utf-8")
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "max",
                "--language",
                "en",
                "--no-cache",
                "--quiet",
                "--verifier-window-seconds",
                "0",
            ]
        )
    )

    readable = (output / "readable.md").read_text(encoding="utf-8")
    assert "Астра" not in readable and "Astra" in readable
    assert "Chachapiti" in readable
    review = (output / "review-needed.md").read_text(encoding="utf-8")
    assert "Взято у проверяющей" not in review
    audit = json.loads((output / "glossary-audit.json").read_text(encoding="utf-8"))
    assert audit["corrections"] == []
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["glossary"]["store"] == str(tmp_path / "glossary.en.yaml")
    assert manifest["glossary"]["learning"] is True
    assert russian_store.read_text(encoding="utf-8") == before


@requires_speech
def test_early_text_is_replaced_by_final_text(tmp_path: Path, spoken_audio: Path, monkeypatch) -> None:
    """Ранний текст уходит в буфер, а файл к концу прогона отдаёт итог readable.md.

    Агент, пришедший к готовой записи Handy, читает этот файл: без перезаписи он
    получал бы версию одной основной модели, без словаря и правок проверяющей.
    """
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    writes: list[str] = []
    original = cli.write_plain_text

    def recording(path: Path, segments) -> None:
        original(path, segments)
        writes.append(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(cli, "write_plain_text", recording)
    output = tmp_path / "result"
    early = tmp_path / "result.txt"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output", str(output),
                "--mode", "fast",
                "--language", "ru",
                "--early-text", str(early),
                "--no-cache",
                "--quiet",
            ]
        )
    )

    assert len(writes) == 2
    readable = (output / "readable.md").read_text(encoding="utf-8")
    final = early.read_text(encoding="utf-8").strip()
    assert final and final == writes[-1].strip()
    for paragraph in final.split("\n\n"):
        assert paragraph in readable


def fake_alias_backend(backend: str, chunks, language: str, work_dir: Path, **kwargs) -> Hypothesis:
    """Основная пишет ослышку из словаря, проверяющая — канон."""
    if backend.endswith("-lid"):
        return language_hypothesis(backend, [{"ru": 0.97, "en": 0.03}] * len(chunks), 0.1)
    text = VERIFIER_TEXT if backend == "gigaam" else PRIMARY_TEXT
    chunks = list(chunks)
    segments = [Segment(chunk.start, chunk.end, text) for chunk in chunks]
    metadata = {
        "chunks": [
            {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end, "text": text}
            for chunk in chunks
        ]
    }
    return Hypothesis(f"{backend}-fake", language, len(chunks), segments, metadata=metadata)


@requires_speech
def test_working_alias_is_not_taken_for_an_ordinary_word(
    tmp_path: Path, spoken_audio: Path, monkeypatch
) -> None:
    """2026-10-05: «Атласиан» у Atlassian ушёл в held_aliases.

    Модели сошлись на этом месте только потому, что словарь заменил алиас в
    обеих гипотезах перед сверкой. Бесспорные слова считались по сырому тексту,
    и рабочий алиас выглядел обычным словом, о котором не спорили.
    """
    monkeypatch.setattr(cli, "run_isolated_backend", fake_alias_backend)
    store = tmp_path / "glossary.yaml"
    monkeypatch.setenv("TRANSCRIBER_GLOSSARY_STORE", str(store))
    store.write_text(
        "version: 1\nentries:\n"
        "  - canonical: PostgreSQL\n    aliases: [постгрес]\n    auto_apply: true\n"
        "    learned:\n      auto_apply_written: true\n      windows: 4\n      latin: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio), "--output", str(output), "--mode", "max",
                "--language", "ru", "--no-cache", "--quiet",
                "--verifier-window-seconds", "0",
            ]
        )
    )

    assert "PostgreSQL" in (output / "readable.md").read_text(encoding="utf-8")
    (entry,) = cli.entries_of(cli.load_document(store))
    assert entry.aliases == ("постгрес",)
