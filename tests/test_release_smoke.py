"""Прогон релиза на настоящих весах: путь целиком, а не качество.

Запускается `pytest -m slow` перед тегом и в `.github/workflows/release-smoke.yml`
на Linux и macOS. Проверяет то, что не ловят тесты с подменёнными бэкендами:
закреплённые веса скачиваются и грузятся, пачка из двух записей доходит до
манифеста, и в манифесте — те ревизии, что в каталоге. WER здесь не считается:
две фразы Common Voice выбраны так, что обе модели распознают их без ошибок, и
искомое слово — проверка того, что текст вообще пришёл.
"""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.catalog import CATALOG, unpinned

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "smoke"
EXPECTED = {
    "common_voice_ru_18988622": "принципами",
    "common_voice_ru_26606848": "инструменты",
}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нужен ffmpeg"),
]


def test_release_path_on_pinned_weights(tmp_path: Path) -> None:
    assert not unpinned(), "смоук релиза гоняется только на закреплённых весах"
    output = tmp_path / "out"
    sources = [str(FIXTURES / f"{name}.mp3") for name in EXPECTED]
    results = cli.run_batch(
        cli.build_parser().parse_args(
            [*sources, "--output", str(output), "--mode", "max", "--language", "ru", "--no-cache", "--quiet"]
        )
    )
    failed = [item for item in results if item.get("error")]
    assert not failed, failed
    pinned = {entry.model: entry.revision for entry in CATALOG if entry.revision}
    for name, word in EXPECTED.items():
        result = output / name
        manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 8
        revisions = manifest["model_revisions"]
        assert revisions, "манифест не назвал ни одной модели"
        for model, revision in revisions.items():
            if model in pinned:
                assert revision == pinned[model], f"{model}: {revision} вместо закреплённой"
        raw = json.loads((result / "raw.json").read_text(encoding="utf-8"))
        # Какие модели реально отработали: на Linux русский маршрут идёт
        # целиком, меняется только проверяющая — faster-whisper вместо MLX.
        models = [hypothesis["model"] for hypothesis in raw["hypotheses"]]
        assert models[0] == "gigaam-v3-e2e-rnnt", models
        expected_verifier = (
            "mlx-community/whisper-large-v3-turbo-asr-fp16"
            if (platform.system(), platform.machine()) == ("Darwin", "arm64")
            else "faster-whisper-medium"
        )
        assert models[1:] == [expected_verifier], models
        for hypothesis in raw["hypotheses"]:
            weights = hypothesis["metadata"].get("weights")
            if weights is not None:
                assert weights["pinned"], f"{hypothesis['model']} загрузила незакреплённые веса"
        assert word in (result / "readable.md").read_text(encoding="utf-8").lower()
