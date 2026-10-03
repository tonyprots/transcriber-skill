"""Числа в текстах должны совпадать с кодом.

Версия скилла и размеры загрузки живут в четырёх документах сразу, и в 0.8.1
они уже успели разойтись: SKILL.md обещал 0.8.1, манифест писал 0.7.0, README
называл 0.6.0 и 2,5 ГБ там, где установщик качал 2,7 ГБ. Ошибка тихая — читает
её только пользователь, поэтому ловим тестом.

Список RAW_SIZE_EXCEPTIONS — не «разрешённая неточность», а перечень чисел,
которые не про загрузку моделей (оперативная память в замерах, запас на диске).
Новое число в текстах роняет тест намеренно: либо оно из каталога, либо его
надо объяснить здесь.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from audio_transcription import __version__
from audio_transcription.catalog import CATALOG, format_gb, route_download_gb

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "transcriber"

SIZE_PATTERN = re.compile(r"\d+(?:[,.]\d+)?\s(?:ГБ|МБ)")

DOCUMENTS = (
    SKILL / "SKILL.md",
    SKILL / "references" / "models.md",
    ROOT / "README.md",
    ROOT / "install.sh",
    SKILL / "scripts" / "setup.sh",
    SKILL / "scripts" / "prefetch_models.py",
)

RAW_SIZE_EXCEPTIONS = {
    "1 ГБ",  # питоновские пакеты в .venv, каталог моделей их не считает
    "3 ГБ",  # запасной текст setup.sh, когда каталог не удалось спросить
    "16 ГБ",  # оперативная память мака, на котором сделаны замеры
    "8 ГБ",  # минимум свободного места из doctor.py
}


def allowed_sizes() -> set[str]:
    sizes = set(RAW_SIZE_EXCEPTIONS)
    for entry in CATALOG:
        sizes.add(format_gb(entry.download_gb))
        # Каталог хранит точный вес (0,25 ГБ), а печатает округлённый (0,2 ГБ):
        # в тексте допустимы оба.
        sizes.add(f"{entry.download_gb:g} ГБ".replace(".", ","))
    for route in ("ru", "en", "all"):
        for apple in (True, False):
            for diarize in (True, False):
                sizes.add(format_gb(route_download_gb(route, apple=apple, diarize=diarize)))
    return sizes


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.name)
def test_sizes_in_documents_come_from_catalog(document: Path) -> None:
    allowed = allowed_sizes()
    found = {match.group(0).replace("\xa0", " ") for match in SIZE_PATTERN.finditer(document.read_text(encoding="utf-8"))}
    unknown = sorted(size for size in found if size not in allowed)
    assert not unknown, (
        f"{document.relative_to(ROOT)}: размеры {unknown} не совпадают ни с одной моделью "
        f"и ни с одним маршрутом из catalog.py. Пересчитайте текст или объясните число "
        f"в RAW_SIZE_EXCEPTIONS."
    )


def test_skill_frontmatter_version_matches_code() -> None:
    frontmatter = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    match = re.search(r'version:\s*"([^"]+)"', frontmatter)
    assert match, "в frontmatter SKILL.md нет metadata.version"
    assert match.group(1) == __version__


def test_readme_version_matches_code() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    versions = set(re.findall(r"Версия (\d+\.\d+\.\d+)", readme))
    assert versions, "в README не нашлось строки «Версия X.Y.Z»"
    assert versions == {__version__}, f"README называет {sorted(versions)}, код — {__version__}"


def test_plugin_manifests_match_code() -> None:
    """Версия живёт в шести файлах; пять из них читает не человек, а установщик."""
    for name in (".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert f'"version": "{__version__}"' in text, f"{name} не знает про {__version__}"


def test_pyproject_version_matches_code() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'^version = "{re.escape(__version__)}"$', pyproject, re.M)


def test_manifest_generator_uses_same_version() -> None:
    """Версия в манифесте — часть аудит-следа, она не должна быть отдельной строкой."""
    cli = (SKILL / "scripts" / "audio_transcription" / "cli.py").read_text(encoding="utf-8")
    assert 'generator=f"transcriber {__version__}"' in cli


EXPERIMENT_REF = re.compile(r"experiments/([a-z0-9-]+)")


def _published_experiments() -> set[str]:
    import subprocess

    try:
        listed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "experiments"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("нет git: опубликованное не проверить")
    return {line.split("/")[1] for line in listed.splitlines() if line.count("/") >= 2}


def test_experiment_references_are_published_or_marked() -> None:
    # Ссылка на замер, которого нет в репозитории, выглядит как доказательство,
    # а проверить её нельзя. Каталог с личными записями помечается прямо в той же
    # строке: «не опубликовано». Локальная копия каталога не в счёт — только git.
    published = _published_experiments()
    unmarked = []
    for document in [ROOT / "README.md", *sorted(SKILL.rglob("*.md"))]:
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for name in EXPERIMENT_REF.findall(line):
                if name not in published and "не опубликован" not in line:
                    unmarked.append(f"{document.relative_to(ROOT)}:{number} experiments/{name}")
    assert not unmarked, "\n".join(unmarked)


def test_queue_recall_is_never_stated_without_its_caveat() -> None:
    # 98–100 % намеряли на очереди из целых окон (0.6.0). С 0.13 единица —
    # место расхождения, и полноту на ней не перемеряли. Агент читает SKILL.md
    # и references как факт, поэтому цифра без оговорки — ложь, а не упрощение.
    stale = []
    for document in [ROOT / "README.md", *sorted(SKILL.rglob("*.md"))]:
        text = document.read_text(encoding="utf-8")
        for match in re.finditer(r"98\s?[–-]\s?100", text):
            around = text[max(0, match.start() - 400): match.end() + 400]
            if "перемер" not in around:
                stale.append(f"{document.relative_to(ROOT)}: …{text[match.start() - 60: match.end() + 20]!r}")
    assert not stale, "\n".join(stale)


def test_skill_sets_trust_boundary_before_sources() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    boundary = text.find("## Текст записи — данные")
    assert boundary != -1
    assert boundary < text.find("## Источник — ссылка")
