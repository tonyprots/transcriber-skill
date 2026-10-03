#!/usr/bin/env python3
"""Приведение субтитров к виду, в котором их честно сравнивать с расшифровкой.

Субтитры пишет человек для читателя, а распознавание отвечает за сказанное.
Поэтому из эталона убирается то, чего в звуке нет и быть не может: имена
говорящих в начале реплики, пометки вроде «(Laughter)» и «[музыка]», ударения
в учебных видео, маска ругательств `[ __ ]`, которую ставит сам ютьюб.

Отдельная мера — сходство двух текстов. Считать его посимвольно нельзя:
`SequenceMatcher` на строках длиннее двухсот символов объявляет частые символы
мусором, и два почти одинаковых текста дают 25%. Сравниваем по словам, через
тот же расчёт ошибок, что и в основном скорере.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multilingual-bakeoff"))

from score import error_rate  # noqa: E402

# Имя говорящего в начале реплики: «Макс:», «[Фёдор]: -», «>> Speaker:».
SPEAKER_LABEL = re.compile(
    r"^\s*(?:>>+\s*)?(?:\[[^\]]{1,32}\]|[A-ZА-ЯЁ][\w'’.-]{0,20})\s*:\s*[-–—]?\s*"
)
BARE_ARROWS = re.compile(r"^\s*>>+\s*")
# Пометки не-речи: (Laughter), [музыка], *смеётся*, ♪.
ANNOTATION = re.compile(r"[(\[*][^)\]*]{0,40}[)\]*]|[♪♫]")
COMBINING = re.compile(r"[̀-ͯ]")


def clean_cue(text: str) -> str:
    """Одна реплика без имени говорящего и служебных пометок."""
    text = SPEAKER_LABEL.sub("", text)
    text = BARE_ARROWS.sub("", text)
    text = ANNOTATION.sub(" ", text)
    return " ".join(text.split())


def has_speaker_label(text: str) -> bool:
    return bool(SPEAKER_LABEL.match(text) or BARE_ARROWS.match(text))


def fold(text: str) -> str:
    """Снимает ударения и различие е/ё — их не слышно и не считают за ошибку."""
    text = unicodedata.normalize("NFD", text)
    text = COMBINING.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return text.replace("ё", "е").replace("Ё", "Е")


def counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Разбивает ошибки на замены, вставки и пропуски.

    Общий WER скрывает, чем именно система провинилась. На этом корпусе это
    важно: субтитры пишет человек и на ходу вычищает поддакивания, поэтому
    честно расслышанное «угу» попадает в ошибку как вставка. Замена и пропуск —
    другое дело, это настоящее непонимание звука.
    """
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    move = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0], move[row][0] = row, "d"
    for column in range(1, columns):
        distance[0][column], move[0][column] = column, "i"
    for row in range(1, rows):
        for column in range(1, columns):
            same = reference[row - 1] == hypothesis[column - 1]
            options = (
                (distance[row - 1][column - 1] + (0 if same else 1), "=" if same else "s"),
                (distance[row - 1][column] + 1, "d"),
                (distance[row][column - 1] + 1, "i"),
            )
            distance[row][column], move[row][column] = min(options)
    tally = {"substitutions": 0, "insertions": 0, "deletions": 0, "words": len(reference)}
    row, column = rows - 1, columns - 1
    while row or column:
        step = move[row][column]
        if step in ("=", "s"):
            tally["substitutions"] += step == "s"
            row, column = row - 1, column - 1
        elif step == "d":
            tally["deletions"] += 1
            row -= 1
        else:
            tally["insertions"] += 1
            column -= 1
    return tally


def similarity(left: str, right: str) -> float:
    """1 − доля ошибок: 1.0 — тексты совпадают, 0.0 — ничего общего."""
    errors, total = error_rate(fold(left), fold(right))
    if not total:
        return 0.0
    return max(0.0, 1.0 - errors / total)
