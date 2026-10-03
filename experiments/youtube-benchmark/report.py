#!/usr/bin/env python3
"""Сводка замера: кто сколько ошибается на ютьюбовском корпусе.

WER считается по всему корпусу разом — ошибки к словам, а не среднее из
средних: короткий фрагмент иначе весит столько же, сколько длинный.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.resolve().parents[1] / "multilingual-bakeoff"))
sys.path.insert(0, str(HERE.resolve().parents[1].parent / "skills" / "transcriber" / "scripts"))

from run_eval import prepared  # noqa: E402
from score import normalize  # noqa: E402
from textnorm import counts  # noqa: E402

TITLES = {
    "youtube-auto": "Автосубтитры ютьюба",
    "whisper-plain": "Whisper на весь файл (обычный скилл)",
    "transcriber-fast": "Наш конвейер, режим fast",
    "transcriber-max": "Наш конвейер, режим max",
}
ORDER = ["youtube-auto", "whisper-plain", "transcriber-fast", "transcriber-max"]


def aggregate(rows: list[dict]) -> dict[str, object]:
    errors = sum(row["errors"] for row in rows)
    words = sum(row["words"] for row in rows)
    speeds = [row["rtfx"] for row in rows if row["rtfx"]]
    return {
        "wer": errors / words if words else 0.0,
        "errors": errors,
        "words": words,
        "items": len(rows),
        "rtfx": statistics.median(speeds) if speeds else None,
    }


def cell(text: str) -> str:
    """Название ролика с вертикальной чертой ломает таблицу — экранируем."""
    return text.replace("|", "\\|")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus")
    parser.add_argument("--output", type=Path, default=HERE / "report.md")
    args = parser.parse_args()

    results = list(json.loads(args.results.read_text(encoding="utf-8")).values())
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in manifest}
    systems = [name for name in ORDER if any(row["system"] == name for row in results)]

    seconds = sum(item["window"][1] - item["window"][0] for item in manifest)
    parts = [
        "# Ютьюб-бенчмарк: мы, автосубтитры и Whisper без обвязки",
        "",
        f"Корпус: {len(manifest)} фрагментов, {seconds / 60:.0f} минут звука. "
        "Эталон — субтитры, написанные человеком; соперники — машинные субтитры "
        "того же видео и одиночный вызов Whisper на весь файл. Филлеры сняты у "
        "всех, включая эталон.",
        "",
        "## Итог",
        "",
    ]

    rows = []
    for system in systems:
        subset = [row for row in results if row["system"] == system]
        stats = aggregate(subset)
        speed = f"{stats['rtfx']:.1f}×" if stats["rtfx"] else "—"
        rows.append(
            [
                TITLES[system],
                f"{stats['wer']:.1%}",
                f"{stats['errors']} / {stats['words']}",
                f"{max(row['wer'] for row in subset):.1%}",
                speed,
            ]
        )
    parts.append(
        table(["Система", "WER", "Ошибок / слов", "Худший фрагмент", "Скорость"], rows)
    )
    parts += [
        "",
        "Столбец «худший фрагмент» важнее среднего. Whisper, которому отдали файл "
        "целиком, где-то выигрывает у нас за счёт длинного контекста, а где-то "
        "срывается в повтор одной фразы — на учебном рассказе про завтрак он "
        "выдал 37% при наших 1,5%. Нарезка по речи меняет редкий провал на "
        "небольшую и ровную ошибку, и это её главный смысл.",
    ]

    for language, name in (("ru", "русский"), ("en", "английский")):
        subset = [row for row in results if row["language"] == language]
        if not subset:
            continue
        parts += ["", f"## Только {name}", ""]
        rows = []
        for system in systems:
            stats = aggregate([row for row in subset if row["system"] == system])
            if not stats["items"]:
                continue
            rows.append(
                [
                    TITLES[system],
                    f"{stats['wer']:.1%}",
                    f"{stats['errors']} / {stats['words']}",
                    str(stats["items"]),
                ]
            )
        parts.append(table(["Система", "WER", "Ошибок / слов", "Фрагментов"], rows))

    parts += [
        "",
        "## Из чего складываются ошибки",
        "",
        "Вставка — слово, которое система расслышала, а в эталоне его нет. "
        "Человек, пишущий субтитры, на ходу убирает поддакивания и повторы, "
        "поэтому дословная расшифровка платит за честность. Замены и пропуски "
        "— это уже непонятый звук.",
        "",
    ]
    rows = []
    for system in systems:
        tally = {"substitutions": 0, "insertions": 0, "deletions": 0, "words": 0}
        for row in results:
            if row["system"] != system:
                continue
            reference = (args.corpus / f"{row['item']}.reference.txt").read_text(encoding="utf-8")
            item_tally = counts(
                normalize(prepared(reference)).split(), normalize(prepared(row["text"])).split()
            )
            for field, value in item_tally.items():
                tally[field] += value
        rows.append(
            [
                TITLES[system],
                f"{tally['substitutions'] / tally['words']:.1%}",
                f"{tally['insertions'] / tally['words']:.1%}",
                f"{tally['deletions'] / tally['words']:.1%}",
            ]
        )
    parts.append(table(["Система", "Замены", "Вставки", "Пропуски"], rows))

    parts += ["", "## По фрагментам", ""]
    header = ["Фрагмент", "Язык"] + [TITLES[name] for name in systems]
    rows = []
    for item in manifest:
        line = [cell(item["title"][:44]), item["language"]]
        for system in systems:
            found = [
                row for row in results if row["system"] == system and row["item"] == item["id"]
            ]
            line.append(f"{found[0]['wer']:.1%}" if found else "—")
        rows.append(line)
    parts.append(table(header, rows))

    window = HERE / "results-window.json"
    if window.exists():
        parts += [
            "",
            "## Проверенная и отвергнутая настройка: окно 28 секунд",
            "",
            "Whisper обучен на тридцатисекундных отрезках, и практика WhisperX "
            "советует склеивать речь примерно в такие окна. У нас общее окно — "
            "20 секунд. Прогон с 28 секундами на том же корпусе, режим `fast`:",
            "",
        ]
        variants = list(json.loads(window.read_text(encoding="utf-8")).values())
        rows = []
        for language, name in (("ru", "русский"), ("en", "английский")):
            wide = aggregate([row for row in variants if row["language"] == language])
            narrow = aggregate(
                [
                    row
                    for row in results
                    if row["system"] == "transcriber-fast" and row["language"] == language
                ]
            )
            rows.append([name, f"{narrow['wer']:.2%}", f"{wide['wer']:.2%}"])
        parts.append(table(["Язык", "Окно 20 с", "Окно 28 с"], rows))
        parts += [
            "",
            "Длинное окно не выиграло нигде: на английском стало хуже, на русском "
            "разница внутри шума. Настройка осталась флагом "
            "`--speech-window-seconds`, умолчание — прежние 20 секунд.",
        ]

    parts += [
        "",
        "## Что эти числа не говорят",
        "",
        "- Фрагменты отобраны по единственному доступному признаку — у видео есть "
        "субтитры, написанные человеком. Это смещает корпус в сторону учебных "
        "каналов и выступлений: там их пишут чаще, чем в разговорных подкастах.",
        "- Чем ближе авторские субтитры к машинным (столбец сходства в "
        "`corpus/manifest.json`), тем меньше замер говорит о качестве: на "
        "студийной записи распознавание ютьюба и так почти безошибочно.",
        "- Скорость измерена на этой машине, по одному прогону на фрагмент.",
        "- `fast` и `max` дают один и тот же текст — на всех двенадцати фрагментах "
        "совпало слово в слово. Так и задумано: проверяющая модель отмечает "
        "места, но не правит. Режим выбирают ради очереди проверки, а не ради "
        "точности, и вдвое большее время `max` уходит именно на неё.",
    ]

    speaker_items = [item for item in manifest if item.get("speaker_labels")]
    if speaker_items:
        names = ", ".join(item["id"] for item in speaker_items)
        parts.append(
            f"- Реплики подписаны именами говорящих в: {names}. Имена сняты из "
            "эталона — их нет в звуке."
        )

    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"→ {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
