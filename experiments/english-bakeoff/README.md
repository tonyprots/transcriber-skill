# Английский bake-off

Выбор моделей английского маршрута, замер 2026-09-20. Результат —
[report.md](report.md), цифры перенесены в
`skills/transcriber/references/quality.md`.

```bash
.venv/bin/python experiments/english-bakeoff/build_corpus.py
.venv/bin/python experiments/english-bakeoff/run.py
.venv/bin/python experiments/english-bakeoff/report.py
```

`build_corpus.py` берёт 30 фрагментов Earnings-22 (CC BY-SA 4.0) и 10 AMI
(CC BY 4.0) с Hugging Face, `run.py` прогоняет кандидатов теми же бэкендами,
что и скилл, `report.py` собирает отчёт, `rover.py` — опыт с голосованием
трёх моделей (`rover.json`). Построчные метрики — `scores.json`.

В репозиторий не входят `corpus/`, `models/` и `runs/`: звук и полные
гипотезы восстанавливаются скриптами выше.
