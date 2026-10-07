# FluidAudio CLI

`fluidaudiocli` — CLI проекта
[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio),
лицензия Apache-2.0 (текст — в `LICENSE` рядом). Скилл вызывает его для
диаризации (с 0.24 — при любом числе голосов):
`fluidaudiocli process <wav> --mode offline --threshold 0.7 --output <json>`,
с известным числом голосов — ещё `--num-speakers N`.

## Откуда берётся бинарник

В git его нет. Его собирает CI этого репозитория
(`.github/workflows/fluidaudio.yml`) из закреплённого коммита
[`5c19d5e`](https://github.com/FluidInference/FluidAudio/commit/5c19d5e12320e22bbfb7a1877b089d2665a69add)
на `macos-15` (arm64), подписывает ad-hoc и публикует в неизменяемом Release
[`fluidaudio-5c19d5e`](https://github.com/tonyprots/transcriber-skill/releases/tag/fluidaudio-5c19d5e)
вместе с attestation происхождения (SLSA, Sigstore). `setup.sh` на Apple
Silicon скачивает его по адресу из `bin/macos-arm64/fluidaudiocli.url` и
кладёт на место, только если SHA-256 совпал с
`bin/macos-arm64/fluidaudiocli.sha256`:

`521016abdc7c2a1cdc31c8fe37043f93062cae6d842c90431a05d96ab9bde4a8`

Прогон и `doctor.py` сверяют эту сумму перед использованием. Что бинарник
собран именно этой сборкой из этого коммита, проверяет любой:

```bash
gh attestation verify bin/macos-arm64/fluidaudiocli --repo tonyprots/transcriber-skill
```

Attestation подписывает GitHub, а не владелец репозитория, поэтому она
доказывает и то, чего не доказывает сумма в том же репозитории: бинарник
вышел из открытого workflow, а не собран где-то ещё.

## Замер

На этой сборке схема из шести голосов A–B–C–D–E–F–A–B–C–D–E–F даёт proxy DER
0,8% (все шесть голосов) при порогах 0,5–0,8, шесть записей MINDS-14 — 9,8%
(шесть голосов) при 0,5–0,9; выбран 0,7. Прежний бинарник, собранный вручную
2026-09-02, давал 3,7% и 25,9% (пять голосов) — значит, он был собран не из
`5c19d5e`, как считалось. Замер 2026-10-03, `experiments/diarization-bakeoff`, не опубликовано: записи схемы собраны локально.

## Своя сборка

Не доверяете и сборке в CI — соберите сами (нужен Swift 6, Xcode 16) и
укажите путь через `--fluidaudio-bin` или `TRANSCRIBER_FLUIDAUDIO_BIN`:

```bash
git clone https://github.com/FluidInference/FluidAudio && cd FluidAudio
git checkout 5c19d5e12320e22bbfb7a1877b089d2665a69add
swift build -c release --product fluidaudiocli
./.build/release/fluidaudiocli process sample.wav --mode offline --threshold 0.7 --output out.json
```

CoreML-модели (`FluidInference/speaker-diarization-coreml`, около 34 МБ)
скачивает `scripts/setup_fluidaudio_models.py` в
`~/Library/Application Support/FluidAudio/Models/`.
