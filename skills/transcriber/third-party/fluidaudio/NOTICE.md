# FluidAudio CLI

`bin/macos-arm64/fluidaudiocli` — сборка CLI из проекта
[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio),
лицензия Apache-2.0 (текст — в `LICENSE` рядом). Бинарник собран 2026-09-02
на macOS arm64 из `main` того дня; подпись ad-hoc, без Developer ID.

Исходники: последний коммит `main` на момент сборки —
[`5c19d5e`](https://github.com/FluidInference/FluidAudio/commit/5c19d5e12320e22bbfb7a1877b089d2665a69add)
(2026-08-30); код в нём тот же, что в `6428e29`, дальше шла только
документация. Коммит восстановлен по времени сборки, а не записан при ней:
сборка Swift побайтно не воспроизводится, поэтому сверить бинарник с
исходниками по хешу нельзя. На этом бинарнике сделан замер диаризации
5+ голосов (references/quality.md).

SHA-256: `2ee686b108c9de9c45987bff986f832dbd12b2b39a30566efaef036d95debd39`
(то же в `bin/macos-arm64/fluidaudiocli.sha256`). `setup.sh` снимает с
бинарника карантин macOS только после сверки с этой суммой, а прогон и
`doctor.py` сверяют её перед использованием. Сумма в том же репозитории
защищает от порчи и подмены файла, но не от подмены репозитория целиком —
от этого защищает только своя сборка, см. ниже.

Скилл вызывает его только для диаризации 5+ голосов:
`fluidaudiocli process <wav> --mode offline --threshold 0.8 --output <json>`.
CoreML-модели (`FluidInference/speaker-diarization-coreml`, около 34 МБ) в
пакет не входят, их скачивает `scripts/setup_fluidaudio_models.py` в
`~/Library/Application Support/FluidAudio/Models/`.

Не доверяете чужому бинарнику — соберите свой и укажите путь через
`--fluidaudio-bin` или переменную `TRANSCRIBER_FLUIDAUDIO_BIN`:

```bash
git clone https://github.com/FluidInference/FluidAudio && cd FluidAudio
git checkout 5c19d5e12320e22bbfb7a1877b089d2665a69add
swift build -c release --product fluidaudiocli
./.build/release/fluidaudiocli process sample.wav --mode offline --output out.json
```
