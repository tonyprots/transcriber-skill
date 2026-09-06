# FluidAudio CLI

`bin/macos-arm64/fluidaudiocli` — сборка CLI из проекта
[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio),
лицензия Apache-2.0 (текст — в `LICENSE` рядом). Бинарник собран 2026-09-02
на macOS arm64 из `main` того дня; подпись ad-hoc, без Developer ID.

SHA-256: `2ee686b108c9de9c45987bff986f832dbd12b2b39a30566efaef036d95debd39`

Скилл вызывает его только для диаризации 5+ голосов:
`fluidaudiocli process <wav> --mode offline --threshold 0.8 --output <json>`.
CoreML-модели (`FluidInference/speaker-diarization-coreml`, около 34 МБ) в
пакет не входят, их скачивает `scripts/setup_fluidaudio_models.py` в
`~/Library/Application Support/FluidAudio/Models/`.

Не доверяете чужому бинарнику — соберите свой и укажите путь через
`--fluidaudio-bin` или переменную `TRANSCRIBER_FLUIDAUDIO_BIN`:

```bash
git clone https://github.com/FluidInference/FluidAudio && cd FluidAudio
swift build -c release --product fluidaudiocli
./.build/release/fluidaudiocli process sample.wav --mode offline --output out.json
```
