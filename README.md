# qwen3vl-mlx-swift

A reusable **Qwen3-VL** Swift/MLX backbone that exposes the pre-`lm_head`
`last_hidden_state` — the conditioning feature that image/edit diffusion models
consume. Adapted from mlx-swift-lm's parity-tested MLXVLM `Qwen3VL.swift`
(Apache-2.0), reduced to the model surface and extended with
`Qwen3VL.lastHiddenState(...)`. Generation/processor machinery is dropped; THW /
KVCache / createAttentionMask come from `MLXLMCommon`.

```swift
import Qwen3VL

let model = try Qwen3VLLoader.load(directory: snapshotURL)  // mlx-community/Qwen3-VL-8B-Instruct
let hidden = model.lastHiddenState(...)                     // pre-lm_head conditioning features
```

## Consuming it

This package is the **Qwen3-VL conditioner backbone consumed by the Boogu-Image
port** (Lumina2/NextDiT DiT + Qwen3-VL-8B conditioner). It serves the raw
conditioning hidden state by version so the diffusion wrapper depends on one
verified VL core rather than re-porting the encoder.

Add by tagged URL:
`.package(url: "https://github.com/xocialize/qwen3vl-mlx-swift", from: "0.1.0")`,
then `import Qwen3VL`.

## Parity gate

`Qwen3VLGate` compares `Qwen3VL.lastHiddenState(...)` against the Boogu torch
goldens (`last_hidden_state`), text-only (T2I) and vision-merged (Edit):

```
swift run Qwen3VLGate --preprocess <weightsDir> <fixturesDir> <image.png>
swift run Qwen3VLGate --t2i        <weightsDir> <fixturesDir>
swift run Qwen3VLGate --edit       <weightsDir> <fixturesDir>
```

Parity fixtures (`fixtures/*.safetensors`) are not tracked in git — regenerate them
from the Boogu Python-MLX oracle. Use `dtype: .float32` for parity gates,
`.bfloat16` for production.
