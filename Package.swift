// swift-tools-version: 6.2
// qwen3vl-mlx-swift — a reusable Qwen3-VL Swift-MLX backbone that exposes the
// pre-lm_head `last_hidden_state` (the feature image/conditioning models consume).
//
// The model code is lifted from mlx-swift-lm's parity-tested MLXVLM `Qwen3VL.swift`
// (Apple Inc., Apache-2.0; port of Blaizzy/mlx-vlm qwen3_vl), reduced to the model
// surface and extended with `Qwen3VL.lastHiddenState(...)`. The generation/processor
// machinery is dropped; THW / KVCache / createAttentionMask come from MLXLMCommon.

import PackageDescription

let package = Package(
    name: "Qwen3VL",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Qwen3VL", targets: ["Qwen3VL"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "Qwen3VL",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3VL"
        ),
        .executableTarget(
            name: "Qwen3VLGate",
            dependencies: ["Qwen3VL"],
            path: "Sources/Qwen3VLGate"
        ),
    ]
)
