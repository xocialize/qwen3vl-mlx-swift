// Weight loading for the reusable Qwen3-VL backbone. Loads the stock
// mlx-community/Qwen3-VL-8B-Instruct snapshot (HF `config.json` + sharded
// safetensors), applies the model's own `sanitize`, and strictly verifies.

import Foundation
import MLX
import MLXNN

public enum Qwen3VLError2: Error, CustomStringConvertible {
    case loading(String)
    public var description: String {
        switch self { case .loading(let m): return "Qwen3VL loading error: \(m)" }
    }
}

public enum Qwen3VLLoader {
    /// Load a Qwen3-VL snapshot directory into a `Qwen3VL`. `dtype` upcasts weights
    /// (use .float32 for parity gates; .bfloat16 for production).
    public static func load(directory: URL, dtype: DType = .bfloat16) throws -> Qwen3VL {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
        let model = Qwen3VL(config)

        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Qwen3VLError2.loading("no .safetensors under \(directory.path)") }

        var raw: [String: MLXArray] = [:]
        for f in files { raw.merge(try MLX.loadArrays(url: f)) { a, _ in a } }
        var weights = model.sanitize(weights: raw)
        weights = weights.mapValues { $0.asType(dtype) }

        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw Qwen3VLError2.loading(
                "missing \(missing.count) module keys, e.g. \(missing.prefix(6))")
        }
        // Drop any extra (e.g. tied lm_head already filtered by sanitize).
        let filtered = weights.filter { moduleKeys.contains($0.key) }
        model.update(parameters: ModuleParameters.unflattened(filtered))
        eval(model)
        return model
    }
}
