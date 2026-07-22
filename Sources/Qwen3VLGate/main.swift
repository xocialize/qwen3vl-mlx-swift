// CLI parity gate for the Qwen3-VL hidden-state backbone. Compares
// `Qwen3VL.lastHiddenState(...)` on pre-tokenized golden inputs vs the Boogu torch
// goldens (last_hidden_state), text-only (T2I) and vision-merged (Edit).
//
//   swift run Qwen3VLGate --t2i  <weightsDir> <fixturesDir>
//   swift run Qwen3VLGate --edit <weightsDir> <fixturesDir>

import CoreGraphics
import Foundation
import ImageIO
import Qwen3VL
import MLX
import MLXLMCommon
import Tokenizers

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

/// Decode a PNG/JPEG to interleaved RGB8 (sRGB).
func decodeRGB(_ url: URL) -> (rgb: [UInt8], width: Int, height: Int)? {
    guard let data = try? Data(contentsOf: url),
          let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let (w, h) = (cg.width, cg.height)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return (rgb, w, h)
}

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).flattened()
    let y = b.asType(.float32).flattened()
    let dot = (x * y).sum().item(Float.self)
    let nx = sqrt((x * x).sum()).item(Float.self)
    let ny = sqrt((y * y).sum()).item(Float.self)
    return dot / (nx * ny)
}

let args = Array(CommandLine.arguments.dropFirst())

// Weights-only probe: verify a snapshot decodes + loads strictly at whatever
// size it is (this backbone was gated at 8B; Mage-Flow ships Qwen3-VL-4B), and
// exercise the flat-positionIds override that Mage-Flow's patched encoder needs.
//   swift run Qwen3VLGate --load-probe <weightsDir>
if args.first == "--load-probe" {
    guard args.count >= 2 else { err("usage: Qwen3VLGate --load-probe <weightsDir>"); exit(2) }
    let dir = URL(fileURLWithPath: args[1])
    var model: Qwen3VL!
    do {
        try Device.withDefaultDevice(.cpu) {
            model = try Qwen3VLLoader.load(directory: dir, dtype: .bfloat16)
        }
    } catch {
        err("[load-probe] LOAD FAILED: \(error)"); exit(1)
    }
    let t = model.config.textConfiguration
    let v = model.config.visionConfiguration
    err("[load-probe] text  : hidden \(t.hiddenSize) inter \(t.intermediateSize) layers \(t.numHiddenLayers) "
        + "heads \(t.numAttentionHeads)/\(t.numKeyValueHeads) headDim \(t.headDim) vocab \(t.vocabSize)")
    err("[load-probe] text  : ropeTheta \(t.ropeTheta) rmsEps \(t.rmsNormEps) tied \(t.tieWordEmbeddings) "
        + "mropeSection \(t.ropeScaling?.mropeSection.map(String.init(describing:)) ?? "nil")")
    err("[load-probe] vision: hidden \(v.hiddenSize) depth \(v.depth) outHidden \(v.outHiddenSize) "
        + "heads \(v.numHeads) patch \(v.patchSize) merge \(v.spatialMergeSize) deepstack \(v.deepstackVisualIndexes)")

    // Synthetic text-only forward. Token ids kept well inside vocab.
    let L = 24
    let ids = MLXArray((0 ..< L).map { Int32(1000 + $0 * 7) }, [1, L])

    let hDefault = try! model.lastHiddenState(inputIds: ids)
    eval(hDefault)
    let flat = Qwen3VL.flatPositionIds(sequenceLength: L)
    let hFlat = try! model.lastHiddenState(inputIds: ids, positionIds: flat)
    eval(hFlat)

    let finite = !hDefault.asType(.float32).flattened().asArray(Float.self).contains { !$0.isFinite }
    err("[load-probe] forward: shape \(hDefault.shape) finite \(finite) "
        + "hidden==config \(hDefault.shape.last == t.hiddenSize)")
    err("[load-probe] flatPositionIds shape \(flat.shape) (expect [3, 1, \(L)])")

    // Text-only has no image tokens, so getRopeIndex already degenerates to a
    // flat arange -> the override must be a NO-OP here.
    let d = abs(hDefault.asType(.float32) - hFlat.asType(.float32)).max().item(Float.self)
    err("[load-probe] override vs default on text-only: max_abs \(d) (expect 0 — no image tokens)")

    var imageOK = true
    if args.count >= 3, let img = decodeRGB(URL(fileURLWithPath: args[2])) {
        // Self-contained image case — no external fixtures. Build an id sequence
        // whose image_pad count matches the processor's merged grid, then check
        // that spatial M-RoPE and the flat override actually DIVERGE. This is the
        // case Mage-Flow depends on; a no-op here would mean the override is
        // cosmetic and conditioning would be silently wrong at high resolution.
        let (pv, thw) = Qwen3VLImageProcessor().preprocess(
            rgb: img.rgb, width: img.width, height: img.height)
        let merge = v.spatialMergeSize
        let nImg = thw.product / (merge * merge)
        var seq: [Int32] = [Int32(model.config.visionStartTokenId)]
        seq += Array(repeating: Int32(model.config.imageTokenIndex), count: nImg)
        seq.append(Int32(model.config.visionEndTokenId))
        seq += (0 ..< 8).map { Int32(1000 + $0 * 7) }
        let iids = MLXArray(seq, [1, seq.count])

        let hImgDefault = try! model.lastHiddenState(
            inputIds: iids, pixelValues: pv, imageGridTHW: [thw])
        eval(hImgDefault)
        let flatI = Qwen3VL.flatPositionIds(sequenceLength: seq.count)
        let hImgFlat = try! model.lastHiddenState(
            inputIds: iids, pixelValues: pv, imageGridTHW: [thw], positionIds: flatI)
        eval(hImgFlat)

        let di = abs(hImgDefault.asType(.float32) - hImgFlat.asType(.float32))
            .max().item(Float.self)
        let ci = cosine(hImgDefault, hImgFlat)
        err("[load-probe] image: grid (\(thw.t),\(thw.h),\(thw.w)) imgTokens \(nImg) "
            + "seq \(seq.count) shape \(hImgDefault.shape)")
        err("[load-probe] image: M-RoPE vs flat  max_abs \(di)  cos \(ci)  "
            + "(expect NON-zero — the paths must diverge)")
        imageOK = di > 0
    } else if args.count >= 3 {
        err("[load-probe] image: could not decode \(args[2])"); imageOK = false
    } else {
        err("[load-probe] image: skipped (pass an image path as arg 3 to test divergence)")
    }

    let pass = finite && hDefault.shape.last == t.hiddenSize && flat.shape == [3, 1, L]
        && d == 0 && imageOK
    err("[load-probe] \(pass ? "PASS" : "FAIL")")
    exit(pass ? 0 : 1)
}

// Autoregressive smoke test — the path Mage-Flow's mandatory content filter needs
// (two greedy .generate() calls, <=192 new tokens, fail-closed).
//   swift run Qwen3VLGate --gen-probe <weightsDir> ["prompt"]
if args.first == "--gen-probe" {
    guard args.count >= 2 else { err("usage: Qwen3VLGate --gen-probe <weightsDir> [prompt]"); exit(2) }
    let dir = URL(fileURLWithPath: args[1])
    let question = args.count >= 3 ? args[2] : "What is the capital of France? Answer in one word."

    var model: Qwen3VL!
    try Device.withDefaultDevice(.cpu) {
        model = try Qwen3VLLoader.load(directory: dir, dtype: .bfloat16)
    }
    let tok = try await AutoTokenizer.from(modelFolder: dir)

    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        + "<|im_start|>user\n\(question)<|im_end|>\n<|im_start|>assistant\n"
    let ids = tok.encode(text: prompt)
    err("[gen-probe] prompt \(ids.count) tokens")

    let cap = args.count >= 4 ? (Int(args[3]) ?? 48) : 48
    let t0 = Date()
    let outIds = try model.generate(
        inputIds: MLXArray(ids.map { Int32($0) }, [1, ids.count]), maxTokens: cap)
    let dt = Date().timeIntervalSince(t0)

    let text = tok.decode(tokens: outIds.map { Int($0) })
    err("[gen-probe] generated \(outIds.count) tokens in \(String(format: "%.1f", dt))s "
        + "(\(String(format: "%.1f", Double(outIds.count) / dt)) tok/s)")
    err("[gen-probe] ids  : \(outIds.prefix(16))")
    err("[gen-probe] text : \"\(text)\"")

    // Structural checks. NOTE: hitting the token cap is NOT a failure — a long
    // answer legitimately runs out of budget. EOS termination is reported
    // separately (verify it with a short-answer prompt), and multi-token output
    // is what exercises the decode-step cache/rope-delta path.
    let inVocab = outIds.allSatisfy { $0 >= 0 && Int($0) < model.config.textConfiguration.vocabSize }
    let hitCap = outIds.count == cap
    let pass = !outIds.isEmpty && inVocab
        && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    err("[gen-probe] inVocab \(inVocab) endedOn \(hitCap ? "CAP" : "EOS") "
        + "multiStepDecode \(outIds.count > 1) -> \(pass ? "PASS" : "FAIL")")
    exit(pass ? 0 : 1)
}

guard args.count >= 3 else { err("usage: Qwen3VLGate --t2i|--edit <weightsDir> <fixturesDir>"); exit(2) }
let gate = args[0]
let weights = URL(fileURLWithPath: args[1])
let fixtures = URL(fileURLWithPath: args[2])

// Standalone NAX split-K GEMM repro — no weights, random seeded tensors.
// Broken window: half-precision, batch 1, M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N).
if gate == "--matmul-probe-rand" {
    // Seeded host-side uniform(-1,1) randoms — deterministic, no MLXRandom dep.
    var lcg: UInt64 = 0x9E37_79B9_7F4A_7C15
    func randArray(_ count: Int) -> [Float] {
        (0 ..< count).map { _ in
            lcg = lcg &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: lcg >> 11)) / Float(Int64.max >> 11)
        }
    }
    let (K, N) = (12288, 4096)
    let b = MLXArray(randArray(N * K), [N, K]).asType(.bfloat16)
    for m in [512, 896, 1024, 2048] {
        let a = MLXArray(randArray(m * K), [m, K]).asType(.bfloat16)
        let y = matmul(a, b.T)
        let yRef = matmul(a.asType(.float32), b.asType(.float32).T)
        eval(y, yRef)
        let mab = abs(y.asType(.float32) - yRef).max().item(Float.self)
        err("  M=\(m) K=\(K) N=\(N) bf16: cos \(cosine(y, yRef)) max_abs_vs_fp32 \(mab)")
    }
    exit(0)
}

// Image preprocessing parity — needs no model weights.
if gate == "--preprocess" {
    guard args.count >= 4, let img = decodeRGB(URL(fileURLWithPath: args[3])) else {
        err("--preprocess <weightsDir> <fixturesDir> <image.png>"); exit(2)
    }
    let g = try! MLX.loadArrays(url: fixtures.appendingPathComponent("cond_edit.safetensors"))
    let (pv, thw) = Qwen3VLImageProcessor().preprocess(rgb: img.rgb, width: img.width, height: img.height)
    eval(pv)
    let mab = abs(pv.asType(.float32) - g["pixel_values"]!).max().item(Float.self)
    err("[preprocess] grid (\(thw.t),\(thw.h),\(thw.w)) pixel_values \(pv.shape) "
        + "cos \(cosine(pv, g["pixel_values"]!)) max_abs \(mab)")
    exit(mab <= 1e-3 ? 0 : 1)
}

var ok = false
do {
    // Load on the CPU stream (avoid a multi-GB read riding a GPU command buffer); the
    // loader evals the model so the fp32 upcast materializes here, not in the forward.
    var model: Qwen3VL!
    try Device.withDefaultDevice(.cpu) {
        let dtype: DType = (args.count > 3 && args[3] == "fp32") ? .float32 : .bfloat16
        model = try Qwen3VLLoader.load(directory: weights, dtype: dtype)
    }
    // Forward on the default (GPU) stream — a CPU-pinned vision forward fences a Metal
    // buffer past the watchdog (skill: load CPU, run GPU).
    switch gate {
    case "--t2i":
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent("cond_t2i.safetensors"))
        let h = try model.lastHiddenState(inputIds: g["input_ids"]!)
        eval(h)
        let cos = cosine(h, g["feats"]!)
        let mab = abs(h.asType(.float32) - g["feats"]!).max().item(Float.self)
        err("[T2I] shape \(h.shape) cos \(cos) max_abs \(mab)")
        ok = cos >= 0.999
    case "--edit":
        let editGolden = (args.count > 3 && args[3] == "fp32")
            ? "cond_edit_mlxvlm_fp32.safetensors" : "cond_edit_mlxvlm.safetensors"
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent(editGolden))
        let grid = g["grid"]!.asType(.int32).asArray(Int32.self)  // [t,h,w]
        let thw = THW(Int(grid[0]), Int(grid[1]), Int(grid[2]))
        let h = try model.lastHiddenState(
            inputIds: g["input_ids"]!, pixelValues: g["pixel_values"]!, imageGridTHW: [thw])
        eval(h)
        let cos = cosine(h, g["feats"]!)
        let mab = abs(h.asType(.float32) - g["feats"]!).max().item(Float.self)
        err("[Edit] shape \(h.shape) cos \(cos) max_abs \(mab)")
        // fp32 reaches ~0.998 (faithful to mlx-vlm); bf16 ~0.967 (SDPA accumulation =
        // the precision level the Python port shipped as clean edits).
        ok = cos >= ((args.count > 3 && args[3] == "fp32") ? 0.998 : 0.96)
    case "--edit-bisect":
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent("edit_intermediates.safetensors"))
        let grid = g["grid"]!.asType(.int32).asArray(Int32.self)
        let thw = THW(Int(grid[0]), Int(grid[1]), Int(grid[2]))
        let d = try model.debugEdit(
            inputIds: g["input_ids"]!, pixelValues: g["pixel_values"]!, imageGridTHW: [thw])
        eval(d.visHidden); eval(d.merged); eval(d.positionIds)
        func stage(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let mab = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            err("  \(name): cos \(cosine(a, b)) max_abs \(mab) shape \(a.shape) vs \(b.shape)")
        }
        stage("vis_hidden", d.visHidden, g["vis_hidden"]!)
        for i in 0..<d.deepstack.count { stage("deep_\(i)", d.deepstack[i], g["deep_\(i)"]!) }
        stage("merged", d.merged, g["merged_embeds"]!)
        stage("position_ids", d.positionIds, g["position_ids"]!)
        ok = true
    case "--matmul-probe":
        // Raw GEMM check: down_proj(gated) vs the oracle's product, plus shape sweep.
        let mlpRef = try MLX.loadArrays(url: fixtures.appendingPathComponent("mlp0_bf16.safetensors"))
        let gated = mlpRef["gated"]!            // [1, 1100, 12288] bf16
        let refOut = mlpRef["mlp_down"]!        // [1, 1100, 4096] bf16
        let w = model.debugDownProjWeight(layer: 0)  // [4096, 12288] bf16
        err("gated dtype \(gated.dtype) w dtype \(w.dtype)")
        for rows in [64, 256, 512, 640, 768, 896, 1024, 1056, 1088, 1100] {
            let x = gated[0..., 0 ..< rows, 0...]
            let y = matmul(x, w.T)
            eval(y)
            let r = refOut[0..., 0 ..< rows, 0...]
            let mab = abs(y.asType(.float32) - r.asType(.float32)).max().item(Float.self)
            err("  rows=\(rows): cos \(cosine(y, r)) max_abs \(mab)")
        }
        // fp32 control at full length
        let y32 = matmul(gated.asType(.float32), w.asType(.float32).T)
        eval(y32)
        err("  fp32 full: cos \(cosine(y32, refOut)) max_abs \(abs(y32 - refOut.asType(.float32)).max().item(Float.self))")
        // 2-D (no batch dim) control
        let y2d = matmul(gated[0], w.T)
        eval(y2d)
        err("  2d full: cos \(cosine(y2d, refOut[0]))")
        ok = true
    case "--attn0-bisect":
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent("edit_intermediates.safetensors"))
        var ref = try MLX.loadArrays(url: fixtures.appendingPathComponent("attn0_bf16.safetensors"))
        if let mlpRef = try? MLX.loadArrays(url: fixtures.appendingPathComponent("mlp0_bf16.safetensors")) {
            ref.merge(mlpRef) { a, _ in a }
        }
        let stages = model.debugAttn0(
            mergedEmbeds: g["merged_embeds"]!, positionIds: g["position_ids"]!)
        for (name, a) in stages {
            guard let b = ref[name] else { err("  \(name): (no ref)"); continue }
            eval(a)
            let mab = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            err("  \(name): cos \(cosine(a, b)) max_abs \(mab) shape \(a.shape) vs \(b.shape)")
        }
        ok = true
    case "--lm-bisect":
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent("edit_intermediates.safetensors"))
        let lm = try MLX.loadArrays(url: fixtures.appendingPathComponent("lm_layers.safetensors"))
        let deepstack = (0..<3).map { g["deep_\($0)"]! }
        let (layers, final) = model.debugLanguageLayers(
            inputIds: g["input_ids"]!, mergedEmbeds: g["merged_embeds"]!,
            positionIds: g["position_ids"]!, deepstackEmbeds: deepstack)
        func stage(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let mab = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            err("  \(name): cos \(cosine(a, b)) max_abs \(mab)")
        }
        for (i, h) in layers.enumerated() {
            stage(String(format: "lm_%02d", i), h, lm[String(format: "lm_%02d", i)]!)
        }
        eval(final)
        stage("lm_final", final, lm["lm_final"]!)
        ok = true
    case "--vision-pre":
        let g = try MLX.loadArrays(url: fixtures.appendingPathComponent("vision_pre.safetensors"))
        let grid = g["grid"]!.asType(.int32).asArray(Int32.self)
        let thw = THW(Int(grid[0]), Int(grid[1]), Int(grid[2]))
        let (patch, pos, rot) = model.debugVisionPre(pixelValues: g["pixel_values"]!, imageGridTHW: [thw])
        eval(patch); eval(pos); eval(rot)
        func stage(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let mab = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            err("  \(name): cos \(cosine(a, b)) max_abs \(mab) shape \(a.shape) vs \(b.shape)")
        }
        stage("patch", patch, g["patch"]!)
        stage("pos", pos, g["pos"]!)
        stage("rot", rot, g["rot"]!)
        ok = true
    default:
        err("unknown gate: \(gate)")
    }
} catch { err("error: \(error)") }
err(ok ? "PASS" : "FAIL")
exit(ok ? 0 : 1)
