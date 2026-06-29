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
guard args.count >= 3 else { err("usage: Qwen3VLGate --t2i|--edit <weightsDir> <fixturesDir>"); exit(2) }
let gate = args[0]
let weights = URL(fileURLWithPath: args[1])
let fixtures = URL(fileURLWithPath: args[2])

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
