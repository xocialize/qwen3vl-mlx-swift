import MLX
import MLXExactConv
import MLXRandom
import XCTest
@testable import Qwen3VL

/// The vision patch embed is a 3-channel conv3d, the shape mlx 0.32.2 sends through
/// `pad_and_slice_conv_3D_gpu`, which frees its zero fill value while the GPU still reads it
/// (AB-T-0214). `PatchEmbed` routes it through ExactConv instead; its output must still match a
/// plain conv3d on the CPU.
final class PatchEmbedTests: XCTestCase {
    func testPatchEmbedMatchesCPUConv3d() {
        let embed = Qwen3VLVision.PatchEmbed(patchSize: 16, temporalPatchSize: 2, inChannels: 3, hiddenSize: 64)
        // The vision tower's input: flattened (C, T, P, P) patches, one row per patch.
        let x = MLXRandom.uniform(low: -1, high: 1, [37, 3 * 2 * 16 * 16], key: MLXRandom.key(7))
        let y = Device.withDefaultDevice(Device(.gpu)) { embed(x) }
        eval(y)

        let patches = x.reshaped(-1, 3, 2, 16, 16).movedAxis(source: 1, destination: 4)
        XCTAssertTrue(
            ExactConv.takesPadHelper3D(input: patches, weight: embed.proj.weight),
            "a plain Conv3d at this shape would take mlx's pad helper")
        var reference = MLXArray(0)
        Device.withDefaultDevice(Device(.cpu)) {
            reference = conv3d(patches, embed.proj.weight, stride: [2, 16, 16])
            if let bias = embed.proj.bias { reference = reference + bias }
            reference = reference.reshaped(-1, 64)
            eval(reference)
        }
        XCTAssertEqual(y.shape, [37, 64])
        XCTAssertLessThan(abs(y - reference).max().item(Float.self), 1e-4)
    }
}
