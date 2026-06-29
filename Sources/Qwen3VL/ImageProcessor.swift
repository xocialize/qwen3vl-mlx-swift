// Qwen3-VL image preprocessing — numpy/PIL-exact port of mlx_vlm's
// processing_qwen3_vl.Qwen3VLImageProcessor: smart-resize (factor = patch·merge = 32),
// PIL bicubic, rescale+normalize, then patchify to (numPatches, C·tps·ps·ps = 1536).
//
// PILResize is lifted from qwen25vl-mlx-swift (Pillow Resample.c 8bpc path) — VLM
// preprocessing must match the reference resampler exactly, not approximately.

import Foundation
import MLX
import MLXLMCommon

/// PIL-exact bicubic resize on interleaved RGB8 (Pillow Resample.c 8bpc).
public enum PILResize {
    static let precisionBits = 32 - 8 - 2  // 22

    static func bicubic(_ xIn: Double) -> Double {
        let a = -0.5
        let x = abs(xIn)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    static func coefficients(inSize: Int, outSize: Int)
        -> (bounds: [(min: Int, count: Int)], coeffs: [[Int32]])
    {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let one = Double(1 << precisionBits)
        var bounds: [(Int, Int)] = []
        var coeffs: [[Int32]] = []
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            let count = xmax - xmin
            var w = [Double](repeating: 0, count: count)
            var total = 0.0
            for x in 0..<count {
                let v = bicubic((Double(x + xmin) - center + 0.5) / filterscale)
                w[x] = v
                total += v
            }
            var k = [Int32](repeating: 0, count: count)
            for x in 0..<count {
                let normalized = total != 0 ? w[x] / total : w[x]
                let scaled = normalized * one
                k[x] = Int32(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds.append((xmin, count))
            coeffs.append(k)
        }
        return (bounds, coeffs)
    }

    @inline(__always) static func clip8(_ v: Int32) -> UInt8 {
        UInt8(min(max(v >> Int32(precisionBits), 0), 255))
    }

    public static func resize(
        rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int
    ) -> [UInt8] {
        if width == outWidth && height == outHeight { return rgb }  // identity
        let half = Int32(1 << (precisionBits - 1))
        let (hBounds, hCoeffs) = coefficients(inSize: width, outSize: outWidth)
        var temp = [UInt8](repeating: 0, count: height * outWidth * 3)
        rgb.withUnsafeBufferPointer { src in
            temp.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let rowIn = y * width * 3, rowOut = y * outWidth * 3
                    for xx in 0..<outWidth {
                        let (xmin, count) = hBounds[xx]; let k = hCoeffs[xx]
                        var s0 = half, s1 = half, s2 = half
                        for x in 0..<count {
                            let p = rowIn + (xmin + x) * 3; let w = k[x]
                            s0 += Int32(src[p]) * w; s1 += Int32(src[p + 1]) * w; s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + xx * 3
                        dst[o] = clip8(s0); dst[o + 1] = clip8(s1); dst[o + 2] = clip8(s2)
                    }
                }
            }
        }
        let (vBounds, vCoeffs) = coefficients(inSize: height, outSize: outHeight)
        var out = [UInt8](repeating: 0, count: outHeight * outWidth * 3)
        temp.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0..<outHeight {
                    let (ymin, count) = vBounds[yy]; let k = vCoeffs[yy]
                    let rowOut = yy * outWidth * 3
                    for xx in 0..<outWidth {
                        let col = xx * 3
                        var s0 = half, s1 = half, s2 = half
                        for y in 0..<count {
                            let p = (ymin + y) * outWidth * 3 + col; let w = k[y]
                            s0 += Int32(src[p]) * w; s1 += Int32(src[p + 1]) * w; s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + col
                        dst[o] = clip8(s0); dst[o + 1] = clip8(s1); dst[o + 2] = clip8(s2)
                    }
                }
            }
        }
        return out
    }
}

public struct Qwen3VLImageProcessor {
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    public let minPixels: Int
    public let maxPixels: Int
    public let mean: [Float]
    public let std: [Float]

    public init(
        patchSize: Int = 16, mergeSize: Int = 2, temporalPatchSize: Int = 2,
        minPixels: Int = 56 * 56, maxPixels: Int = 14 * 14 * 4 * 1280,
        mean: [Float] = [0.5, 0.5, 0.5], std: [Float] = [0.5, 0.5, 0.5]
    ) {
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalPatchSize = temporalPatchSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.mean = mean
        self.std = std
    }

    /// HF qwen2_vl smart_resize: round each side to `factor`, then beta-scale for min/max.
    public func smartResize(height: Int, width: Int) -> (Int, Int) {
        let factor = patchSize * mergeSize
        var hBar = Int((Double(height) / Double(factor)).rounded()) * factor
        var wBar = Int((Double(width) / Double(factor)).rounded()) * factor
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    /// Interleaved RGB8 -> (pixelValues [numPatches, C·tps·ps²], grid THW).
    public func preprocess(rgb: [UInt8], width: Int, height: Int) -> (MLXArray, THW) {
        let (rh, rw) = smartResize(height: height, width: width)
        let resized = PILResize.resize(rgb: rgb, width: width, height: height, outWidth: rw, outHeight: rh)

        // rescale + normalize into CHW float (x/255 - mean)/std.
        let c = 3, plane = rh * rw
        var chw = [Float](repeating: 0, count: c * plane)
        for i in 0..<plane {
            let p = i * 3
            for ch in 0..<c {
                chw[ch * plane + i] = (Float(resized[p + ch]) / 255 - mean[ch]) / std[ch]
            }
        }

        // patchify (mirror the numpy reshape/transpose).
        let ps = patchSize, tps = temporalPatchSize, ms = mergeSize
        let gh = rh / ps, gw = rw / ps
        var img = MLXArray(chw, [1, 1, c, rh, rw])
        img = repeated(img, count: tps, axis: 1)  // [1, tps, C, rh, rw]
        img = img.reshaped([1, 1, tps, c, gh / ms, ms, ps, gw / ms, ms, ps])
        img = img.transposed(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        let pixelValues = img.reshaped([gh * gw, c * tps * ps * ps])
        return (pixelValues, THW(1, gh, gw))
    }
}
