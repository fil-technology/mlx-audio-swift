import Foundation
import Testing
@preconcurrency import MLX
@testable import MLXAudioTTS

/// Parity for the MOSS-Audio-Tokenizer-Nano decode path against the Python
/// `mlx_audio.codec.MossAudioTokenizer` reference, decoding the `en_3` voice
/// codes shipped in the bundled voice pack.
struct MossAudioTokenizerTests {
    static let expectedSamples = 226560
    static let expectedChannels = 2
    static let expectedAbsMean: Float = 0.058494430
    static let expectedAbsMax: Float = 0.515329540
    static let expectedRMSChannel0: Float = 0.100649700
    static let expectedFirst16: [Float] = [-0.000194583, -0.000258387, -0.000347259, -0.000301253, -0.000396699, -0.000389186, -0.000451766, -0.000436864, -0.000299173, -0.000375579, -0.000365294, -0.000203828, -0.000224753, -0.000366761, -0.000393677, -0.000337255]

    @Test func decodeMatchesReference() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }
        let codec = try await MossAudioTokenizerModel.fromPretrained()
        let codes = try MossVoicePack.bundled().codes(for: "en_3")

        let audio = try codec.decodeAudioCodes(codes, numQuantizers: 16)
        audio.eval()

        #expect(audio.dim(0) == Self.expectedSamples, "sample count \(audio.dim(0))")
        #expect(audio.dim(1) == Self.expectedChannels, "channel count \(audio.dim(1))")

        let absolute = MLX.abs(audio.asType(.float32))
        let absMean = absolute.mean().item(Float.self)
        let absMax = absolute.max().item(Float.self)
        #expect(abs(absMean - Self.expectedAbsMean) < 1e-3, "abs-mean \(absMean) vs \(Self.expectedAbsMean)")
        #expect(abs(absMax - Self.expectedAbsMax) < 5e-3, "abs-max \(absMax) vs \(Self.expectedAbsMax)")

        let channel0 = audio[0..., 0].asType(.float32)
        let rms = MLX.sqrt((channel0 * channel0).mean()).item(Float.self)
        #expect(abs(rms - Self.expectedRMSChannel0) < 1e-3, "rms \(rms) vs \(Self.expectedRMSChannel0)")

        let first16 = channel0[0 ..< 16].asArray(Float.self)
        var worst: Float = 0
        for (actual, expected) in zip(first16, Self.expectedFirst16) {
            worst = max(worst, abs(actual - expected))
        }
        #expect(worst < 1e-4, "max|delta| over first 16 samples = \(worst)")
    }
}
