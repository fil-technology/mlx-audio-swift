import Foundation
import Testing
@preconcurrency import MLX
@testable import MLXAudioTTS

/// Numerical parity for the MOSS-TTS-Nano backbone against the Python
/// `mlx-audio` reference (`mlx_audio/tts/models/moss_tts_nano`).
///
/// Fixtures were produced from `mlx-community/MOSS-TTS-Nano-100M` with the
/// `en_3` bundled voice. The suite skips itself when the weights are not in the
/// local HF cache.
struct MossTTSNanoForwardTests {
    static let text = "Global markets closed higher on Tuesday after the central bank signalled it would hold interest rates steady through the end of the year."
    static let expectedTextColumn: [Int] = [4, 600, 289, 10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 10356, 10425, 3965, 7738, 11492, 505, 587, 10356, 10425, 352, 500, 856, 11492, 505, 587, 10356, 10425, 2128, 1247, 594, 11492, 505, 587, 10356, 10425, 348, 909, 561, 3648, 11492, 505, 587, 10356, 10425, 2818, 1305, 355, 348, 909, 11492, 505, 587, 10356, 10425, 484, 339, 10367, 783, 11492, 505, 587, 10356, 10425, 2427, 980, 11492, 558, 760, 4002, 7139, 10027, 4250, 346, 8756, 1084, 280, 9301, 5792, 1813, 359, 316, 329, 580, 2546, 1198, 6423, 2799, 5190, 917, 280, 1069, 308, 280, 683, 10380, 10356, 14, 5, 4, 8165, 430, 6]
    static let expectedRow40: [Int] = [8, 230, 7, 741, 491, 457, 126, 421, 710, 845, 244, 603, 445, 432, 227, 25, 641]
    static let expectedHiddenFirst32: [Float] = [2.349609, -0.335205, -0.169556, -0.245361, 1.642578, -0.623535, -0.470215, -1.041016, -0.734863, -0.673340, -1.376953, -0.618652, 0.721191, 0.588867, 1.947266, -1.240234, -0.472900, 0.730469, -1.552734, -2.390625, 0.383301, 0.616699, 1.707031, -3.083984, 0.046295, -0.733887, -1.036133, -0.002930, 0.731445, -1.943359, -0.557617, 0.412598]
    static let expectedHiddenAbsMean: Float = 0.741311
    static let expectedHiddenAbsMax: Float = 12.195312
    static let expectedGreedyFirstFrame: [Int] = [245, 230, 340, 124, 69, 1006, 580, 410, 92, 422, 149, 401, 409, 545, 79, 482]

    static func loadModel() async throws -> MossTTSNanoModel? {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return nil
        }
        return try await MossTTSNanoModel.fromPretrained("mlx-community/MOSS-TTS-Nano-100M")
    }

    @Test func promptAssemblyMatchesReference() async throws {
        guard let model = try await Self.loadModel() else { return }
        let codes = try MossVoicePack.bundled().codes(for: "en_3")
        let inputIDs = try model.buildInferenceInputIDs(text: Self.text, promptCodes: codes)

        #expect(inputIDs.dim(0) == 1)
        #expect(inputIDs.dim(1) == Self.expectedTextColumn.count)
        #expect(inputIDs.dim(2) == 17)

        let textColumn = inputIDs[0, 0..., 0].asArray(Int32.self).map(Int.init)
        #expect(textColumn == Self.expectedTextColumn)

        let row40 = inputIDs[0, 40, 0...].asArray(Int32.self).map(Int.init)
        #expect(row40 == Self.expectedRow40)
    }

    @Test func transformerForwardMatchesReference() async throws {
        guard let model = try await Self.loadModel() else { return }
        let codes = try MossVoicePack.bundled().codes(for: "en_3")
        let inputIDs = try model.buildInferenceInputIDs(text: Self.text, promptCodes: codes)

        let embeddings = try model.buildInputEmbeddings(inputIDs)
        let hidden = model.transformer(inputEmbeddings: embeddings, cache: model.transformer.makeCache())
        hidden.eval()

        #expect(hidden.dim(1) == Self.expectedTextColumn.count)
        #expect(hidden.dim(2) == 768)

        let absolute = MLX.abs(hidden.asType(.float32))
        let absMean = absolute.mean().item(Float.self)
        let absMax = absolute.max().item(Float.self)
        #expect(abs(absMean - Self.expectedHiddenAbsMean) < 2e-3,
                "hidden abs-mean \(absMean) vs reference \(Self.expectedHiddenAbsMean)")
        #expect(abs(absMax - Self.expectedHiddenAbsMax) < 5e-2,
                "hidden abs-max \(absMax) vs reference \(Self.expectedHiddenAbsMax)")

        let lastRow = hidden[0, -1, 0..<32].asType(.float32).asArray(Float.self)
        var worst: Float = 0
        for (actual, expected) in zip(lastRow, Self.expectedHiddenFirst32) {
            worst = max(worst, abs(actual - expected))
        }
        #expect(worst < 5e-2, "max|delta| on final hidden row = \(worst)")
    }

    /// Greedy rollout of the first frame's 16 channels. Exact token agreement
    /// is a much sharper check than tensor closeness — a single wrong logit
    /// ordering anywhere in the local transformer changes a code.
    @Test func greedyFirstFrameMatchesReference() async throws {
        guard let model = try await Self.loadModel() else { return }
        let codes = try MossVoicePack.bundled().codes(for: "en_3")
        let inputIDs = try model.buildInferenceInputIDs(text: Self.text, promptCodes: codes)
        let frame = try model.greedyFirstFrameForTesting(promptInputIDs: inputIDs)
        #expect(frame == Self.expectedGreedyFirstFrame,
                "greedy frame \(frame) vs reference \(Self.expectedGreedyFirstFrame)")
    }
}
