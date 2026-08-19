import Foundation
import Testing
@preconcurrency import MLX
@testable import MLXAudioTTS

/// The generation loop carries a running "already emitted" mask instead of
/// recomputing it from the frame history each step. These check the fast path
/// is exactly equivalent to the reference formulation.
struct MossTTSNanoSamplingTests {
    @Test func incrementalSeenMaskMatchesHistoryRecomputation() {
        let vocabSize = 32
        let history = MLXArray([Int32(3), 7, 7, 19, 0]).reshaped(1, 5)
        let logits = MLXArray((0 ..< vocabSize).map { Float($0) - 16.0 }).reshaped(1, vocabSize)

        let fromHistory = MossTTSNanoSampling.applyRepetitionPenalty(
            logits, previousTokenIDs: history, repetitionPenalty: 1.2
        )

        // Accumulate the mask token by token, the way the loop does.
        let vocabRange = MLXArray(Int32(0) ..< Int32(vocabSize))
        var seen: MLXArray?
        for value in [Int32(3), 7, 7, 19, 0] {
            let emitted = MLXArray([value]).reshaped(1, 1) .== vocabRange
            seen = seen.map { $0 .|| emitted } ?? emitted
        }
        let fromMask = MossTTSNanoSampling.applyRepetitionPenalty(
            logits, seenMask: seen, repetitionPenalty: 1.2
        )

        let delta = MLX.abs(fromHistory - fromMask).max().item(Float.self)
        #expect(delta == 0.0, "incremental mask diverged by \(delta)")
    }

    @Test func penaltyPushesRepeatedTokensAway() {
        let vocabSize = 8
        let logits = MLXArray([Float(2.0), -2.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0]).reshaped(1, vocabSize)
        let history = MLXArray([Int32(0), 1]).reshaped(1, 2)
        let penalized = MossTTSNanoSampling.applyRepetitionPenalty(
            logits, previousTokenIDs: history, repetitionPenalty: 2.0
        ).asArray(Float.self)

        // Positive logits are divided, negative multiplied — both reduce rank.
        #expect(penalized[0] == 1.0, "positive repeated logit should be divided")
        #expect(penalized[1] == -4.0, "negative repeated logit should be multiplied")
        #expect(penalized[2] == 0.5, "untouched logit changed")
        #expect(penalized[6] == 3.0, "untouched logit changed")
    }

    @Test func noPenaltyWhenFactorIsOne() {
        let logits = MLXArray([Float(1.0), 2.0, 3.0]).reshaped(1, 3)
        let history = MLXArray([Int32(0)]).reshaped(1, 1)
        let out = MossTTSNanoSampling.applyRepetitionPenalty(
            logits, previousTokenIDs: history, repetitionPenalty: 1.0
        )
        #expect(MLX.abs(out - logits).max().item(Float.self) == 0.0)
    }
}
