import Foundation
@preconcurrency import MLX
import MLXNN

/// Token sampling for MOSS-TTS-Nano.
///
/// Ported from `mlx_audio/tts/models/moss_tts_nano/sampling.py` plus the
/// `apply_top_k` / `apply_top_p` helpers in `mlx_audio.lm.sample_utils`
/// (Apache-2.0).
///
/// Top-k and top-p filter *log-probabilities* and the resulting mask is then
/// transferred back onto the raw logits. Filtering the logits directly would
/// not be equivalent once a temperature has been applied, so the two-step dance
/// is deliberate.
enum MossTTSNanoSampling {

    private static func maskLogits(_ logits: MLXArray, from logProbs: MLXArray) -> MLXArray {
        MLX.where(logProbs .== -Float.infinity, MLXArray(-Float.infinity), logits)
    }

    /// HF-style repetition penalty: negative scores are multiplied and positive
    /// scores divided, so a repeated token moves away from being chosen either
    /// way.
    static func applyRepetitionPenalty(
        _ logits: MLXArray,
        previousTokenIDs: MLXArray?,
        repetitionPenalty: Float
    ) -> MLXArray {
        guard let previousTokenIDs, repetitionPenalty != 1.0, previousTokenIDs.size > 0 else {
            return logits
        }
        let vocabSize = logits.dim(-1)
        var previous = previousTokenIDs.asType(.int32)
        previous = MLX.where(previous .< 0, MLXArray(Int32(0)), previous)
        previous = MLX.where(previous .>= MLXArray(Int32(vocabSize)), MLXArray(Int32(0)), previous)

        // Build the "already emitted" mask by broadcasting a comparison rather
        // than materialising a vocab x vocab identity matrix.
        let vocabRange = MLXArray(Int32(0) ..< Int32(vocabSize))
        let seen = MLX.any(previous[.ellipsis, .newAxis] .== vocabRange, axis: -2)
        return applyRepetitionPenalty(logits, seenMask: seen, repetitionPenalty: repetitionPenalty)
    }

    /// Same penalty, but driven by a pre-accumulated `[1, vocab]` boolean mask.
    ///
    /// Recomputing "which ids has this channel already emitted?" from the full
    /// frame history costs O(frames) per channel per frame, i.e. quadratic over
    /// an utterance. Carrying the mask forward and OR-ing in each new token is
    /// exactly equivalent and O(1).
    static func applyRepetitionPenalty(
        _ logits: MLXArray,
        seenMask: MLXArray?,
        repetitionPenalty: Float
    ) -> MLXArray {
        guard let seenMask, repetitionPenalty != 1.0 else { return logits }
        let penalized = MLX.where(
            logits .< 0,
            logits * repetitionPenalty,
            logits / repetitionPenalty
        )
        return MLX.where(seenMask, penalized, logits)
    }

    static func applyTopK(_ logits: MLXArray, topK: Int?) -> MLXArray {
        guard let topK, topK > 0, topK < logits.dim(-1) else { return logits }
        let logProbs = MLXNN.logSoftMax(logits, axis: -1)
        // Everything ranked below the k-th largest log-prob is masked out.
        let maskIndices = MLX.argPartition(-logProbs, kth: topK - 1, axis: -1)[.ellipsis, topK...]
        let masked = MLX.putAlong(
            logProbs, maskIndices, values: MLXArray(-Float.infinity).asType(logProbs.dtype), axis: -1
        )
        return maskLogits(logits, from: masked)
    }

    static func applyTopP(_ logits: MLXArray, topP: Float?) -> MLXArray {
        guard let topP, topP > 0.0, topP < 1.0 else { return logits }
        let logProbs = MLXNN.logSoftMax(logits, axis: -1)
        let probs = MLX.exp(logProbs)

        let sortedIndices = MLX.argSort(logProbs, axis: -1)  // ascending
        let sortedProbs = MLX.takeAlong(probs, sortedIndices, axis: -1)
        var cumulative = MLX.cumsum(sortedProbs, axis: -1)

        // Scatter ranks back so `cumulative` lines up with the original order.
        let ranks = MLXArray(Int32(0) ..< Int32(sortedIndices.dim(-1))).asType(sortedIndices.dtype)
        let inverseIndices = MLX.putAlong(
            MLX.zeros(like: sortedIndices), sortedIndices, values: ranks, axis: -1
        )
        cumulative = MLX.takeAlong(cumulative, inverseIndices, axis: -1)

        let filtered = MLX.where(cumulative .> (1.0 - topP), logProbs, MLXArray(-Float.infinity))
        return maskLogits(logits, from: filtered)
    }

    static func sampleNextToken(
        _ logits: MLXArray,
        doSample: Bool,
        temperature: Float = 1.0,
        topK: Int? = nil,
        topP: Float? = nil,
        previousTokenIDs: MLXArray? = nil,
        seenMask: MLXArray? = nil,
        repetitionPenalty: Float = 1.0
    ) -> MLXArray {
        var scores = seenMask != nil
            ? applyRepetitionPenalty(logits, seenMask: seenMask, repetitionPenalty: repetitionPenalty)
            : applyRepetitionPenalty(
                logits, previousTokenIDs: previousTokenIDs, repetitionPenalty: repetitionPenalty
            )
        guard doSample else {
            return MLX.argMax(scores, axis: -1).asType(.int32)
        }
        precondition(temperature > 0, "temperature must be positive when sampling")
        scores = scores / temperature
        scores = applyTopK(scores, topK: topK)
        scores = applyTopP(scores, topP: topP)
        return categorical(scores, axis: -1).asType(.int32)
    }

    /// The assistant may only say "another audio frame follows" or "audio
    /// ends", so the text head is restricted to those two ids before sampling.
    static func sampleAssistantTextToken(
        _ textLogits: MLXArray,
        audioAssistantSlotTokenID: Int,
        audioEndTokenID: Int,
        doSample: Bool,
        temperature: Float,
        topK: Int,
        topP: Float
    ) -> MLXArray {
        let candidateIDs = MLXArray([Int32(audioAssistantSlotTokenID), Int32(audioEndTokenID)])
        let candidateScores = textLogits[.ellipsis, candidateIDs]
        let sampled = MossTTSNanoSampling.sampleNextToken(
            candidateScores,
            doSample: doSample,
            temperature: temperature,
            topK: min(topK, 2),
            topP: topP
        )
        return candidateIDs[sampled]
    }
}
