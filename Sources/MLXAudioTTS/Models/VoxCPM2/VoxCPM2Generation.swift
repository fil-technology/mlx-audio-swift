import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXAudioCore

// VoxCPM2 generation with the upstream prompt modes (OpenBMB/VoxCPM
// `VoxCPM2Model._generate_with_prompt_cache`), per-call sampling parameters,
// an optional seed, and incremental audio for streaming.

/// Per-call knobs of VoxCPM2. `nil` means the checkpoint's value.
public struct VoxCPM2GenerationParameters: Sendable {
    /// Upper bound on generated patches (one patch is `patchSize` latent
    /// frames, 160 ms at 48 kHz). `nil` uses the checkpoint's `max_length`.
    /// Upstream also bounds every call to `textTokens * badcaseRatio + 10`
    /// patches, which keeps a missed stop from running to the limit.
    public var maxTokens: Int?
    public var minTokens: Int
    /// Diffusion sampling steps of the local DiT (upstream default 10).
    public var inferenceTimesteps: Int?
    /// Classifier-free guidance scale (upstream default 2.0).
    public var cfgValue: Float?
    /// Seeds MLX's global generator before the run so a call reproduces.
    public var seed: UInt64?
    /// Patches decoded together on the stream path: the newest patch plus
    /// this many minus one of left context for the causal decoder.
    public var streamingWindow: Int
    /// Audio-to-text ratio bound, in patches per text token.
    public var badcaseRatio: Float

    public init(
        maxTokens: Int? = nil, minTokens: Int = 2, inferenceTimesteps: Int? = nil,
        cfgValue: Float? = nil, seed: UInt64? = nil, streamingWindow: Int = 4,
        badcaseRatio: Float = 6
    ) {
        self.maxTokens = maxTokens
        self.minTokens = minTokens
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgValue = cfgValue
        self.seed = seed
        self.streamingWindow = max(1, streamingWindow)
        self.badcaseRatio = badcaseRatio
    }
}

/// What the model hears before the target text. Audio is mono at
/// `VoxCPM2Model.sampleRate`.
///
/// - `referenceAudio` alone: *reference* mode, the clip is wrapped in the
///   `ref_audio` tokens and only lends its timbre (upstream "controllable
///   cloning"; a style description goes at the front of the text).
/// - `promptAudio` with `promptText`: *continuation* mode, the model continues
///   the clip as if the target text followed its transcript (upstream
///   "ultimate cloning"; the transcript must be exact).
/// - all three: continuation with an isolated timbre reference as well.
public struct VoxCPM2Prompt: Sendable {
    public var referenceAudio: MaterializedArray?
    public var promptAudio: MaterializedArray?
    public var promptText: String?

    public init(referenceAudio: MaterializedArray? = nil, promptAudio: MaterializedArray? = nil, promptText: String? = nil) {
        self.referenceAudio = referenceAudio
        self.promptAudio = promptAudio
        self.promptText = promptText
    }

    public var isEmpty: Bool { referenceAudio == nil && promptAudio == nil }

    enum Mode { case zeroShot, reference, continuation, referenceContinuation }
    var mode: Mode {
        switch (referenceAudio != nil, promptAudio != nil) {
        case (false, false): return .zeroShot
        case (true, false): return .reference
        case (false, true): return .continuation
        case (true, true): return .referenceContinuation
        }
    }
}

extension VoxCPM2Model {

    /// Collapses whitespace the way upstream does before tokenising.
    static func normalizeText(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
    }

    /// Encodes a clip to latent patches `(T, P, D)`; `leftPad` aligns the clip
    /// to a patch boundary from the left (upstream pads prompts left and
    /// references right).
    func encodeWav(_ audio: MLXArray, leftPad: Bool) throws -> MLXArray {
        guard leftPad else { return try encodeWav(audio) }
        var samples = audio.ndim == 1 ? audio : audio.flattened()
        if sampleRate != audioVAE.encodeSampleRate {
            samples = MLXArray(try resampleAudio(samples.asArray(Float.self), from: sampleRate, to: audioVAE.encodeSampleRate))
        }
        let patchLen = patchSize * audioVAE.chunkSize
        let remainder = samples.dim(0) % patchLen
        if remainder != 0 {
            samples = MLX.padded(samples, widths: [IntOrPair((patchLen - remainder, 0))])
        }
        // Already aligned: the generic path resamples again only if rates differ,
        // which they no longer do once we pass encoder-rate samples through the
        // VAE directly.
        var inp = samples.expandedDimensions(axes: [0, 1]) // (1, 1, T)
        inp = inp.transposed(0, 2, 1) // (1, T, 1)
        let feat = audioVAE.encode(inp, sampleRate: audioVAE.encodeSampleRate).squeezed(axis: 0)
        let numPatches = feat.dim(0) / patchSize
        return feat[..<(numPatches * patchSize)].reshaped(numPatches, patchSize, -1)
    }

    struct PromptSequence {
        var textToken: MLXArray
        var audioFeat: MLXArray
        var textMask: MLXArray
        var audioMask: MLXArray
        /// Prompt patches the stream path may use as left context.
        var contextPatches: [MLXArray]
        var targetTokenCount: Int
    }

    /// Assembles the model input for the four upstream modes.
    func makeSequence(targetText: String, prompt: VoxCPM2Prompt, streamingWindow: Int) throws -> PromptSequence {
        let latentDim = audioVAE.latentDim
        let target = Self.normalizeText(targetText)
        guard !target.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw AudioGenerationError.invalidInput("target text must be a non-empty string")
        }
        let mode = prompt.mode
        if (prompt.promptAudio == nil) != (prompt.promptText == nil) {
            throw AudioGenerationError.invalidInput("VoxCPM2 continuation needs both the prompt clip and its transcript")
        }

        let fullText: String
        switch mode {
        case .continuation, .referenceContinuation:
            fullText = Self.normalizeText(prompt.promptText ?? "") + target
        case .zeroShot, .reference:
            fullText = target
        }
        let textIds = try tokenize(fullText) + [Int(audioStartToken)]
        let targetTokenCount = try tokenize(target).count
        let tokenArray = MLXArray(textIds.map { Int32($0) })
        let textLength = tokenArray.dim(0)
        let textPadFeat = MLXArray.zeros([textLength, patchSize, latentDim])

        var contextPatches: [MLXArray] = []
        func context(from promptFeat: MLXArray) -> [MLXArray] {
            let count = min(max(streamingWindow - 1, 0), promptFeat.dim(0))
            guard count > 0 else { return [] }
            return (0 ..< count).map { i in
                promptFeat[(promptFeat.dim(0) - count + i) ..< (promptFeat.dim(0) - count + i + 1)]
                    .reshaped(1, 1, patchSize, latentDim)
            }
        }

        switch mode {
        case .zeroShot:
            return .init(textToken: tokenArray, audioFeat: textPadFeat,
                         textMask: MLXArray.ones([textLength]), audioMask: MLXArray.zeros([textLength]),
                         contextPatches: [], targetTokenCount: targetTokenCount)
        case .continuation:
            let promptFeat = try encodeWav(prompt.promptAudio!, leftPad: true)
            let audioLength = promptFeat.dim(0)
            contextPatches = context(from: promptFeat)
            return .init(
                textToken: MLX.concatenated([tokenArray, MLXArray.zeros([audioLength]).asType(.int32)]),
                audioFeat: MLX.concatenated([textPadFeat, promptFeat], axis: 0),
                textMask: MLX.concatenated([MLXArray.ones([textLength]), MLXArray.zeros([audioLength])]),
                audioMask: MLX.concatenated([MLXArray.zeros([textLength]), MLXArray.ones([audioLength])]),
                contextPatches: contextPatches, targetTokenCount: targetTokenCount)
        case .reference:
            let refFeat = try encodeWav(prompt.referenceAudio!, leftPad: false)
            let (refTokens, refFeats, refTMask, refAMask) = makeRefPrefix(refFeat)
            return .init(
                textToken: MLX.concatenated([refTokens, tokenArray]),
                audioFeat: MLX.concatenated([refFeats, textPadFeat], axis: 0),
                textMask: MLX.concatenated([refTMask, MLXArray.ones([textLength])]),
                audioMask: MLX.concatenated([refAMask, MLXArray.zeros([textLength])]),
                contextPatches: [], targetTokenCount: targetTokenCount)
        case .referenceContinuation:
            let refFeat = try encodeWav(prompt.referenceAudio!, leftPad: false)
            let promptFeat = try encodeWav(prompt.promptAudio!, leftPad: true)
            let audioLength = promptFeat.dim(0)
            let (refTokens, refFeats, refTMask, refAMask) = makeRefPrefix(refFeat)
            contextPatches = context(from: promptFeat)
            return .init(
                textToken: MLX.concatenated([refTokens, tokenArray, MLXArray.zeros([audioLength]).asType(.int32)]),
                audioFeat: MLX.concatenated([refFeats, textPadFeat, promptFeat], axis: 0),
                textMask: MLX.concatenated([refTMask, MLXArray.ones([textLength]), MLXArray.zeros([audioLength])]),
                audioMask: MLX.concatenated([refAMask, MLXArray.zeros([textLength]), MLXArray.ones([audioLength])]),
                contextPatches: contextPatches, targetTokenCount: targetTokenCount)
        }
    }

    /// Runs the autoregressive loop and hands every new latent patch
    /// `(1, 1, P, D)` to `onPatch`. Returns the patches generated.
    func runGeneration(
        text: String,
        prompt: VoxCPM2Prompt,
        parameters: VoxCPM2GenerationParameters,
        onContext: (([MLXArray]) -> Void)? = nil,
        onPatch: ((MLXArray) throws -> Void)? = nil
    ) async throws -> [MLXArray] {
        let sequence = try makeSequence(targetText: text, prompt: prompt, streamingWindow: parameters.streamingWindow)
        onContext?(sequence.contextPatches)
        let inferenceTimesteps = max(1, parameters.inferenceTimesteps ?? config.inferenceTimesteps)
        let cfgValue = parameters.cfgValue ?? config.cfgScale
        let ratioBound = Int(Float(sequence.targetTokenCount) * parameters.badcaseRatio) + 10
        let maxTokens = max(parameters.minTokens + 1, min(parameters.maxTokens ?? config.maxLength, ratioBound))
        let minTokens = parameters.minTokens
        if let seed = parameters.seed { MLXRandom.seed(seed) }

        let scaleEmb = Float(config.lmConfig.useMup ? config.lmConfig.scaleEmb : 1)

        let bTextToken = sequence.textToken.expandedDimensions(axis: 0)
        let bAudioFeat = sequence.audioFeat.expandedDimensions(axis: 0)
        let bTextMask = sequence.textMask.expandedDimensions(axis: 0)
        let bAudioMask = sequence.audioMask.expandedDimensions(axis: 0)

        var featEmbed = featEncoder(bAudioFeat)
        featEmbed = encToLmProj(featEmbed)
        let textEmbed = baseLM.embedTokens!(bTextToken) * scaleEmb
        let combinedEmbed = bTextMask[0..., 0..., .newAxis] * textEmbed
            + bAudioMask[0..., 0..., .newAxis] * featEmbed

        var prefixFeatCond = bAudioFeat[0..., -1, 0..., 0...] // (1, P, D)

        var (encOutputs, lmCache) = baseLM(inputsEmbeds: combinedEmbed)
        encOutputs = fsqLayer(encOutputs) * bAudioMask[0..., 0..., .newAxis]
            + encOutputs * bTextMask[0..., 0..., .newAxis]
        var lmHidden = encOutputs[0..., -1, 0...]

        let residualInput = fusionConcatProj(
            MLX.concatenated([encOutputs, bAudioMask[0..., 0..., .newAxis] * featEmbed], axis: -1)
        )
        var (residualOutputs, resCache) = residualLM(inputsEmbeds: residualInput)
        var residualHidden = residualOutputs[0..., -1, 0...]

        var predFeatSeq: [MLXArray] = []
        predFeatSeq.reserveCapacity(maxTokens)

        for i in 0 ..< maxTokens {
            try Task.checkCancellation()
            let ditH = MLX.concatenated([lmToDitProj(lmHidden), resToDitProj(residualHidden)], axis: -1)
            let condIn = prefixFeatCond.transposed(0, 2, 1) // (B, D, P)

            var predFeat = featDecoder.sample(
                mu: ditH, nTimesteps: inferenceTimesteps, patchSize: patchSize,
                cond: condIn, cfgValue: cfgValue
            )
            predFeat = predFeat.transposed(0, 2, 1) // (B, P, D)
            let patch = predFeat[0..., .newAxis, 0..., 0...] // (B, 1, P, D)
            predFeatSeq.append(patch)

            let currEmbedProj = encToLmProj(featEncoder(patch))

            let stopLogits = stopHead(silu(stopProj(lmHidden)))
            let stopFlag = MLX.argMax(stopLogits, axis: -1).item(Int.self)
            eval(predFeat)
            try onPatch?(patch)
            if i > minTokens && stopFlag == 1 { break }

            let newLmOut: MLXArray
            (newLmOut, lmCache) = baseLM(inputsEmbeds: currEmbedProj, cache: lmCache)
            lmHidden = fsqLayer(newLmOut[0..., -1, 0...])

            let currResInput = fusionConcatProj(
                MLX.concatenated([lmHidden[0..., .newAxis, 0...], currEmbedProj], axis: -1)
            )
            let newResOut: MLXArray
            (newResOut, resCache) = residualLM(inputsEmbeds: currResInput, cache: resCache)
            residualHidden = newResOut[0..., -1, 0...]
            prefixFeatCond = predFeat
            eval(lmHidden, residualHidden)
        }
        guard !predFeatSeq.isEmpty else {
            throw AudioGenerationError.invalidInput("Model generated no audio patches")
        }
        return predFeatSeq
    }

    /// Decodes patches `(1, T, P, D)` to mono samples at `sampleRate`.
    func decodePatches(_ patches: [MLXArray]) -> MLXArray {
        let all = MLX.concatenated(patches, axis: 1)
        let flat = all.reshaped(all.dim(0), -1, featDim)
        let audio = audioVAE.decode(flat).flattened()
        eval(audio)
        return audio
    }

    /// Samples one patch produces at the output rate.
    var samplesPerPatch: Int { patchSize * audioVAE.decodeChunkSize }

    /// Whole-take generation with explicit prompt and parameters.
    public func generate(
        text: String,
        prompt: VoxCPM2Prompt,
        parameters: VoxCPM2GenerationParameters
    ) async throws -> MaterializedArray {
        let patches = try await runGeneration(text: text, prompt: prompt, parameters: parameters)
        return decodePatches(patches).materialized()
    }

    /// Streams audio patch by patch (160 ms at 48 kHz per event). Each new
    /// patch is decoded together with the last `streamingWindow - 1` patches
    /// (prompt patches at the start of a continuation) and only its own
    /// samples are emitted, so the causal decoder sees context and the output
    /// has no overlap to crossfade. The final event is `.info`.
    public func generateStream(
        text: String,
        prompt: VoxCPM2Prompt,
        parameters: VoxCPM2GenerationParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                var firstAudio: TimeInterval?
                var emitted = 0
                let window = parameters.streamingWindow
                let perPatch = self.samplesPerPatch
                var recent: [MLXArray] = []
                _ = try await self.runGeneration(text: text, prompt: prompt, parameters: parameters, onContext: { recent = $0 }) { patch in
                    recent.append(patch)
                    if recent.count > window { recent.removeFirst(recent.count - window) }
                    let audio = self.decodePatches(recent)
                    let total = audio.dim(0)
                    let chunk = audio[max(0, total - perPatch)...]
                    eval(chunk)
                    if firstAudio == nil { firstAudio = Date().timeIntervalSince(startTime) }
                    emitted += chunk.dim(0)
                    continuation.yield(.audio(chunk.materialized()))
                }
                let generateTime = Date().timeIntervalSince(startTime)
                continuation.yield(.info(AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: emitted,
                    prefillTime: firstAudio ?? generateTime,
                    generateTime: generateTime,
                    tokensPerSecond: Double(emitted) / max(generateTime, 0.001),
                    peakMemoryUsage: 0
                )))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}
