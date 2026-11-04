import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXAudioCore
import MLXNN

public final class MossTTSNanoModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MossTTSNanoConfig

    @ModuleInfo(key: "transformer") var transformer: MossGPT2Model
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: [Embedding]
    @ModuleInfo(key: "local_transformer") var localTransformer: MossGPT2Model

    public var tokenizer: MossTextTokenizing?
    public var audioTokenizer: MossAudioTokenizing?
    private var hfToken: String?
    private var cache: HubCache = .default

    // MARK: Fork additions (fil-technology/mlx-audio-swift)

    /// Text-token budget per generated chunk. The reference uses 75.
    public var maxTextTokensPerChunk: Int = 75

    /// Codebooks predicted per frame (1...n_vq). `nil` uses all of them.
    ///
    /// MOSS predicts each codebook sequentially through the local transformer,
    /// so this is close to a linear dial on generation time: the residual
    /// quantiser is ordered coarse-to-fine, and dropping the finest levels costs
    /// high-frequency detail rather than intelligibility. Measured 2.4x faster
    /// per word at 8 of 16.
    public var codebookCount: Int?

    private var voicePack: MossVoicePack?

    /// Voice used when a caller passes neither `voice` nor `refAudio`.
    ///
    /// MOSS has no speaker embeddings, so *some* reference is always required.
    /// `en_news` is a newsreader read, steadier than the conversational clips.
    public static let defaultVoice = "en_news"

    /// Names accepted as `voice` without reference audio.
    public var availableVoices: [String] { voicePack?.names ?? [] }

    /// Prompt codes for a bundled voice name.
    func promptCodes(forVoice voice: String?) throws -> MLXArray {
        guard let voicePack else {
            throw MossTTSNanoError.unknownVoice("bundled voice pack failed to load")
        }
        let requested = (voice?.isEmpty == false) ? voice! : Self.defaultVoice
        return try voicePack.codes(for: requested)
    }

    /// The codec decodes to `[frames, channels]`; `SpeechGenerationModel`
    /// callers expect a flat mono waveform.
    static func monoDownmix(_ audio: MLXArray) -> MLXArray {
        guard audio.ndim == 2, audio.dim(1) > 1 else {
            return audio.ndim == 2 ? audio[0..., 0] : audio
        }
        return audio.mean(axis: -1)
    }

    public var sampleRate: Int { config.audioTokenizerSampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 375,
            temperature: 0.7,
            topP: 0.9,
            topK: 50,
            repetitionPenalty: 1.1
        )
    }

    public init(config: MossTTSNanoConfig) {
        self.config = config
        _transformer.wrappedValue = MossGPT2Model(config: config.gpt2Config, useTokenEmbedding: true)
        _audioEmbeddings.wrappedValue = config.audioCodebookSizes.map {
            Embedding(embeddingCount: $0, dimensions: config.gpt2Config.nEmbd)
        }
        _localTransformer.wrappedValue = MossGPT2Model(config: config.localGPT2Config(), useTokenEmbedding: false)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            if key == "text_lm_head.weight" { return false }
            if key.hasPrefix("audio_lm_heads.") { return false }
            if key == "local_transformer.wte.weight" { return false }
            if key.hasPrefix("transformer.wpe."), transformer.wpe == nil { return false }
            if key.hasPrefix("local_transformer.wpe."), localTransformer.wpe == nil { return false }
            return true
        }
    }

    public func encodeReferenceAudio(
        _ refAudio: MLXArray,
        numQuantizers: Int? = nil
    ) throws -> MLXArray {
        guard let audioTokenizer else {
            throw MossTTSNanoError.audioTokenizerNotInitialized
        }
        return try audioTokenizer.encodeAudio(
            refAudio,
            numQuantizers: numQuantizers ?? config.nVQ
        )
    }

    public func decodeAudioTokenIDs(
        _ audioTokenIDs: MLXArray,
        numQuantizers: Int? = nil
    ) throws -> MLXArray {
        guard let audioTokenizer else {
            throw MossTTSNanoError.audioTokenizerNotInitialized
        }
        return try audioTokenizer.decodeAudioCodes(
            audioTokenIDs,
            numQuantizers: numQuantizers ?? config.nVQ
        )
    }

    private func ensureAudioTokenizer() async throws {
        if audioTokenizer != nil { return }
        if let modelPath = config.modelPath {
            let audioTokenizerDirectory = URL(fileURLWithPath: modelPath)
                .appendingPathComponent("audio_tokenizer", isDirectory: true)
            if FileManager.default.fileExists(
                atPath: audioTokenizerDirectory.appendingPathComponent("config.json").path
            ) {
                audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(audioTokenizerDirectory)
                return
            }
        }
        let source = resolvedAudioTokenizerSource()
        audioTokenizer = try await MLXMossAudioTokenizer.fromPretrained(
            source,
            hfToken: hfToken,
            cache: cache
        )
    }

    private func resolvedAudioTokenizerSource() -> String {
        guard let source = config.audioTokenizerPretrainedNameOrPath?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !source.isEmpty
        else {
            return mossDefaultAudioTokenizerRepo
        }

        if source.lowercased() == "openmoss-team/moss-audio-tokenizer-nano" {
            return mossDefaultAudioTokenizerRepo
        }
        return source
    }

    func textLMHead(_ hiddenStates: MLXArray) throws -> MLXArray {
        try matmul(hiddenStates, transformer.tokenEmbeddingWeight.transposed(1, 0))
    }

    func audioLMHead(_ hiddenStates: MLXArray, channelIndex: Int) -> MLXArray {
        matmul(hiddenStates, audioEmbeddings[channelIndex].weight.transposed(1, 0))
    }

    public func buildInputsEmbeds(_ inputIDs: MLXArray) throws -> MLXArray {
        guard inputIDs.ndim == 3, inputIDs.dim(2) == config.nVQ + 1 else {
            throw MossTTSNanoError.invalidInput(
                "Expected inputIDs shape [batch, seq, \(config.nVQ + 1)], got \(inputIDs.shape)"
            )
        }

        let textIDs = inputIDs[0..., 0..., 0]
        var inputsEmbeds = try transformer.tokenEmbedding(textIDs)
        for (channelIndex, embedding) in audioEmbeddings.enumerated() {
            let channelIDs = inputIDs[0..., 0..., channelIndex + 1]
            let validMask = channelIDs .!= MLXArray(Int32(config.audioPadTokenID))
            let safeIDs = MLX.where(validMask, channelIDs, MLXArray(Int32(0)))
            let audioEmbeds = embedding(safeIDs)
            inputsEmbeds = inputsEmbeds + audioEmbeds * validMask.expandedDimensions(axis: -1).asType(inputsEmbeds.dtype)
        }
        return inputsEmbeds
    }

    public func buildTextRows(_ tokenIDs: [Int]) -> MLXArray {
        let rowWidth = config.nVQ + 1
        guard !tokenIDs.isEmpty else {
            return MLXArray.zeros([0, rowWidth], type: Int32.self)
        }

        var rows = Array(repeating: Int32(config.audioPadTokenID), count: tokenIDs.count * rowWidth)
        for (rowIndex, tokenID) in tokenIDs.enumerated() {
            rows[rowIndex * rowWidth] = Int32(tokenID)
        }
        return MLXArray(rows, [tokenIDs.count, rowWidth]).asType(.int32)
    }

    public func buildAudioPrefixRows(promptAudioCodes: MLXArray, slotTokenID: Int) throws -> MLXArray {
        guard promptAudioCodes.ndim == 2 else {
            throw MossTTSNanoError.invalidInput(
                "promptAudioCodes must have shape [frames, n_vq], got \(promptAudioCodes.shape)"
            )
        }

        let frameCount = promptAudioCodes.dim(0)
        let sourceChannels = promptAudioCodes.dim(1)
        let rowWidth = config.nVQ + 1
        let copyChannels = min(sourceChannels, config.nVQ)
        let codes = promptAudioCodes.asType(.int32).asArray(Int32.self)

        var rows = Array(repeating: Int32(config.audioPadTokenID), count: frameCount * rowWidth)
        for frameIndex in 0 ..< frameCount {
            rows[frameIndex * rowWidth] = Int32(slotTokenID)
            for channel in 0 ..< copyChannels {
                rows[frameIndex * rowWidth + 1 + channel] = codes[frameIndex * sourceChannels + channel]
            }
        }
        return MLXArray(rows, [frameCount, rowWidth]).asType(.int32)
    }

    public func buildInferenceInputIDs(
        text: String,
        tokenizer: MossTextTokenizing,
        mode: String = "voice_clone",
        promptText: String? = nil,
        promptAudioCodes: MLXArray? = nil
    ) throws -> (inputIDs: MLXArray, attentionMask: MLXArray) {
        let normalizedMode = mode.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard normalizedMode == "voice_clone" || normalizedMode == "continuation" else {
            throw MossTTSNanoError.invalidInput("mode must be either 'voice_clone' or 'continuation'")
        }

        let sections: [MLXArray]
        if normalizedMode == "voice_clone" {
            guard let promptAudioCodes else {
                throw MossTTSNanoError.invalidInput("voice_clone mode requires promptAudioCodes")
            }
            if promptText != nil {
                throw MossTTSNanoError.invalidInput("voice_clone mode does not accept promptText")
            }
            let textTokenIDs = mossEncodeText(tokenizer, text)
            let prefixTokenIDs = mossBuildUserPromptPrefix(tokenizer: tokenizer, config: config)
                + [config.audioStartTokenID]
            let suffixTokenIDs = [config.audioEndTokenID]
                + mossBuildUserPromptAfterReference(tokenizer: tokenizer)
                + textTokenIDs
                + mossBuildAssistantPromptPrefix(tokenizer: tokenizer, config: config)
                + [config.audioStartTokenID]
            sections = [
                buildTextRows(prefixTokenIDs),
                try buildAudioPrefixRows(
                    promptAudioCodes: promptAudioCodes,
                    slotTokenID: config.audioUserSlotTokenID
                ),
                buildTextRows(suffixTokenIDs),
            ]
        } else {
            if (promptText == nil) != (promptAudioCodes == nil) {
                throw MossTTSNanoError.invalidInput(
                    "continuation mode accepts target text only, or both promptText and promptAudioCodes"
                )
            }
            let effectiveText = promptText.map { $0 + text } ?? text
            let promptTokenIDs = mossBuildPromptTokenIDs(
                tokenizer: tokenizer,
                config: config,
                textTokenIDs: mossEncodeText(tokenizer, effectiveText)
            )
            var continuationSections = [
                buildTextRows(promptTokenIDs),
                buildTextRows([config.audioStartTokenID]),
            ]
            if let promptAudioCodes {
                continuationSections.append(
                    try buildAudioPrefixRows(
                        promptAudioCodes: promptAudioCodes,
                        slotTokenID: config.audioAssistantSlotTokenID
                    )
                )
            }
            sections = continuationSections
        }

        let rows = MLX.concatenated(sections, axis: 0)
        let inputIDs = rows.expandedDimensions(axis: 0)
        let attentionMask = MLXArray.ones([1, rows.dim(0)], dtype: .bool)
        return (inputIDs, attentionMask)
    }

    public func leftPadInferenceBatch(
        inputIDBatches: [MLXArray],
        attentionMaskBatches: [MLXArray]
    ) throws -> (inputIDs: MLXArray, attentionMask: MLXArray) {
        guard !inputIDBatches.isEmpty else {
            throw MossTTSNanoError.invalidInput("inputIDBatches must not be empty")
        }
        guard inputIDBatches.count == attentionMaskBatches.count else {
            throw MossTTSNanoError.invalidInput("inputIDBatches and attentionMaskBatches must have the same count")
        }

        let batchSize = inputIDBatches.count
        let maxSeqLen = inputIDBatches.map { $0.dim(1) }.max() ?? 0
        let rowWidth = config.nVQ + 1
        var flatIDs: [Int32] = []
        var flatMask: [Bool] = []
        flatIDs.reserveCapacity(batchSize * maxSeqLen * rowWidth)
        flatMask.reserveCapacity(batchSize * maxSeqLen)

        for (inputIDs, attentionMask) in zip(inputIDBatches, attentionMaskBatches) {
            let seqLen = inputIDs.dim(1)
            let padLen = maxSeqLen - seqLen
            for _ in 0 ..< padLen {
                flatIDs.append(Int32(config.padTokenID))
                flatIDs.append(contentsOf: Array(repeating: Int32(config.audioPadTokenID), count: config.nVQ))
                flatMask.append(false)
            }
            flatIDs.append(contentsOf: inputIDs.asType(.int32).asArray(Int32.self))
            flatMask.append(contentsOf: attentionMask.asType(.bool).asArray(Bool.self))
        }

        return (
            MLXArray(flatIDs, [batchSize, maxSeqLen, rowWidth]).asType(.int32),
            MLXArray(flatMask, [batchSize, maxSeqLen]).asType(.bool)
        )
    }

    public func resolveNQ(_ nq: Int?) throws -> Int {
        guard let nq else { return config.nVQ }
        guard nq >= 1, nq <= config.nVQ else {
            throw MossTTSNanoError.invalidInput("nq must be in [1, \(config.nVQ)], got \(nq)")
        }
        return nq
    }

    public func generateAudioTokenIDs(
        promptInputIDs: MLXArray,
        attentionMask initialAttentionMask: MLXArray? = nil,
        nq: Int? = nil,
        maxNewFrames: Int = 375,
        doSample: Bool = true,
        textTemperature: Float = 1.0,
        textTopP: Float = 1.0,
        textTopK: Int = 50,
        audioTemperature: Float = 0.8,
        audioTopP: Float = 0.95,
        audioTopK: Int = 25,
        audioRepetitionPenalty: Float = 1.2,
        useKVCache: Bool = true
    ) throws -> MLXArray {
        var promptInputIDs = promptInputIDs
        if promptInputIDs.ndim == 2 {
            promptInputIDs = promptInputIDs.expandedDimensions(axis: 0)
        }
        guard promptInputIDs.ndim == 3 else {
            throw MossTTSNanoError.invalidInput(
                "Expected promptInputIDs with 3 dimensions, got \(promptInputIDs.shape)"
            )
        }
        guard promptInputIDs.dim(0) == 1 else {
            throw MossTTSNanoError.notImplemented(
                "Batched MOSS-TTS-Nano token generation is not implemented yet."
            )
        }

        let effectiveNQ = try resolveNQ(nq)
        let cache = useKVCache ? transformer.makeCache() : nil
        var currentModelInputIDs = promptInputIDs
        var currentAttentionMask = (initialAttentionMask ?? MLXArray.ones(promptInputIDs.shape.dropLast(), dtype: .bool)).asType(.bool)
        var generatedFrames: [MLXArray] = []
        generatedFrames.reserveCapacity(maxNewFrames)

        for _ in 0 ..< maxNewFrames {
            let globalInputsEmbeds = try buildInputsEmbeds(currentModelInputIDs)
            let globalOutputs = try transformer(
                inputsEmbeds: globalInputsEmbeds,
                attentionMask: currentAttentionMask,
                cache: cache
            )
            let globalHidden = globalOutputs[0..., -1, 0...]

            var localInputsEmbeds = globalHidden.expandedDimensions(axis: 1)
            var localOutputs = try localTransformer(inputsEmbeds: localInputsEmbeds)
            var localHidden = localOutputs[0..., -1, 0...]
            let textLogits = try textLMHead(localHidden)
            let nextTextToken = try mossSampleAssistantTextToken(
                textLogits: textLogits,
                audioAssistantSlotTokenID: config.audioAssistantSlotTokenID,
                audioEndTokenID: config.audioEndTokenID,
                doSample: doSample,
                temperature: textTemperature,
                topK: textTopK,
                topP: textTopP
            )
            eval(nextTextToken)
            if nextTextToken[0].item(Int.self) != config.audioAssistantSlotTokenID {
                break
            }

            var currentLocalInput = try transformer.tokenEmbedding(nextTextToken)
            var frameTokens: [MLXArray] = []
            frameTokens.reserveCapacity(effectiveNQ)
            let history = generatedFrames.isEmpty ? nil : MLX.stacked(generatedFrames, axis: 1)
            for channelIndex in 0 ..< effectiveNQ {
                localInputsEmbeds = MLX.concatenated(
                    [localInputsEmbeds, currentLocalInput.expandedDimensions(axis: 1)],
                    axis: 1
                )
                localOutputs = try localTransformer(inputsEmbeds: localInputsEmbeds)
                localHidden = localOutputs[0..., -1, 0...]
                let channelLogits = audioLMHead(localHidden, channelIndex: channelIndex)
                let previousTokens = history?[0..., 0..., channelIndex]
                let channelToken = try mossSampleNextToken(
                    logits: channelLogits,
                    doSample: doSample,
                    temperature: audioTemperature,
                    topK: audioTopK,
                    topP: audioTopP,
                    previousTokenIDs: previousTokens,
                    repetitionPenalty: audioRepetitionPenalty
                )
                frameTokens.append(channelToken)
                currentLocalInput = audioEmbeddings[channelIndex](channelToken)
            }

            var frame = MLX.stacked(frameTokens, axis: -1).asType(.int32)
            if effectiveNQ < config.nVQ {
                let pad = MLX.full(
                    [frame.dim(0), config.nVQ - effectiveNQ],
                    values: Int32(config.audioPadTokenID),
                    type: Int32.self
                )
                frame = MLX.concatenated([frame, pad], axis: -1)
            }
            generatedFrames.append(frame)

            let textColumn = MLX.full(
                [frame.dim(0), 1, 1],
                values: Int32(config.audioAssistantSlotTokenID),
                type: Int32.self
            )
            let nextRow = MLX.concatenated([textColumn, frame.expandedDimensions(axis: 1)], axis: -1)
            currentModelInputIDs = nextRow
            currentAttentionMask = MLX.concatenated(
                [currentAttentionMask, MLXArray.ones([frame.dim(0), 1], dtype: .bool)],
                axis: 1
            )
            eval(frame)

            if !useKVCache {
                promptInputIDs = MLX.concatenated([promptInputIDs, nextRow], axis: 1)
                currentModelInputIDs = promptInputIDs
            }
        }

        guard !generatedFrames.isEmpty else {
            return MLXArray.zeros([1, 0, config.nVQ], type: Int32.self)
        }
        return MLX.stacked(generatedFrames, axis: 1).asType(.int32)
    }

    public func generateAudioTokenIDs(
        text: String,
        promptAudioCodes: MLXArray,
        mode: String = "voice_clone",
        promptText: String? = nil,
        maxNewFrames: Int = 375,
        doSample: Bool = true
    ) throws -> MLXArray {
        guard let tokenizer else {
            throw MossTTSNanoError.tokenizerNotInitialized
        }
        let normalizedText = mossLightweightNormalizeText(text)
        let chunks = try mossSplitTextIntoBestSentences(
            tokenizer: tokenizer,
            text: normalizedText,
            maxTokens: 75
        )
        var allAudioTokens: [MLXArray] = []
        for chunk in chunks {
            let prepared = try buildInferenceInputIDs(
                text: chunk,
                tokenizer: tokenizer,
                mode: mode,
                promptText: mode == "continuation" ? promptText : nil,
                promptAudioCodes: promptAudioCodes
            )
            allAudioTokens.append(
                try generateAudioTokenIDs(
                    promptInputIDs: prepared.inputIDs,
                    attentionMask: prepared.attentionMask,
                    maxNewFrames: maxNewFrames,
                    doSample: doSample
                )
            )
        }
        return allAudioTokens.isEmpty
            ? MLXArray.zeros([1, 0, config.nVQ], type: Int32.self)
            : MLX.concatenated(allAudioTokens, axis: 1).asType(.int32)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        _ = refText
        _ = language
        var rendered: [MLXArray] = []
        try await forEachRenderedChunk(
            text: text, voice: voice, refAudio: refAudio, generationParameters: generationParameters
        ) { rendered.append($0) }
        guard !rendered.isEmpty else { return MLXArray.zeros([0], type: Float.self).materialized() }
        return (rendered.count == 1 ? rendered[0] : MLX.concatenated(rendered, axis: 0)).materialized()
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = refText
        _ = language
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    try await self.forEachRenderedChunk(
                        text: text, voice: voice, refAudio: refAudio,
                        generationParameters: generationParameters
                    ) { continuation.yield(.audio($0.materialized())) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Fork change: generates and decodes sentence by sentence, handing each
    /// mono chunk to `body`, instead of concatenating every chunk's codes and
    /// decoding once.
    ///
    /// The codec's deepest decoder stage attends over `frames * 32` positions,
    /// so a single decode grows with the SQUARE of the total length: two and a
    /// half minutes of speech peaked at 19.7 GB in one pass and 1.7 GB per
    /// chunk. Streaming callers also get their first audio after one sentence
    /// rather than after the whole text.
    ///
    /// The reference comes from `refAudio` when given (encoded with the codec,
    /// as upstream does) and otherwise from the bundled voice pack by name, so
    /// a caller that only asks for speech still gets a stable voice.
    private func forEachRenderedChunk(
        text: String,
        voice: String?,
        refAudio: MaterializedArray?,
        generationParameters: GenerateParameters,
        _ body: (MLXArray) -> Void
    ) async throws {
        guard let tokenizer else {
            throw MossTTSNanoError.tokenizerNotInitialized
        }
        try await ensureAudioTokenizer()
        let promptAudioCodes: MLXArray
        if let refAudio {
            promptAudioCodes = try encodeReferenceAudio(refAudio, numQuantizers: config.nVQ)
        } else {
            promptAudioCodes = try promptCodes(forVoice: voice)
        }

        let normalizedText = mossLightweightNormalizeText(text)
        let chunks = try mossSplitTextIntoBestSentences(
            tokenizer: tokenizer,
            text: normalizedText,
            maxTokens: maxTextTokensPerChunk
        )
        for chunk in chunks {
            try Task.checkCancellation()
            let prepared = try buildInferenceInputIDs(
                text: chunk,
                tokenizer: tokenizer,
                mode: "voice_clone",
                promptText: nil,
                promptAudioCodes: promptAudioCodes
            )
            let audioTokens = try generateAudioTokenIDs(
                promptInputIDs: prepared.inputIDs,
                attentionMask: prepared.attentionMask,
                nq: codebookCount,
                maxNewFrames: generationParameters.maxTokens ?? 375,
                doSample: generationParameters.temperature > 0,
                audioTemperature: generationParameters.temperature,
                audioTopP: generationParameters.topP,
                audioTopK: generationParameters.topK,
                audioRepetitionPenalty: generationParameters.repetitionPenalty ?? 1.1
            )
            guard audioTokens.dim(1) > 0 else { continue }
            let stereo = try decodeAudioTokenIDs(audioTokens.asType(.int32), numQuantizers: config.nVQ)
            let mono = Self.monoDownmix(stereo)
            mono.eval()
            body(mono)
            // Hand back the decoder's working set before the next chunk.
            MLX.Memory.clearCache()
        }
    }

    public static func fromPretrained(
        _ modelRepo: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> MossTTSNanoModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.model", "*.index.json"],
            hfToken: hfToken,
            cache: cache
        )
        return try await fromModelDirectory(modelDir, hfToken: hfToken, cache: cache)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> MossTTSNanoModel {
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        var config = try JSONDecoder().decode(MossTTSNanoConfig.self, from: configData)
        config.modelPath = modelDir.path
        let model = MossTTSNanoModel(config: config)
        model.hfToken = hfToken
        model.cache = cache

        let weights = try loadWeights(from: modelDir)
        let sanitizedWeights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
        eval(model)

        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.model")
        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            model.tokenizer = try MossSentencePieceTokenizer(modelURL: tokenizerURL)
        }
        let audioTokenizerURL = modelDir.appendingPathComponent("audio_tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: audioTokenizerURL.appendingPathComponent("config.json").path) {
            model.audioTokenizer = try MLXMossAudioTokenizer.fromModelDirectory(audioTokenizerURL)
        }
        model.voicePack = try? MossVoicePack.bundled()
        return model
    }
}
