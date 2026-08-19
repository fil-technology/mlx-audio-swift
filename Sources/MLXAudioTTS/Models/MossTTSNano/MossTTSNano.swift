import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon
import HuggingFace
import MLXAudioCore

/// Decodes MOSS audio codes back to a waveform.
///
/// Split out so the backbone can be exercised (and validated) without the
/// codec, and so a streaming decoder can be swapped in later.
public protocol MossAudioDecoding: AnyObject {
    var sampleRate: Int { get }
    /// - Parameter audioCodes: `[batch, frames, nq]` integer codes.
    /// - Returns: interleaved samples for a single item.
    func decodeAudioCodes(_ audioCodes: MLXArray, numQuantizers: Int) throws -> MLXArray
}

/// MOSS-TTS-Nano — a 0.1B autoregressive TTS backbone over MOSS-Audio-Tokenizer
/// codes.
///
/// Ported from `mlx_audio/tts/models/moss_tts_nano/moss_tts_nano.py`
/// (Apache-2.0).
///
/// The model has **no built-in speaker embeddings**: every utterance is cloned
/// from reference audio. `voice:` selects a pre-encoded clip from the bundled
/// voice pack, which is why the codec *encoder* is not required on device.
public final class MossTTSNanoModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MossTTSNanoConfiguration

    @ModuleInfo(key: "transformer") var transformer: MossGPT2Model
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: [Embedding]
    @ModuleInfo(key: "local_transformer") var localTransformer: MossGPT2Model

    /// Text-token budget per generated chunk.
    ///
    /// The reference uses 75, which favours prosodic consistency. Lower values
    /// cut both time-to-first-audio and peak memory, because the codec decodes
    /// one chunk at a time and its deepest decoder stage attends over
    /// `frames * 32` positions — cost there grows faster than linearly with
    /// chunk length.
    public var maxTextTokensPerChunk: Int = 75

    private var tokenizer: SentencePieceBPETokenizer?
    private var audioDecoder: MossAudioDecoding?
    private var voicePack: MossVoicePack?

    public var sampleRate: Int { config.audioTokenizerSampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.8, topP: 0.95)
    }

    public init(_ config: MossTTSNanoConfiguration) {
        self.config = config
        self._transformer.wrappedValue = MossGPT2Model(config.gpt2, usesTokenEmbedding: true)
        self._audioEmbeddings.wrappedValue = config.audioCodebookSizes.map {
            Embedding(embeddingCount: $0, dimensions: config.gpt2.nEmbd)
        }
        self._localTransformer.wrappedValue = MossGPT2Model(
            config.localGPT2Configuration(), usesTokenEmbedding: false
        )
        super.init()
    }

    // MARK: - Weights

    /// Drops the tensors the reference discards. Both LM heads are **tied** to
    /// their embedding matrices, so the checkpoint's copies are dead weight
    /// (~37M parameters of it).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "text_lm_head.weight" { continue }
            if key.hasPrefix("audio_lm_heads.") { continue }
            if key == "local_transformer.wte.weight" { continue }
            if key.hasPrefix("transformer.wpe.") && transformer.wpe == nil { continue }
            if key.hasPrefix("local_transformer.wpe.") && localTransformer.wpe == nil { continue }
            sanitized[key] = value
        }
        return sanitized
    }

    // MARK: - Tied heads

    private func textLMHead(_ hidden: MLXArray) -> MLXArray {
        guard let wte = transformer.wte else {
            fatalError("MOSS-TTS-Nano global transformer must own a token embedding")
        }
        return hidden.matmul(wte.weight.T)
    }

    private func audioLMHead(_ hidden: MLXArray, channel: Int) -> MLXArray {
        hidden.matmul(audioEmbeddings[channel].weight.T)
    }

    // MARK: - Row construction

    /// A row is `[textToken, ch0, ch1, … ch15]`. Audio channels that carry no
    /// code hold `audioPadTokenId`, which is *outside* every codebook and so is
    /// masked out of the embedding sum rather than looked up.
    func buildInputEmbeddings(_ inputIDs: MLXArray) throws -> MLXArray {
        guard inputIDs.ndim == 3, inputIDs.dim(-1) == config.rowWidth else {
            throw MossTTSNanoError.invalidPromptCodes(
                "expected [batch, seq, \(config.rowWidth)], got \(inputIDs.shape)"
            )
        }
        guard let wte = transformer.wte else {
            throw MossTTSNanoError.invalidConfiguration("global transformer has no token embedding")
        }

        var embeddings = wte(inputIDs[.ellipsis, 0])
        for channel in 0 ..< audioEmbeddings.count {
            let channelIDs = inputIDs[.ellipsis, channel + 1]
            let valid = channelIDs .!= MLXArray(Int32(config.audioPadTokenId))
            let safeIDs = MLX.where(valid, channelIDs, MLXArray(Int32(0)))
            embeddings = embeddings + audioEmbeddings[channel](safeIDs) * valid[.ellipsis, .newAxis]
        }
        return embeddings
    }

    private func textRows(_ tokenIDs: [Int]) -> MLXArray {
        guard !tokenIDs.isEmpty else {
            return MLXArray.zeros([0, config.rowWidth], type: Int32.self)
        }
        var rows = MLXArray.full(
            [tokenIDs.count, config.rowWidth],
            values: MLXArray(Int32(config.audioPadTokenId)),
            type: Int32.self
        )
        rows[0..., 0] = MLXArray(tokenIDs.map { Int32($0) })
        return rows
    }

    private func audioPrefixRows(_ promptCodes: MLXArray, slotTokenID: Int) throws -> MLXArray {
        guard promptCodes.ndim == 2 else {
            throw MossTTSNanoError.invalidPromptCodes(
                "prompt codes must be [frames, nq], got \(promptCodes.shape)"
            )
        }
        let frames = promptCodes.dim(0)
        var rows = MLXArray.full(
            [frames, config.rowWidth],
            values: MLXArray(Int32(config.audioPadTokenId)),
            type: Int32.self
        )
        rows[0..., 0] = MLXArray.full([frames], values: MLXArray(Int32(slotTokenID)), type: Int32.self)
        let copyChannels = min(promptCodes.dim(1), config.nVQ)
        rows[0..., 1 ..< (1 + copyChannels)] = promptCodes[0..., 0 ..< copyChannels].asType(.int32)
        return rows
    }

    /// Builds the interleaved prompt for voice cloning:
    /// `<user …><audio_start>[reference codes]<audio_end>… text …<audio_start>`
    func buildInferenceInputIDs(text: String, promptCodes: MLXArray) throws -> MLXArray {
        guard let tokenizer else {
            throw MossTTSNanoError.tokenizerUnavailable("tokenizer.model was not loaded")
        }
        let textTokens = tokenizer.encode(text)
        let prefix = MossTTSNanoText.userPromptPrefix(tokenizer: tokenizer, config: config)
            + [config.audioStartTokenId]
        let suffix = [config.audioEndTokenId]
            + MossTTSNanoText.userPromptAfterReference(tokenizer: tokenizer)
            + textTokens
            + MossTTSNanoText.assistantPromptPrefix(tokenizer: tokenizer, config: config)
            + [config.audioStartTokenId]

        let sections = [
            textRows(prefix),
            try audioPrefixRows(promptCodes, slotTokenID: config.audioUserSlotTokenId),
            textRows(suffix),
        ]
        return MLX.concatenated(sections, axis: 0).expandedDimensions(axis: 0)
    }

    // MARK: - Generation

    public struct GenerationOptions: Sendable {
        public var maxNewFrames: Int = 375
        public var doSample: Bool = true
        public var textTemperature: Float = 1.0
        public var textTopP: Float = 1.0
        public var textTopK: Int = 50
        public var audioTemperature: Float = 0.8
        public var audioTopP: Float = 0.95
        public var audioTopK: Int = 25
        public var audioRepetitionPenalty: Float = 1.2
        public var maxTextTokensPerChunk: Int = 75
        public init() {}
    }

    /// Autoregressively rolls out audio frames for one prompt.
    ///
    /// Two nested loops: the *global* transformer advances one frame at a time
    /// with a KV cache, and the *local* transformer predicts the 16 codebook
    /// channels of that frame sequentially. The local stack is re-run over its
    /// (at most 17-position) sequence rather than cached — that is what the
    /// reference does and it is cheap at this size.
    func generateAudioTokenIDs(
        promptInputIDs: MLXArray,
        options: GenerationOptions
    ) throws -> MLXArray {
        let effectiveNQ = config.nVQ
        let cache = transformer.makeCache()
        var currentInputIDs = promptInputIDs
        var generatedFrames: [MLXArray] = []
        // Running "already emitted" mask per channel for the repetition penalty,
        // accumulated instead of recomputed from history each frame.
        var seenMasks: [MLXArray?] = Array(repeating: nil, count: effectiveNQ)
        let vocabRange = MLXArray(Int32(0) ..< Int32(config.audioVocabSize))

        for _ in 0 ..< options.maxNewFrames {
            let globalEmbeddings = try buildInputEmbeddings(currentInputIDs)
            let globalOutputs = transformer(inputEmbeddings: globalEmbeddings, cache: cache)
            let globalHidden = globalOutputs[0..., -1, 0...]

            var localEmbeddings = globalHidden.expandedDimensions(axis: 1)
            let localOutputs = localTransformer(inputEmbeddings: localEmbeddings)
            let textLogits = textLMHead(localOutputs[0..., -1, 0...])

            let nextTextToken = MossTTSNanoSampling.sampleAssistantTextToken(
                textLogits,
                audioAssistantSlotTokenID: config.audioAssistantSlotTokenId,
                audioEndTokenID: config.audioEndTokenId,
                doSample: options.doSample,
                temperature: options.textTemperature,
                topK: options.textTopK,
                topP: options.textTopP
            )
            nextTextToken.eval()
            guard nextTextToken.item(Int.self) == config.audioAssistantSlotTokenId else { break }

            guard let wte = transformer.wte else {
                throw MossTTSNanoError.invalidConfiguration("global transformer has no token embedding")
            }
            var currentLocalInput = wte(nextTextToken)
            var frameTokens: [MLXArray] = []

            for channel in 0 ..< effectiveNQ {
                localEmbeddings = MLX.concatenated(
                    [localEmbeddings, currentLocalInput.expandedDimensions(axis: 1)], axis: 1
                )
                let outputs = localTransformer(inputEmbeddings: localEmbeddings)
                let channelLogits = audioLMHead(outputs[0..., -1, 0...], channel: channel)
                let channelToken = MossTTSNanoSampling.sampleNextToken(
                    channelLogits,
                    doSample: options.doSample,
                    temperature: options.audioTemperature,
                    topK: options.audioTopK,
                    topP: options.audioTopP,
                    seenMask: seenMasks[channel],
                    repetitionPenalty: options.audioRepetitionPenalty
                )
                frameTokens.append(channelToken)
                let emitted = channelToken[.ellipsis, .newAxis] .== vocabRange
                seenMasks[channel] = seenMasks[channel].map { $0 .|| emitted } ?? emitted
                currentLocalInput = audioEmbeddings[channel](channelToken)
            }

            let frame = MLX.stacked(frameTokens, axis: -1)
            generatedFrames.append(frame)

            let textColumn = MLXArray.full(
                [frame.dim(0), 1, 1],
                values: MLXArray(Int32(config.audioAssistantSlotTokenId)),
                type: Int32.self
            )
            currentInputIDs = MLX.concatenated(
                [textColumn, frame.expandedDimensions(axis: 1)], axis: -1
            )
            frame.eval()
        }

        guard !generatedFrames.isEmpty else {
            return MLXArray.zeros([1, 0, config.nVQ], type: Int32.self)
        }
        return MLX.stacked(generatedFrames, axis: 1).asType(.int32)
    }

    /// Collapses the codec's stereo output to mono.
    ///
    /// `SpeechGenerationModel` is a mono contract — `makePCMBuffer` builds a
    /// 1-channel `AVAudioPCMBuffer` from a flat `[Float]`. MOSS is the first
    /// 2-channel model here, and handing its interleaved frames straight to
    /// that path plays L and R as consecutive mono samples: double length at
    /// half speed. MOSS renders what is effectively dual-mono (measured
    /// per-channel RMS agrees to five decimal places), so averaging is
    /// lossless in practice and safe if the channels ever diverge.
    static func monoDownmix(_ audio: MLXArray) -> MLXArray {
        guard audio.ndim == 2, audio.dim(1) > 1 else {
            return audio.ndim == 2 ? audio[0..., 0] : audio
        }
        return audio.mean(axis: -1)
    }

    /// Deterministic greedy rollout of a single frame's 16 channels.
    ///
    /// Used only by the parity tests: exact token agreement with the Python
    /// reference is a far sharper signal than tensor closeness, because one
    /// mis-ordered logit anywhere in the local stack flips a code.
    func greedyFirstFrameForTesting(promptInputIDs: MLXArray) throws -> [Int] {
        let embeddings = try buildInputEmbeddings(promptInputIDs)
        let hidden = transformer(inputEmbeddings: embeddings, cache: transformer.makeCache())
        let globalHidden = hidden[0..., -1, 0...]

        guard let wte = transformer.wte else {
            throw MossTTSNanoError.invalidConfiguration("global transformer has no token embedding")
        }
        var localEmbeddings = globalHidden.expandedDimensions(axis: 1)
        var currentLocalInput = wte(MLXArray([Int32(config.audioAssistantSlotTokenId)]))

        var frame: [Int] = []
        for channel in 0 ..< config.nVQ {
            localEmbeddings = MLX.concatenated(
                [localEmbeddings, currentLocalInput.expandedDimensions(axis: 1)], axis: 1
            )
            let outputs = localTransformer(inputEmbeddings: localEmbeddings)
            let logits = audioLMHead(outputs[0..., -1, 0...], channel: channel)
            let token = MLX.argMax(logits, axis: -1).asType(.int32)
            token.eval()
            frame.append(token.item(Int.self))
            currentLocalInput = audioEmbeddings[channel](token)
        }
        return frame
    }

    /// Voice used when a caller does not name one.
    ///
    /// MOSS has no speaker embeddings, so *some* reference is always required.
    /// Rather than fail a caller that simply asked for speech, fall back to a
    /// stable bundled clip — callers that care pick from ``availableVoices``.
    /// Kept fixed so output is reproducible across runs. `en_news` is a
    /// newsreader read, which is the common case for this framework and
    /// steadier than the conversational clips.
    public static let defaultVoice = "en_news"

    /// Resolves prompt codes from a bundled voice name.
    func promptCodes(forVoice voice: String?) throws -> MLXArray {
        guard let voicePack else {
            throw MossTTSNanoError.audioTokenizerUnavailable(
                "the bundled MOSS voice pack failed to load, so no reference is available"
            )
        }
        let requested = (voice?.isEmpty == false) ? voice! : Self.defaultVoice
        return try voicePack.codes(for: requested)
    }

    public var availableVoices: [String] { voicePack?.names ?? [] }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let codes = try await audioCodes(
            text: text, voice: voice, refAudio: refAudio, parameters: generationParameters
        )
        guard let audioDecoder else {
            throw MossTTSNanoError.audioTokenizerUnavailable("no audio decoder attached")
        }
        let stereo = try audioDecoder.decodeAudioCodes(codes, numQuantizers: config.nVQ)
        return Self.monoDownmix(stereo)
    }

    /// Generates codes for the whole text, chunk by chunk.
    func audioCodes(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        parameters: GenerateParameters
    ) async throws -> MLXArray {
        if refAudio != nil {
            throw MossTTSNanoError.audioTokenizerUnavailable(
                "reference-audio cloning needs the MOSS codec encoder, which is not ported yet; use a bundled voice"
            )
        }
        guard let tokenizer else {
            throw MossTTSNanoError.tokenizerUnavailable("tokenizer.model was not loaded")
        }
        let promptCodes = try promptCodes(forVoice: voice)

        var options = GenerationOptions()
        options.audioTemperature = Float(parameters.temperature)
        options.audioTopP = Float(parameters.topP)
        options.maxTextTokensPerChunk = maxTextTokensPerChunk

        let normalized = MossTTSNanoText.lightweightNormalize(text)
        let chunks = try MossTTSNanoText.splitIntoBestSentences(
            tokenizer: tokenizer, text: normalized, maxTokens: options.maxTextTokensPerChunk
        )

        var all: [MLXArray] = []
        for chunk in chunks {
            try Task.checkCancellation()
            let inputIDs = try buildInferenceInputIDs(text: chunk, promptCodes: promptCodes)
            all.append(try generateAudioTokenIDs(promptInputIDs: inputIDs, options: options))
        }
        guard !all.isEmpty else {
            return MLXArray.zeros([1, 0, config.nVQ], type: Int32.self)
        }
        return MLX.concatenated(all, axis: 1)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language, generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    /// Streams one decoded chunk at a time. MOSS re-primes the reference codes
    /// per chunk, so chunk boundaries are natural cut points and each chunk can
    /// be handed to the player as soon as it is decoded.
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let task = Task { @Sendable [weak self, continuation] in
            guard let self else { continuation.finish(); return }
            do {
                if refAudio != nil {
                    throw MossTTSNanoError.audioTokenizerUnavailable(
                        "reference-audio cloning needs the MOSS codec encoder, which is not ported yet; use a bundled voice"
                    )
                }
                guard let tokenizer = self.tokenizer else {
                    throw MossTTSNanoError.tokenizerUnavailable("tokenizer.model was not loaded")
                }
                guard let audioDecoder = self.audioDecoder else {
                    throw MossTTSNanoError.audioTokenizerUnavailable("no audio decoder attached")
                }
                let promptCodes = try self.promptCodes(forVoice: voice)

                var options = GenerationOptions()
                options.audioTemperature = Float(generationParameters.temperature)
                options.audioTopP = Float(generationParameters.topP)
                options.maxTextTokensPerChunk = self.maxTextTokensPerChunk

                let normalized = MossTTSNanoText.lightweightNormalize(text)
                let chunks = try MossTTSNanoText.splitIntoBestSentences(
                    tokenizer: tokenizer, text: normalized, maxTokens: options.maxTextTokensPerChunk
                )

                for chunk in chunks {
                    try Task.checkCancellation()
                    let inputIDs = try self.buildInferenceInputIDs(text: chunk, promptCodes: promptCodes)
                    let codes = try self.generateAudioTokenIDs(promptInputIDs: inputIDs, options: options)
                    guard codes.dim(1) > 0 else { continue }
                    let stereo = try audioDecoder.decodeAudioCodes(codes, numQuantizers: self.config.nVQ)
                    let audio = Self.monoDownmix(stereo)
                    audio.eval()
                    // The codec's deepest decoder stage allocates attention
                    // buffers proportional to (frames * 32)^2. Once a chunk is
                    // out those are dead but stay in MLX's buffer cache, and
                    // with a 24 s look-ahead several chunks' worth accumulate —
                    // enough to push an iPhone into jetsam territory. Returning
                    // them between chunks costs a little re-allocation and a lot
                    // less resident memory.
                    MLX.Memory.clearCache()
                    continuation.yield(.audio(audio))
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> MossTTSNanoModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw MossTTSNanoError.invalidConfiguration("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache,
            // MOSS ships a SentencePiece `tokenizer.model`, which the default
            // download globs (safetensors/json/txt/wav) would skip.
            additionalPatterns: ["*.model"]
        )

        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MossTTSNanoConfiguration.self, from: configData)

        let model = MossTTSNanoModel(config)

        var weights: [String: MLXArray] = [:]
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        for file in contents where file.pathExtension == "safetensors" {
            weights.merge(try MLX.loadArrays(url: file)) { current, _ in current }
        }
        guard !weights.isEmpty else {
            throw MossTTSNanoError.invalidConfiguration("no .safetensors weights in \(modelDir.path)")
        }

        let sanitized = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(model)

        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw MossTTSNanoError.tokenizerUnavailable("tokenizer.model missing from \(modelDir.path)")
        }
        model.tokenizer = try SentencePieceBPETokenizer(modelPath: tokenizerPath)
        model.voicePack = try? MossVoicePack.bundled()

        return model
    }

    /// Attaches the codec used to turn generated codes into audio.
    public func attach(audioDecoder: MossAudioDecoding) {
        self.audioDecoder = audioDecoder
    }

    func attach(tokenizer: SentencePieceBPETokenizer) {
        self.tokenizer = tokenizer
    }

    func attach(voicePack: MossVoicePack) {
        self.voicePack = voicePack
    }
}
