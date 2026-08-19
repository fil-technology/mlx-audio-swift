import Foundation

/// Configuration for the MOSS-TTS-Nano backbone.
///
/// Ported from `mlx_audio/tts/models/moss_tts_nano/config.py` (Apache-2.0).
/// The published `config.json` for `mlx-community/MOSS-TTS-Nano-100M` is flat —
/// it carries `hidden_size`/`vocab_size` but no nested `gpt2_config` — so every
/// GPT-2 field falls back to the reference defaults unless overridden.
public struct MossGPT2Configuration: Decodable, Sendable {
    public var modelType: String = "gpt2"
    public var vocabSize: Int = 16384
    public var nPositions: Int = 32768
    public var nEmbd: Int = 768
    public var nLayer: Int = 12
    public var nHead: Int = 12
    public var nInner: Int = 3072
    public var activationFunction: String = "gelu_new"
    public var layerNormEpsilon: Float = 1e-5
    public var scaleAttnWeights: Bool = true
    public var scaleAttnByInverseLayerIdx: Bool = false
    public var positionEmbeddingType: String = "rope"
    public var ropeBase: Float = 10000.0
    public var padTokenId: Int = 3
    public var bosTokenId: Int = 1
    public var eosTokenId: Int = 2

    public var headDim: Int { nEmbd / nHead }
    public var usesRoPE: Bool { positionEmbeddingType.lowercased() == "rope" }
    public var usesAbsolutePositions: Bool { positionEmbeddingType.lowercased() == "absolute" }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case nPositions = "n_positions"
        case nEmbd = "n_embd"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case nInner = "n_inner"
        case activationFunction = "activation_function"
        case layerNormEpsilon = "layer_norm_epsilon"
        case scaleAttnWeights = "scale_attn_weights"
        case scaleAttnByInverseLayerIdx = "scale_attn_by_inverse_layer_idx"
        case positionEmbeddingType = "position_embedding_type"
        case ropeBase = "rope_base"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        // Aliases accepted by the reference `GPT2Config.from_dict`.
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MossGPT2Configuration()
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? d.modelType
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? d.vocabSize
        nPositions = try c.decodeIfPresent(Int.self, forKey: .nPositions) ?? d.nPositions
        // `from_dict` prefers the explicit GPT-2 key, then the HF-style alias.
        nEmbd = try c.decodeIfPresent(Int.self, forKey: .nEmbd)
            ?? c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? d.nEmbd
        nLayer = try c.decodeIfPresent(Int.self, forKey: .nLayer)
            ?? c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? d.nLayer
        nHead = try c.decodeIfPresent(Int.self, forKey: .nHead)
            ?? c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? d.nHead
        nInner = try c.decodeIfPresent(Int.self, forKey: .nInner)
            ?? c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? d.nInner
        activationFunction = try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? d.activationFunction
        layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? d.layerNormEpsilon
        scaleAttnWeights = try c.decodeIfPresent(Bool.self, forKey: .scaleAttnWeights) ?? d.scaleAttnWeights
        scaleAttnByInverseLayerIdx = try c.decodeIfPresent(Bool.self, forKey: .scaleAttnByInverseLayerIdx) ?? d.scaleAttnByInverseLayerIdx
        positionEmbeddingType = try c.decodeIfPresent(String.self, forKey: .positionEmbeddingType) ?? d.positionEmbeddingType
        ropeBase = try c.decodeIfPresent(Float.self, forKey: .ropeBase) ?? d.ropeBase
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? d.padTokenId
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? d.bosTokenId
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? d.eosTokenId
    }
}

public struct MossTTSNanoConfiguration: Decodable, Sendable {
    public var modelType: String = "moss_tts_nano"
    public var gpt2: MossGPT2Configuration = .init()
    public var nVQ: Int = 16
    public var audioVocabSize: Int = 1024
    public var audioCodebookSizes: [Int] = Array(repeating: 1024, count: 16)
    public var audioPadTokenId: Int = 1024
    public var padTokenId: Int = 3
    public var imStartTokenId: Int = 4
    public var imEndTokenId: Int = 5
    public var audioStartTokenId: Int = 6
    public var audioEndTokenId: Int = 7
    public var audioUserSlotTokenId: Int = 8
    public var audioAssistantSlotTokenId: Int = 9
    public var audioTokenizerType: String = "moss-audio-tokenizer-nano"
    public var audioTokenizerRepo: String?
    public var audioTokenizerSampleRate: Int = 48000
    public var localTransformerLayers: Int = 1

    /// Row width of the interleaved token matrix: one text column plus one
    /// column per VQ channel.
    public var rowWidth: Int { nVQ + 1 }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case gpt2 = "gpt2_config"
        case nVQ = "n_vq"
        case audioVocabSize = "audio_vocab_size"
        case audioCodebookSizes = "audio_codebook_sizes"
        case audioPadTokenId = "audio_pad_token_id"
        case padTokenId = "pad_token_id"
        case imStartTokenId = "im_start_token_id"
        case imEndTokenId = "im_end_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case audioEndTokenId = "audio_end_token_id"
        case audioUserSlotTokenId = "audio_user_slot_token_id"
        case audioAssistantSlotTokenId = "audio_assistant_slot_token_id"
        case audioTokenizerType = "audio_tokenizer_type"
        case audioTokenizerRepo = "audio_tokenizer_pretrained_name_or_path"
        case audioTokenizerSampleRate = "audio_tokenizer_sample_rate"
        case localTransformerLayers = "local_transformer_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MossTTSNanoConfiguration()
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? d.modelType
        // The flat published config has no `gpt2_config`; decoding the top level
        // again picks up `hidden_size` / `vocab_size` through the alias keys.
        gpt2 = try c.decodeIfPresent(MossGPT2Configuration.self, forKey: .gpt2)
            ?? (try? MossGPT2Configuration(from: decoder))
            ?? d.gpt2
        nVQ = try c.decodeIfPresent(Int.self, forKey: .nVQ) ?? d.nVQ
        audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? d.audioVocabSize
        audioCodebookSizes = try c.decodeIfPresent([Int].self, forKey: .audioCodebookSizes)
            ?? Array(repeating: audioVocabSize, count: nVQ)
        guard audioCodebookSizes.count == nVQ else {
            throw MossTTSNanoError.invalidConfiguration(
                "audio_codebook_sizes must have one entry per VQ channel (expected \(nVQ), got \(audioCodebookSizes.count))"
            )
        }
        audioPadTokenId = try c.decodeIfPresent(Int.self, forKey: .audioPadTokenId) ?? d.audioPadTokenId
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? d.padTokenId
        imStartTokenId = try c.decodeIfPresent(Int.self, forKey: .imStartTokenId) ?? d.imStartTokenId
        imEndTokenId = try c.decodeIfPresent(Int.self, forKey: .imEndTokenId) ?? d.imEndTokenId
        audioStartTokenId = try c.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? d.audioStartTokenId
        audioEndTokenId = try c.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? d.audioEndTokenId
        audioUserSlotTokenId = try c.decodeIfPresent(Int.self, forKey: .audioUserSlotTokenId) ?? d.audioUserSlotTokenId
        audioAssistantSlotTokenId = try c.decodeIfPresent(Int.self, forKey: .audioAssistantSlotTokenId) ?? d.audioAssistantSlotTokenId
        audioTokenizerType = try c.decodeIfPresent(String.self, forKey: .audioTokenizerType) ?? d.audioTokenizerType
        audioTokenizerRepo = try c.decodeIfPresent(String.self, forKey: .audioTokenizerRepo)
        audioTokenizerSampleRate = try c.decodeIfPresent(Int.self, forKey: .audioTokenizerSampleRate) ?? d.audioTokenizerSampleRate
        localTransformerLayers = try c.decodeIfPresent(Int.self, forKey: .localTransformerLayers) ?? d.localTransformerLayers
    }

    /// The local transformer is the same GPT-2 stack shrunk to `n_vq + 1`
    /// positions and `local_transformer_layers` layers.
    public func localGPT2Configuration() -> MossGPT2Configuration {
        var local = gpt2
        local.nPositions = rowWidth
        local.nLayer = localTransformerLayers
        return local
    }
}

public enum MossTTSNanoError: Error, LocalizedError {
    case invalidConfiguration(String)
    case tokenizerUnavailable(String)
    case unknownVoice(String)
    case referenceAudioRequired
    case invalidPromptCodes(String)
    case audioTokenizerUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "MOSS-TTS-Nano configuration is invalid: \(detail)"
        case .tokenizerUnavailable(let detail):
            return "MOSS-TTS-Nano tokenizer is unavailable: \(detail)"
        case .unknownVoice(let name):
            return "MOSS-TTS-Nano has no bundled voice named '\(name)'."
        case .referenceAudioRequired:
            return "MOSS-TTS-Nano generates by voice cloning: supply a bundled voice name or reference audio."
        case .invalidPromptCodes(let detail):
            return "MOSS-TTS-Nano prompt codes are invalid: \(detail)"
        case .audioTokenizerUnavailable(let detail):
            return "MOSS-Audio-Tokenizer-Nano is unavailable: \(detail)"
        }
    }
}
