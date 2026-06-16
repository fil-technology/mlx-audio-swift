import Foundation

public struct KittenTTSAlbertConfig: Codable, Sendable {
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let embeddingSize: Int
    public let innerGroupNum: Int
    public let numHiddenGroups: Int
    public let hiddenDropoutProb: Float
    public let attentionProbsDropoutProb: Float
    public let typeVocabSize: Int
    public let layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case embeddingSize = "embedding_size"
        case innerGroupNum = "inner_group_num"
        case numHiddenGroups = "num_hidden_groups"
        case hiddenDropoutProb = "hidden_dropout_prob"
        case attentionProbsDropoutProb = "attention_probs_dropout_prob"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
    }
}

public struct KittenTTSISTFTNetConfig: Codable, Sendable {
    public let resblockKernelSizes: [Int]
    public let upsampleRates: [Int]
    public let upsampleInitialChannel: Int
    public let resblockDilationSizes: [[Int]]
    public let upsampleKernelSizes: [Int]
    public let genISTFTNFFT: Int
    public let genISTFTHopSize: Int

    enum CodingKeys: String, CodingKey {
        case resblockKernelSizes = "resblock_kernel_sizes"
        case upsampleRates = "upsample_rates"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case genISTFTNFFT = "gen_istft_n_fft"
        case genISTFTHopSize = "gen_istft_hop_size"
    }
}

public struct KittenTTSConfiguration: Codable, Sendable {
    public let modelType: String
    public let sampleRate: Int
    public let hiddenDim: Int
    public let maxConvDim: Int
    public let maxDur: Int
    public let nLayer: Int
    public let nMels: Int
    public let nToken: Int
    public let styleDim: Int
    public let textEncoderKernelSize: Int
    public let asrResDim: Int
    public let decoderOutDim: Int?
    public let voicesPath: String
    public let speedPriors: [String: Float]
    public let voiceAliases: [String: String]
    public let activationQuantModules: [String]
    public let plbert: KittenTTSAlbertConfig
    public let istftnet: KittenTTSISTFTNetConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case hiddenDim = "hidden_dim"
        case maxConvDim = "max_conv_dim"
        case maxDur = "max_dur"
        case nLayer = "n_layer"
        case nMels = "n_mels"
        case nToken = "n_token"
        case styleDim = "style_dim"
        case textEncoderKernelSize = "text_encoder_kernel_size"
        case asrResDim = "asr_res_dim"
        case decoderOutDim = "decoder_out_dim"
        case voicesPath = "voices_path"
        case speedPriors = "speed_priors"
        case voiceAliases = "voice_aliases"
        case activationQuantModules = "activation_quant_modules"
        case plbert
        case istftnet
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "kitten_tts"
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
        hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        maxConvDim = try container.decode(Int.self, forKey: .maxConvDim)
        maxDur = try container.decode(Int.self, forKey: .maxDur)
        nLayer = try container.decode(Int.self, forKey: .nLayer)
        nMels = try container.decode(Int.self, forKey: .nMels)
        nToken = try container.decode(Int.self, forKey: .nToken)
        styleDim = try container.decode(Int.self, forKey: .styleDim)
        textEncoderKernelSize = try container.decode(Int.self, forKey: .textEncoderKernelSize)
        asrResDim = try container.decode(Int.self, forKey: .asrResDim)
        decoderOutDim = try container.decodeIfPresent(Int.self, forKey: .decoderOutDim)
        voicesPath = try container.decodeIfPresent(String.self, forKey: .voicesPath) ?? "voices.npz"
        speedPriors = try container.decodeIfPresent([String: Float].self, forKey: .speedPriors) ?? [:]
        voiceAliases = try container.decodeIfPresent([String: String].self, forKey: .voiceAliases) ?? [:]
        activationQuantModules = try container.decodeIfPresent([String].self, forKey: .activationQuantModules) ?? []
        plbert = try container.decode(KittenTTSAlbertConfig.self, forKey: .plbert)
        istftnet = try container.decode(KittenTTSISTFTNetConfig.self, forKey: .istftnet)
    }

    public static func load(from url: URL) throws -> KittenTTSConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(KittenTTSConfiguration.self, from: data)
    }
}
