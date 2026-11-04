import Foundation

/// Fork addition (fil-technology/mlx-audio-swift): a queryable view of what
/// ``TTS/loadModel(modelRepo:hfToken:cache:)`` can route to.
///
/// Upstream keeps the routing table inside a `switch` in `TTSModel.swift`, which
/// a caller cannot ask "would this load?" without actually loading it. TTSMLX's
/// model store needs exactly that answer to list downloadable models, and its
/// background downloader needs to know which side repositories a model fetches
/// at load time so an onboarding prefetch does not stall on first playback.
///
/// Keep `entries` in sync with the `switch` in `TTS.loadResolvedModel`; the
/// canonical type of every entry is the string upstream's `inferModelType`
/// produces for that family.
public enum TTSModelRegistry {
    struct Entry {
        let canonicalType: String
        let aliases: Set<String>
        /// Repositories fetched in addition to the model's own at load or first
        /// generation. Listed only when upstream names them unconditionally.
        let companionRepositories: [String]

        init(_ canonicalType: String, aliases: [String] = [], companions: [String] = []) {
            self.canonicalType = canonicalType
            self.aliases = Set([canonicalType] + aliases)
            self.companionRepositories = companions
        }
    }

    static let entries: [Entry] = [
        .init("arktts", aliases: ["audio8"]),
        .init("bailingmm", aliases: ["ming_omni_tts"]),
        .init("bark", companions: ["mlx-community/encodec-24khz-float32"]),
        .init("breeze", aliases: ["breeze_tts"]),
        .init("chatterbox", aliases: ["chatterbox_tts", "chatterbox_turbo"],
              companions: ["mlx-community/S3TokenizerV2"]),
        .init("confucius4", aliases: ["confucius4_tts"]),
        .init("csm", aliases: ["sesame"]),
        .init("dense", aliases: ["ming_dense"], companions: ["mlx-community/Step-Audio-2-token2wav"]),
        .init("dia", companions: ["mlx-community/descript-audio-codec-44khz"]),
        .init("echo_tts", aliases: ["echo"], companions: ["jordand/fish-s1-dac-min"]),
        .init("fish_speech", aliases: ["fish_qwen3_omni"]),
        .init("higgs_audio", aliases: ["higgs_audio_v2"]),
        .init("higgs_audio_v3", aliases: ["higgs_multimodal_qwen3"]),
        .init("indextts", aliases: ["index_tts"]),
        .init("irodori_tts", aliases: ["irodori"],
              companions: ["llm-jp/llm-jp-3-150m", "Aratako/Semantic-DACVAE-Japanese-32dim"]),
        // English G2P weights; the multilingual lexicons are fetched per language on demand.
        .init("kitten_tts", aliases: ["kitten"], companions: ["beshkenadze/kitten-tts-g2p"]),
        .init("kokoro", aliases: ["kokoro_tts"], companions: ["beshkenadze/kitten-tts-g2p"]),
        .init("kugelaudio"),
        .init("lingv3_tts", aliases: ["ming_tiny"]),
        .init("llama_tts", aliases: ["llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts"],
              companions: ["mlx-community/snac_24khz"]),
        .init("longcat_audiodit", aliases: ["audiodit"]),
        .init("melotts", aliases: ["melo_tts"]),
        .init("moss_tts", aliases: ["moss_tts_delay", "moss_tts_local"],
              companions: ["OpenMOSS-Team/MOSS-Audio-Tokenizer"]),
        .init("moss_tts_nano", companions: ["mlx-community/MOSS-Audio-Tokenizer-Nano"]),
        .init("omnivoice"),
        .init("outetts", aliases: ["oute_tts"], companions: ["mlx-community/dac-speech-24khz-1.5kbps"]),
        .init("pocket_tts", aliases: ["phonon"]),
        .init("qwen3", aliases: ["qwen"], companions: ["mlx-community/snac_24khz"]),
        .init("qwen3_tts"),
        .init("rumik_oss", aliases: ["rumik"]),
        .init("soprano", aliases: ["soprano_tts"]),
        .init("spark", aliases: ["spark_tts"]),
        .init("tada", aliases: ["tada_tts"]),
        .init("vibevoice"),
        .init("vibevoice_streaming"),
        .init("voxcpm", aliases: ["vox_cpm", "voxcpm1", "voxcpm1.5"]),
        .init("voxcpm2", aliases: ["vox_cpm2"]),
        .init("voxtral_tts"),
        .init("zonos2"),
    ]

    /// Canonical loader type for a model, or `nil` when nothing here can load it.
    ///
    /// - Parameters:
    ///   - modelType: `model_type` from the model's `config.json`, if known.
    ///   - architectures: `architectures` from `config.json`, if known.
    ///   - repo: repository id, used as a last-resort hint the same way the
    ///     loader uses it.
    public static func canonicalModelType(
        modelType: String? = nil,
        architectures: [String] = [],
        repo: String? = nil
    ) -> String? {
        if let normalized = normalizedModelType(modelType) {
            return normalized
        }
        for architecture in architectures {
            let lowered = architecture.lowercased()
            if let match = entries.first(where: { entry in
                entry.aliases.contains { lowered.contains($0.replacingOccurrences(of: "_", with: "")) }
            }) {
                return match.canonicalType
            }
        }
        if let repo, let inferred = TTS.inferModelType(from: repo) {
            return normalizedModelType(inferred) ?? inferred
        }
        return nil
    }

    /// Repositories that must be present, in addition to the model's own, for
    /// a model of this type to generate without further network access.
    public static func companionRepositories(
        modelType: String? = nil,
        architectures: [String] = [],
        repo: String? = nil
    ) -> [String] {
        guard let canonical = canonicalModelType(
            modelType: modelType, architectures: architectures, repo: repo
        ) else { return [] }
        return entries.first { $0.canonicalType == canonical }?.companionRepositories ?? []
    }

    /// Every loader type this build can route to.
    public static var supportedModelTypes: [String] {
        entries.map(\.canonicalType)
    }

    static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let normalized = modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return nil }
        return entries.first { $0.aliases.contains(normalized) }?.canonicalType
    }
}
