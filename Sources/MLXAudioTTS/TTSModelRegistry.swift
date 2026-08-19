import Foundation
import HuggingFace
import MLXAudioCore

struct TTSModelRegistryEntry {
    let canonicalType: String
    let aliases: Set<String>
    let repoMatchers: [String]
    let loader: @Sendable (_ modelRepo: String, _ cache: HubCache) async throws -> any SpeechGenerationModel

    init(
        canonicalType: String,
        aliases: [String] = [],
        repoMatchers: [String] = [],
        loader: @escaping @Sendable (_ modelRepo: String, _ cache: HubCache) async throws -> any SpeechGenerationModel
    ) {
        self.canonicalType = canonicalType
        self.aliases = Set([canonicalType] + aliases)
        self.repoMatchers = repoMatchers
        self.loader = loader
    }
}

public enum TTSModelRegistry {
    static let entries: [TTSModelRegistryEntry] = [
        .init(
            canonicalType: "qwen3_tts",
            aliases: ["qwen3-tts"],
            repoMatchers: ["qwen3_tts", "qwen3-tts"],
            loader: { modelRepo, cache in
                try await Qwen3TTSModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "qwen3",
            aliases: ["qwen"],
            repoMatchers: ["qwen3", "qwen"],
            loader: { modelRepo, cache in
                try await Qwen3Model.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "llama_tts",
            aliases: ["llama3_tts", "llama3", "llama", "orpheus", "orpheus_tts"],
            repoMatchers: ["llama", "orpheus"],
            loader: { modelRepo, cache in
                try await LlamaTTSModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "csm",
            aliases: ["sesame"],
            repoMatchers: ["csm", "sesame"],
            loader: { modelRepo, cache in
                try await MarvisTTSModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "soprano",
            aliases: ["soprano_tts"],
            repoMatchers: ["soprano"],
            loader: { modelRepo, cache in
                try await SopranoModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "pocket_tts",
            repoMatchers: ["pocket_tts", "pocket-tts"],
            loader: { modelRepo, cache in
                try await PocketTTSModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "kitten_tts",
            aliases: ["kitten", "kitten-tts"],
            repoMatchers: ["kitten_tts", "kitten-tts"],
            loader: { modelRepo, cache in
                try await KittenTTSModel.fromPretrained(modelRepo, cache: cache)
            }
        ),
        .init(
            canonicalType: "moss_tts_nano",
            aliases: ["moss-tts-nano", "moss_tts", "moss"],
            repoMatchers: ["moss-tts-nano", "moss_tts_nano"],
            loader: { modelRepo, cache in
                let model = try await MossTTSNanoModel.fromPretrained(modelRepo, cache: cache)
                // MOSS keeps its codec in a separate repo, so the decoder is
                // fetched and attached here rather than by the model loader.
                //
                // The published config names the *PyTorch* codec repo
                // (`OpenMOSS-Team/...`), whose remote-code weights this loader
                // cannot read, so honour it only when it already points at an
                // MLX conversion and otherwise use the mlx-community build.
                let configured = model.config.audioTokenizerRepo
                let repo = (configured?.lowercased().contains("mlx") == true)
                    ? configured!
                    : "mlx-community/MOSS-Audio-Tokenizer-Nano"
                let decoder = try await MossAudioTokenizerModel.fromPretrained(repo, cache: cache)
                model.attach(audioDecoder: decoder)
                return model
            }
        )
    ]

    /// Canonical loader type for a model, or `nil` when nothing here can load
    /// it.
    ///
    /// Exposed so callers (notably TTSMLX's model store) can ask *this* table
    /// whether a model is runnable, instead of keeping a parallel copy that
    /// silently goes stale — a model missing from such a copy reports as
    /// "unsupported by the current MLX runtime" even though the loader exists.
    ///
    /// - Parameters:
    ///   - modelType: `model_type` from the model's `config.json`, if known.
    ///   - architectures: `architectures` from `config.json`, if known.
    ///   - repo: repository id, used only as a last-resort hint.
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
                entry.aliases.contains(where: { lowered.contains($0.replacingOccurrences(of: "_", with: "")) })
                    || entry.repoMatchers.contains(where: { lowered.contains($0.replacingOccurrences(of: "_", with: "")) })
            }) {
                return match.canonicalType
            }
        }
        if let repo {
            return inferModelType(from: repo)
        }
        return nil
    }

    /// Every loader type this build can route to.
    public static var supportedModelTypes: [String] {
        entries.map(\.canonicalType)
    }

    static func normalizedModelType(_ modelType: String?) -> String? {
        guard let modelType else { return nil }
        let normalized = modelType
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard normalized.isEmpty == false else { return nil }

        return entries.first(where: { $0.aliases.contains(normalized) })?.canonicalType
    }

    static func inferModelType(from modelRepo: String) -> String? {
        let lower = modelRepo.lowercased()
        for entry in entries {
            if entry.repoMatchers.contains(where: { lower.contains($0) }) {
                return entry.canonicalType
            }
        }
        return nil
    }

    static func loadModel(
        modelRepo: String,
        modelType: String?,
        cache: HubCache = .default
    ) async throws -> any SpeechGenerationModel {
        let resolvedType = normalizedModelType(modelType) ?? inferModelType(from: modelRepo)
        guard let resolvedType else {
            throw TTSModelError.unsupportedModelType(modelType)
        }

        guard let entry = entries.first(where: { $0.canonicalType == resolvedType }) else {
            throw TTSModelError.unsupportedModelType(modelType ?? resolvedType)
        }

        return try await entry.loader(modelRepo, cache)
    }
}
