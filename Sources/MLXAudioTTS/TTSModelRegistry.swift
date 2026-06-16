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

enum TTSModelRegistry {
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
        )
    ]

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
