import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public final class KittenTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: KittenTTSConfiguration
    public let modelFolder: URL
    public let voices: [String: MLXArray]

    private let textCleaner = KittenTextCleaner()
    private let textPreprocessor = KittenTextPreprocessor()

    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 1200,
            temperature: 0.7,
            topP: 0.95
        )
    }

    private init(
        config: KittenTTSConfiguration,
        modelFolder: URL,
        voices: [String: MLXArray]
    ) {
        self.config = config
        self.modelFolder = modelFolder
        self.voices = voices
        super.init()
    }

    public static func fromPretrained(
        _ modelRepo: String,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> KittenTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        let config = try KittenTTSConfiguration.load(from: modelDir.appendingPathComponent("config.json"))
        let voices = try loadVoices(from: modelDir.appendingPathComponent(config.voicesPath))
        return KittenTTSModel(config: config, modelFolder: modelDir, voices: voices)
    }

    public var availableVoices: [String] {
        Array(Set(Array(voices.keys) + Array(config.voiceAliases.keys))).sorted()
    }

    public func resolveVoiceName(_ voice: String?) throws -> String {
        let requested = voice?.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = config.voiceAliases.keys.sorted().first ?? voices.keys.sorted().first
        guard let normalizedInput = (requested?.isEmpty == false ? requested : fallback) else {
            throw AudioGenerationError.invalidInput("No KittenTTS voices were loaded from \(config.voicesPath).")
        }

        if let alias = config.voiceAliases[normalizedInput], voices[alias] != nil {
            return alias
        }
        if voices[normalizedInput] != nil {
            return normalizedInput
        }

        let lower = normalizedInput.lowercased()
        if let aliasMatch = config.voiceAliases.first(where: { $0.key.lowercased() == lower }),
           voices[aliasMatch.value] != nil {
            return aliasMatch.value
        }
        if let voiceMatch = voices.keys.first(where: { $0.lowercased() == lower }) {
            return voiceMatch
        }

        throw AudioGenerationError.invalidInput(
            "Voice '\(normalizedInput)' not available. Choose from: \(availableVoices.joined(separator: ", "))"
        )
    }

    public func voiceEmbedding(for voice: String?) throws -> MLXArray {
        let resolvedVoice = try resolveVoiceName(voice)
        guard let embedding = voices[resolvedVoice] else {
            throw AudioGenerationError.invalidInput("Missing KittenTTS voice embedding for \(resolvedVoice).")
        }
        return embedding
    }

    public func preprocess(text: String) -> String {
        textPreprocessor.process(text)
    }

    public func chunkedInput(text: String, maxLength: Int = 400) -> [String] {
        textPreprocessor.chunk(text, maxLength: maxLength)
    }

    public func tokenizePrephonemizedText(_ phonemes: String) -> [Int] {
        let tokens = KittenBasicEnglishTokenizer.tokenize(phonemes).joined(separator: " ")
        var ids = textCleaner.tokenIDs(for: tokens)
        ids.insert(0, at: 0)
        ids.append(0)
        return ids
    }

    public func prepareInputIDs(text: String, phonemes: String? = nil) -> MLXArray {
        let source = phonemes ?? preprocess(text: text)
        let tokenIDs = tokenizePrephonemizedText(source)
        return MLXArray(tokenIDs.map { Int32($0) }).expandedDimensions(axis: 0)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        _ = text
        _ = voice
        _ = refAudio
        _ = refText
        _ = language
        _ = generationParameters
        throw AudioGenerationError.invalidInput(
            "KittenTTSModel loading is implemented, but audio generation is not wired up yet."
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = text
        _ = voice
        _ = refAudio
        _ = refText
        _ = language
        _ = generationParameters
        return AsyncThrowingStream { continuation in
            continuation.finish(
                throwing: AudioGenerationError.invalidInput(
                    "KittenTTSModel streaming is not implemented yet."
                )
            )
        }
    }
}

private extension KittenTTSModel {
    static func loadVoices(from url: URL) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioGenerationError.modelNotInitialized("KittenTTS voices file not found at \(url.lastPathComponent).")
        }

        let arrays = try MLX.loadArrays(url: url)
        guard arrays.isEmpty == false else {
            throw AudioGenerationError.modelNotInitialized("KittenTTS voices file was empty.")
        }
        return arrays
    }
}

enum KittenBasicEnglishTokenizer {
    static func tokenize(_ text: String) -> [String] {
        let pattern = #"\w+|[^\w\s]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text.split(whereSeparator: \.isWhitespace).map(String.init)
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }
}

struct KittenTextCleaner {
    private let symbolToIndex: [Character: Int]

    init() {
        let pad = "$"
        let punctuation = ";:,.!?¡¿—…\"«»\" "
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        let ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        let symbols = Array((pad + punctuation + letters + ipa))
        var map: [Character: Int] = [:]
        for (index, symbol) in symbols.enumerated() {
            map[symbol] = index
        }
        self.symbolToIndex = map
    }

    func tokenIDs(for text: String) -> [Int] {
        text.compactMap { symbolToIndex[$0] }
    }
}

struct KittenTextPreprocessor {
    func process(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return "" }
        let ensuredPunctuation = Self.ensureTrailingPunctuation(for: trimmed)
        let normalized = ensuredPunctuation.folding(options: .diacriticInsensitive, locale: .current)
        return normalized.lowercased()
    }

    func chunk(_ text: String, maxLength: Int = 400) -> [String] {
        guard maxLength > 0 else { return [] }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return [] }

        let sentences = trimmed.components(separatedBy: CharacterSet(charactersIn: ".!?"))
        var chunks: [String] = []

        for rawSentence in sentences {
            let sentence = rawSentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard sentence.isEmpty == false else { continue }

            if sentence.count <= maxLength {
                chunks.append(Self.ensureTrailingPunctuation(for: sentence))
                continue
            }

            var current = ""
            for word in sentence.split(whereSeparator: \.isWhitespace) {
                let candidate = current.isEmpty ? String(word) : current + " " + word
                if candidate.count <= maxLength {
                    current = candidate
                } else {
                    if current.isEmpty == false {
                        chunks.append(Self.ensureTrailingPunctuation(for: current))
                    }
                    current = String(word)
                }
            }

            if current.isEmpty == false {
                chunks.append(Self.ensureTrailingPunctuation(for: current))
            }
        }

        return chunks
    }

    private static func ensureTrailingPunctuation(for text: String) -> String {
        guard let last = text.last else { return text }
        if ".!?,;:".contains(last) {
            return text
        }
        return text + ","
    }
}
