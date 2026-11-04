import Foundation
@preconcurrency import MLX

/// Fork addition: pre-encoded reference clips ("voices") for MOSS-TTS-Nano.
///
/// MOSS-TTS-Nano has no speaker embeddings; it clones from reference audio on
/// every call. Upstream therefore requires `refAudio` and runs the codec
/// *encoder* on it. For a product that just wants a named voice, shipping the
/// clips already encoded to `[frames, nq]` prompt codes skips that encode on
/// every generation and keeps the whole pack at tens of kilobytes.
///
/// Codes were produced with `mlx-community/MOSS-Audio-Tokenizer-Nano` from the
/// English reference clips in OpenMOSS/MOSS-TTS-Nano `assets/audio` (Apache-2.0),
/// plus `en_news`: an 11 s excerpt of the English news sample published on the
/// OpenMOSS MOSS-TTS-Nano demo page, the closest match to newsreader delivery.
public final class MossVoicePack: @unchecked Sendable {
    private let codesByName: [String: MLXArray]

    public var names: [String] { codesByName.keys.sorted() }

    public init(codesByName: [String: MLXArray]) {
        self.codesByName = codesByName
    }

    public convenience init(contentsOf url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        guard !arrays.isEmpty else {
            throw MossTTSNanoError.unknownVoice("voice pack at \(url.lastPathComponent) is empty")
        }
        self.init(codesByName: arrays)
    }

    private static let cached: MossVoicePack? = {
        guard let url = Bundle.module.url(
            forResource: "voices", withExtension: "safetensors", subdirectory: "MossVoices"
        ) ?? Bundle.module.url(forResource: "voices", withExtension: "safetensors") else {
            return nil
        }
        return try? MossVoicePack(contentsOf: url)
    }()

    /// The pack shipped with this package.
    public static func bundled() throws -> MossVoicePack {
        guard let cached else {
            throw MossTTSNanoError.unknownVoice("bundled MOSS voice pack is unavailable")
        }
        return cached
    }

    public func codes(for name: String) throws -> MLXArray {
        guard let codes = codesByName[name] else {
            throw MossTTSNanoError.unknownVoice("\(name) (available: \(names.joined(separator: ", ")))")
        }
        return codes.asType(.int32)
    }
}
