import Foundation
import Testing
import HuggingFace
import MLXAudioCore
@testable import MLXAudioTTS

/// Parity tests for the SentencePiece BPE reader against the Python
/// `sentencepiece` reference. Fixtures were produced with the tokenizer.model
/// shipped by `mlx-community/MOSS-TTS-Nano-100M`.
///
/// The tokenizer path is resolved from the local HF cache; the suite skips
/// itself when the model has not been downloaded on this machine.
struct MossTTSNanoTokenizerTests {
    /// Resolves the MOSS model directory through the same cache the loader
    /// uses, so tests see whatever `fromPretrained` would see. Downloads on
    /// first run (~285 MB).
    static func resolveModelDirectory() async -> URL? {
        guard let repoID = Repo.ID(rawValue: "mlx-community/MOSS-TTS-Nano-100M") else { return nil }
        return try? await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", hfToken: nil, cache: .default,
            additionalPatterns: ["*.model"]
        )
    }

    /// Set `MOSS_REQUIRE_FIXTURES=1` to turn "weights not present" into a
    /// failure instead of a skip. Without it a missing download would make
    /// these suites report a green run that verified nothing.
    static var requiresFixtures: Bool {
        ProcessInfo.processInfo.environment["MOSS_REQUIRE_FIXTURES"] == "1"
    }

    static func tokenizerOrSkip() async throws -> SentencePieceBPETokenizer? {
        guard let dir = await resolveModelDirectory() else {
            #expect(!requiresFixtures, "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return nil
        }
        return try SentencePieceBPETokenizer(
            modelPath: dir.appendingPathComponent("tokenizer.model")
        )
    }

    @Test func vocabularySizeMatchesConfig() async throws {
        guard let tokenizer = try await Self.tokenizerOrSkip() else { return }
        #expect(tokenizer.vocabularySize == 16384)
    }

    @Test func encodesReferenceFixtures() async throws {
        guard let tokenizer = try await Self.tokenizerOrSkip() else { return }

        let cases: [(String, [Int])] = [
            ("Hello world.", [7026, 1177, 10380]),
            (
                "Global markets closed higher on Tuesday after the central bank signalled it would hold interest rates steady through the end of the year.",
                [558, 760, 4002, 7139, 10027, 4250, 346, 8756, 1084, 280, 9301, 5792, 1813, 359, 316, 329, 580, 2546, 1198, 6423, 2799, 5190, 917, 280, 1069, 308, 280, 683, 10380]
            ),
            ("user\n", [600, 289]),
            ("<user_inst>\n- Reference(s):\n", [10356, 13, 10356, 10425, 1860, 4546, 12907, 10363, 13325, 11492]),
            ("\n</user_inst>", [10356, 14]),
            ("assistant\n", [8165, 430]),
            ("None", [505, 587]),
            ("The S&P 500 rose 1.2% to 5,431.", [453, 348, 12589, 10490, 9219, 354, 610, 8520, 10384, 10903, 300, 1589, 10364, 10377, 10386, 3878]),
            ("Reporting from Kyiv, our correspondent said the ceasefire held overnight.", [3557, 478, 287, 538, 794, 10371, 840, 10364, 642, 1774, 573, 10376, 805, 355, 772, 280, 1548, 955, 10375, 1283, 6815, 722, 8631, 10380]),
            ("Analysts cautioned that inflation remains above target — a risk.", [2793, 7657, 315, 10363, 3420, 2189, 316, 309, 7086, 1099, 4368, 6389, 4998, 10356, 12416, 273, 3593, 10380]),
            ("naïve café résumé", [813, 14592, 323, 3420, 10375, 10615, 354, 3815, 544, 10615]),
            ("  spaced   out  text  ", [716, 6956, 502, 5078]),
        ]

        for (text, expected) in cases {
            let actual = tokenizer.encode(text)
            #expect(actual == expected, "mismatch for \(text.debugDescription): got \(actual), expected \(expected)")
        }
    }

    @Test func roundTripsThroughDecode() async throws {
        guard let tokenizer = try await Self.tokenizerOrSkip() else { return }
        let text = "Global markets closed higher on Tuesday."
        #expect(tokenizer.decode(tokenizer.encode(text)).trimmingCharacters(in: .whitespaces) == text)
    }

    /// SentencePiece emits nothing for whitespace-only input. MOSS's prompt
    /// template separates turns with "\n", so a stray dummy-prefix token here
    /// shifts every downstream position.
    @Test func whitespaceOnlyInputProducesNoTokens() async throws {
        guard let tokenizer = try await Self.tokenizerOrSkip() else { return }
        for text in ["\n", " ", "", "  ", "\t", "\n\n"] {
            #expect(tokenizer.encode(text).isEmpty,
                    "\(text.debugDescription) produced \(tokenizer.encode(text))")
        }
        // Surrounding whitespace is still trimmed to the same token as the bare word.
        #expect(tokenizer.encode(" a ") == tokenizer.encode("a"))
    }
}
