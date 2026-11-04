import Foundation
import MLX
import Testing

@testable import MLXAudioTTS

@Suite("Chatterbox generation tests")
struct ChatterboxGenerationTests {
    @Test func emotionOverrideAppliesToDefaultConditioning() {
        let defaults = MLXArray(Float(0.5))

        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: nil).item(Float.self) == 0.5)
        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: 0.9).item(Float.self) == 0.9)
    }

    @Test func punctuationNormalizationMatchesReference() {
        // Mirrors `punc_norm` in resemble-ai/chatterbox.
        #expect(ChatterboxModel.normalizePunctuation("") == "You need to add some text for me to talk.")
        #expect(ChatterboxModel.normalizePunctuation("hello   world") == "Hello world.")
        #expect(ChatterboxModel.normalizePunctuation("Wait... what: this; that \u{2014} ok") == "Wait,  what, this,  that - ok.")
        #expect(ChatterboxModel.normalizePunctuation("\u{201C}Quoted\u{201D} and \u{2018}single\u{2019}!") == "\"Quoted\" and 'single'!")
        #expect(ChatterboxModel.normalizePunctuation("Already ends?") == "Already ends?")
        #expect(ChatterboxModel.normalizePunctuation("Trailing spaces   ") == "Trailing spaces.")
    }

    @Test func languageTagAcceptsCodesAndNames() {
        #expect(ChatterboxModel.languageTag(for: "fr") == "[fr]")
        #expect(ChatterboxModel.languageTag(for: "FR-ca") == "[fr]")
        #expect(ChatterboxModel.languageTag(for: "pt_BR") == "[pt]")
        #expect(ChatterboxModel.languageTag(for: "French") == "[fr]")
        #expect(ChatterboxModel.languageTag(for: " English ") == "[en]")
        #expect(ChatterboxModel.languageTag(for: "Chinese") == "[zh]")
        #expect(ChatterboxModel.languageTag(for: nil) == nil)
        #expect(ChatterboxModel.languageTag(for: "") == nil)
        #expect(ChatterboxModel.languageTag(for: "Klingon") == nil)
        #expect(ChatterboxModel.languageTag(for: "123") == nil)
    }

    @Test func multilingualCheckpointsWithoutT3BlockDecodeAsMultilingual() throws {
        // mlx-community/chatterbox-{fp16,8bit,4bit}: no t3 block, only the flag and vocab size.
        let flagged = try JSONDecoder().decode(
            ChatterboxConfiguration.self,
            from: Data(#"{"model_type": "chatterbox", "multilingual": true, "vocab_size": 2454}"#.utf8))
        #expect(flagged.t3Config.isMultilingual)
        #expect(!flagged.meanflow)

        let byVocab = try JSONDecoder().decode(
            ChatterboxConfiguration.self,
            from: Data(#"{"model_type": "chatterbox", "vocab_size": 2454}"#.utf8))
        #expect(byVocab.t3Config.isMultilingual)

        let english = try JSONDecoder().decode(
            ChatterboxConfiguration.self,
            from: Data(#"{"model_type": "chatterbox"}"#.utf8))
        #expect(!english.t3Config.isMultilingual)
        #expect(english.t3Config.textTokensDictSize == 704)

        let turbo = try JSONDecoder().decode(
            ChatterboxConfiguration.self,
            from: Data(#"{"model_type": "chatterbox_turbo"}"#.utf8))
        #expect(turbo.t3Config.isGPT)
        #expect(turbo.meanflow)
    }
}
