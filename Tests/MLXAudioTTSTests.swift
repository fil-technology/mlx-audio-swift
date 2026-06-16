//  Run the TTS suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SopranoTextCleaningTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/SopranoTextCleaningTests/testTextCleaning()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs


// MARK: - Text Cleaning Unit Tests

struct SopranoTextCleaningTests {

    @Test func testTextCleaning() {
        // Test number normalization
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        // Test abbreviations
        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        // Test ordinals
        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }
}

struct TTSModelRegistryTests {

    @Test func normalizesKnownAliases() {
        #expect(TTSModelRegistry.normalizedModelType("qwen") == "qwen3")
        #expect(TTSModelRegistry.normalizedModelType("orpheus_tts") == "llama_tts")
        #expect(TTSModelRegistry.normalizedModelType("pocket_tts") == "pocket_tts")
        #expect(TTSModelRegistry.normalizedModelType("kitten_tts") == "kitten_tts")
    }

    @Test func infersKnownFamiliesFromRepoNames() {
        #expect(TTSModelRegistry.inferModelType(from: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit") == "qwen3_tts")
        #expect(TTSModelRegistry.inferModelType(from: "mlx-community/orpheus-3b-0.1-ft-bf16") == "llama_tts")
        #expect(TTSModelRegistry.inferModelType(from: "mlx-community/pocket-tts") == "pocket_tts")
        #expect(TTSModelRegistry.inferModelType(from: "mlx-community/kitten-tts-mini-0.8") == "kitten_tts")
    }

    @Test func leavesUnknownFamiliesUnresolved() {
        #expect(TTSModelRegistry.normalizedModelType("totally_custom_tts") == nil)
        #expect(TTSModelRegistry.inferModelType(from: "mlx-community/unknown-voice-model") == nil)
    }
}

struct KittenTTSConfigTests {

    @Test func decodesKittenTTSConfiguration() throws {
        let json = """
        {
          "model_type": "kitten_tts",
          "sample_rate": 24000,
          "hidden_dim": 512,
          "max_conv_dim": 1024,
          "max_dur": 50,
          "n_layer": 3,
          "n_mels": 80,
          "n_token": 178,
          "style_dim": 128,
          "text_encoder_kernel_size": 5,
          "asr_res_dim": 64,
          "decoder_out_dim": 512,
          "voices_path": "voices.npz",
          "speed_priors": {},
          "voice_aliases": {
            "Bella": "expr-voice-2-f",
            "Jasper": "expr-voice-2-m"
          },
          "activation_quant_modules": [
            "bert_encoder",
            "decoder.generator.conv_post"
          ],
          "plbert": {
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
            "embedding_size": 128,
            "inner_group_num": 1,
            "num_hidden_groups": 1,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12
          },
          "istftnet": {
            "resblock_kernel_sizes": [3, 3],
            "upsample_rates": [10, 6],
            "upsample_initial_channel": 512,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5
          }
        }
        """

        let config = try JSONDecoder().decode(KittenTTSConfiguration.self, from: Data(json.utf8))
        #expect(config.modelType == "kitten_tts")
        #expect(config.sampleRate == 24_000)
        #expect(config.hiddenDim == 512)
        #expect(config.voiceAliases["Bella"] == "expr-voice-2-f")
        #expect(config.activationQuantModules.contains("bert_encoder"))
        #expect(config.plbert.hiddenSize == 768)
        #expect(config.istftnet.upsampleRates == [10, 6])
    }
}

struct KittenTTSModelShellTests {

    @Test func preprocessesAndTokenizesBasicInput() {
        let cleaned = KittenTextPreprocessor().process("Hello from Kitten TTS")
        #expect(cleaned == "hello from kitten tts,")

        let ids = KittenTextCleaner().tokenIDs(for: "hello,")
        #expect(ids.isEmpty == false)
    }

    @Test func chunksLongInputIntoPunctuatedSegments() {
        let chunks = KittenTextPreprocessor().chunk(
            "Hello from Kitten TTS. This sentence is intentionally longer so it has to be wrapped into multiple chunks for preprocessing.",
            maxLength: 36
        )

        #expect(chunks.isEmpty == false)
        #expect(chunks.allSatisfy { $0.count <= 37 })
        #expect(chunks.allSatisfy { ".!?,;:".contains($0.last ?? ",") })
    }
}
