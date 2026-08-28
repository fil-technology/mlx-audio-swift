import Foundation
import Testing
@preconcurrency import MLX
@testable import MLXAudioTTS

/// Measures MOSS on real book prose, so a whole-book estimate rests on
/// measured throughput rather than on a benchmark sentence.
///
/// The passage is 373 words of technical writing from LLM Engineer's
/// Handbook — long sentences, acronyms and parenthetical asides, which is
/// harder for a TTS model than a news headline and therefore a fairer basis
/// for extrapolating a 130,000-word book.
@Suite(.serialized)
struct MossBookThroughputTests {
    static let passage = "Retrieval-augmented generation (RAG) is fundamental in most generative AI applications. RAG’s core responsibility is to inject custom data into the large language model (LLM) to perform a given action (e.g., summarize, reformulate, and extract the injected data). You often want to use the LLM on data it wasn’t trained on (e.g., private or new data). As fine-tuning an LLM is a highly costly operation, RAG is a compelling strategy that bypasses the need for constant fine-tuning to access that new data. We will start this chapter with a theoretical part that focuses on the fundamentals of RAG and how it works. We will then walk you through all the components of a naïve RAG system: chunking, embedding, and vector DBs. Ultimately, we will present various optimizations used for an advanced RAG system. Then, we will continue exploring LLM Twin’s RAG feature pipeline architecture. At this step, we will apply all the theoretical aspects we discussed at the beginning of the chapter. Finally, we will go through a practical example by implementing the LLM Twin’s RAG feature pipeline based on the system design described throughout the book. By the end of this chapter, you will have a clear and comprehensive understanding of what RAG is and how it is applied to our LLM Twin use case. Any LLM is bound to understand the data it was trained on, sometimes called parameterized knowledge. Thus, even if the LLM can perfectly answer what happened in the past, it won’t have access to the newest data or any other external sources on which it wasn’t trained. Let’s take the most powerful model from OpenAI as an example, which, in the summer of 2024, is GPT-4o. The model is trained on data up to October 2023. Thus, if we ask what happened during the 2020 pandemic, it can be answered perfectly due to its parametrized knowledge. However, it will not know the answer if we ask about the 2024 European Football Championship results due to its bounded parametrized knowledge. Another scenario is that it will start confidently hallucinating and provide a faulty answer. RAG overcomes these two limitations of LLMs. It provides access to external or latest data and prevents hallucinations, enhancing generative AI models’ accuracy and reliability."

    @Test func measuresThroughputOnRealBookProse() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }
        let model = try await TTS.loadModel(modelRepo: "mlx-community/MOSS-TTS-Nano-100M")
        guard let moss = model as? MossTTSNanoModel else { return }

        let words = Self.passage.split(separator: " ").count
        for codebooks in [16, 8] {
            moss.codebookCount = codebooks
            MLX.GPU.resetPeakMemory()
            let started = Date()
            let audio = try await moss.generate(
                text: Self.passage, voice: "en_news", refAudio: nil, refText: nil,
                language: "English", generationParameters: moss.defaultGenerationParameters
            )
            audio.eval()
            let elapsed = Date().timeIntervalSince(started)
            let seconds = Double(audio.dim(0)) / 48_000.0
            let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576.0

            print(String(
                format: "[moss-book] codebooks=%2d words=%d audio=%6.1fs (%.2f w/s) "
                      + "generated in %6.1fs (%.2fx realtime) peak=%.0fMB",
                codebooks, words, seconds, Double(words) / seconds,
                elapsed, seconds / elapsed, peakMB
            ))
            #expect(seconds > 10, "expected a substantial passage, got \(seconds)s")
        }
        moss.codebookCount = nil
    }
}
