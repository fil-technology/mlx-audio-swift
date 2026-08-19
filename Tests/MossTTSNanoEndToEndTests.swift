import Foundation
import Testing
@preconcurrency import MLX
import MLXAudioCore
@testable import MLXAudioTTS

/// End-to-end: load through the registry exactly as an app would, generate a
/// news-style passage with a bundled voice, and check the waveform is real
/// audio of a plausible length.
///
/// Writes the result to `MOSS_E2E_OUTPUT` (or a temp file) so it can be
/// listened to; the assertions here are structural, since sampling makes exact
/// waveform comparison meaningless.
struct MossTTSNanoEndToEndTests {
    static let newsText = """
        Global markets closed higher on Tuesday after the central bank signalled \
        it would hold interest rates steady through the end of the year. \
        Analysts said the decision eased fears of a prolonged slowdown, though \
        several cautioned that inflation remains above target.
        """

    @Test func generatesNewsAudioThroughRegistry() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }

        let model = try await TTS.loadModel(modelRepo: "mlx-community/MOSS-TTS-Nano-100M")
        #expect(model.sampleRate == 48000)

        guard let moss = model as? MossTTSNanoModel else {
            Issue.record("registry returned \(type(of: model)) instead of MossTTSNanoModel")
            return
        }
        #expect(moss.availableVoices.contains("en_3"), "voices: \(moss.availableVoices)")

        MLX.GPU.resetPeakMemory()
        let started = Date()
        let audio = try await moss.generate(
            text: Self.newsText, voice: "en_3", refAudio: nil, refText: nil, language: "English",
            generationParameters: moss.defaultGenerationParameters
        )
        audio.eval()
        let elapsed = Date().timeIntervalSince(started)

        #expect(audio.ndim == 2, "expected [samples, channels], got \(audio.shape)")
        #expect(audio.dim(1) == 2, "MOSS decodes 48 kHz stereo")

        let seconds = Double(audio.dim(0)) / 48000.0
        // ~250 characters of news copy should land in a broad but sane window.
        #expect(seconds > 4.0 && seconds < 45.0, "generated \(seconds)s")

        let channel0 = audio[0..., 0].asType(.float32)
        let rms = MLX.sqrt((channel0 * channel0).mean()).item(Float.self)
        #expect(rms > 0.005, "audio is silent (rms \(rms))")
        #expect(rms < 1.0, "audio is clipping (rms \(rms))")

        let peak = MLX.abs(channel0).max().item(Float.self)
        #expect(peak <= 1.05, "peak \(peak) exceeds full scale")

        let destination = ProcessInfo.processInfo.environment["MOSS_E2E_OUTPUT"]
            .map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("moss_e2e.wav")
        try Self.writeStereoWAV(audio, sampleRate: 48000, to: destination)
        let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576.0
        print("[moss-e2e] \(seconds)s of audio in \(elapsed)s (RTF \(seconds / elapsed)x), peak \(peakMB) MB -> \(destination.path)")
    }

    /// Minimal 16-bit PCM stereo WAV writer. `saveAudioArray` in MLXAudioCore
    /// is mono-only and not public, and MOSS decodes to 48 kHz stereo.
    static func writeStereoWAV(_ audio: MLXArray, sampleRate: Int, to url: URL) throws {
        let frames = audio.dim(0)
        let channels = audio.dim(1)
        let interleaved = audio.asType(.float32).asArray(Float.self)

        var data = Data()
        func appendLE<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

        let byteRate = sampleRate * channels * 2
        let dataBytes = frames * channels * 2

        data.append(contentsOf: Array("RIFF".utf8))
        appendLE(UInt32(36 + dataBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendLE(UInt32(16))
        appendLE(UInt16(1))                       // PCM
        appendLE(UInt16(channels))
        appendLE(UInt32(sampleRate))
        appendLE(UInt32(byteRate))
        appendLE(UInt16(channels * 2))            // block align
        appendLE(UInt16(16))                      // bits per sample
        data.append(contentsOf: Array("data".utf8))
        appendLE(UInt32(dataBytes))

        data.reserveCapacity(data.count + dataBytes)
        for sample in interleaved {
            let clamped = max(-1.0, min(1.0, sample))
            appendLE(Int16(clamped * 32767.0))
        }
        try data.write(to: url)
    }

    /// The streaming path decodes one chunk at a time. That matters a lot for
    /// peak memory: the codec's deepest decoder stage attends over
    /// `frames * 32` positions, so decoding a whole passage in one shot costs
    /// far more than decoding it sentence by sentence.
    @Test func streamingKeepsPeakMemoryLower() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }
        let model = try await TTS.loadModel(modelRepo: "mlx-community/MOSS-TTS-Nano-100M")
        guard let moss = model as? MossTTSNanoModel else {
            Issue.record("registry returned \(type(of: model))")
            return
        }

        MLX.GPU.resetPeakMemory()
        let started = Date()
        var totalFrames = 0
        var chunks = 0
        var firstChunkAt: TimeInterval?

        let stream = moss.generateStream(
            text: Self.newsText, voice: "en_3", refAudio: nil, refText: nil,
            language: "English", generationParameters: moss.defaultGenerationParameters,
            streamingInterval: 2.0
        )
        for try await generation in stream {
            guard case .audio(let samples) = generation else { continue }
            samples.eval()
            if firstChunkAt == nil { firstChunkAt = Date().timeIntervalSince(started) }
            totalFrames += samples.dim(0)
            chunks += 1
        }

        let elapsed = Date().timeIntervalSince(started)
        let seconds = Double(totalFrames) / 48000.0
        let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576.0
        print("[moss-stream] \(chunks) chunks, \(seconds)s audio in \(elapsed)s "
              + "(RTF \(seconds / elapsed)x), first chunk at \(firstChunkAt ?? -1)s, peak \(peakMB) MB")

        #expect(chunks > 0, "stream produced no audio")
        #expect(seconds > 4.0, "stream produced only \(seconds)s")
    }

    /// Measures how the chunk budget trades time-to-first-audio and peak
    /// memory against itself. Informational: it prints a table rather than
    /// asserting thresholds, since absolute numbers are machine-dependent.
    @Test func chunkBudgetSweep() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }
        let model = try await TTS.loadModel(modelRepo: "mlx-community/MOSS-TTS-Nano-100M")
        guard let moss = model as? MossTTSNanoModel else { return }

        for budget in [20, 40, 75] {
            moss.maxTextTokensPerChunk = budget
            MLX.GPU.resetPeakMemory()
            let started = Date()
            var chunks = 0
            var frames = 0
            var firstAt: TimeInterval?

            let stream = moss.generateStream(
                text: Self.newsText, voice: "en_3", refAudio: nil, refText: nil,
                language: "English", generationParameters: moss.defaultGenerationParameters,
                streamingInterval: 2.0
            )
            for try await generation in stream {
                guard case .audio(let samples) = generation else { continue }
                samples.eval()
                if firstAt == nil { firstAt = Date().timeIntervalSince(started) }
                frames += samples.dim(0)
                chunks += 1
            }
            let elapsed = Date().timeIntervalSince(started)
            let peakMB = Double(MLX.GPU.peakMemory) / 1_048_576.0
            print(String(
                format: "[moss-sweep] budget=%3d chunks=%2d first=%5.2fs total=%5.2fs audio=%5.2fs peak=%7.1fMB",
                budget, chunks, firstAt ?? -1, elapsed, Double(frames) / 48000.0, peakMB
            ))
        }
        moss.maxTextTokensPerChunk = 75
    }

    /// TTSMLX's Reader path builds synthesis options without a voice, so a nil
    /// voice must still produce audio rather than throwing.
    @Test func nilVoiceFallsBackToBundledDefault() async throws {
        guard await MossTTSNanoTokenizerTests.resolveModelDirectory() != nil else {
            #expect(!MossTTSNanoTokenizerTests.requiresFixtures,
                    "MOSS weights unavailable and MOSS_REQUIRE_FIXTURES=1")
            return
        }
        let model = try await TTS.loadModel(modelRepo: "mlx-community/MOSS-TTS-Nano-100M")
        guard let moss = model as? MossTTSNanoModel else { return }

        #expect(moss.availableVoices.contains(MossTTSNanoModel.defaultVoice))

        let audio = try await moss.generate(
            text: "Markets closed higher on Tuesday.", voice: nil, refAudio: nil,
            refText: nil, language: "English",
            generationParameters: moss.defaultGenerationParameters
        )
        audio.eval()
        #expect(audio.dim(0) > 0, "nil voice produced no audio")
        #expect(audio.dim(1) == 2)
    }
}
