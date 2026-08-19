// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "MLXAudio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        // Core foundation library
        .library(name: "MLXAudioCore", targets: ["MLXAudioCore"]),

        // Audio codec implementations
        .library(name: "MLXAudioCodecs", targets: ["MLXAudioCodecs"]),

        // Text-to-Speech
        .library(name: "MLXAudioTTS", targets: ["MLXAudioTTS"]),

        // Speech-to-Text (placeholder)
        .library(name: "MLXAudioSTT", targets: ["MLXAudioSTT"]),

        // Voice Activity Detection / Speaker Diarization
        .library(name: "MLXAudioVAD", targets: ["MLXAudioVAD"]),

        // Language Identification
        .library(name: "MLXAudioLID", targets: ["MLXAudioLID"]),

        // Speech-to-Speech
        .library(name: "MLXAudioSTS", targets: ["MLXAudioSTS"]),

        // SwiftUI components
        .library(name: "MLXAudioUI", targets: ["MLXAudioUI"]),

        // Legacy combined library (for backwards compatibility)
        .library(
            name: "MLXAudio",
            targets: ["MLXAudioCore", "MLXAudioCodecs", "MLXAudioTTS", "MLXAudioSTT", "MLXAudioVAD", "MLXAudioLID", "MLXAudioSTS", "MLXAudioUI"]
        ),
        .executable(
            name: "mlx-audio-swift-tts",
            targets: ["mlx-audio-swift-tts"],
        ),
        .executable(
            name: "mlx-audio-swift-codec",
            targets: ["mlx-audio-swift-codec"],
        ),
        .executable(
            name: "mlx-audio-swift-sts",
            targets: ["mlx-audio-swift-sts"],
        ),
        .executable(
            name: "mlx-audio-swift-stt",
            targets: ["mlx-audio-swift-stt"],
        ),
        .executable(
            name: "mlx-audio-swift-lid",
            targets: ["mlx-audio-swift-lid"],
        ),

    ],
    dependencies: [
        // The background-safe fork, not upstream. Upstream publishes the same
        // version tags, so when both URLs appear in one graph SwiftPM resolves
        // the shared `mlx-swift` identity to whichever it sees first — and a
        // consumer can silently end up on upstream, losing the Metal
        // check_error patch that keeps the app alive when iOS revokes GPU
        // access on backgrounding. Naming the fork here removes one of the two
        // conflicting declarations. See Docs/mlx-swift-bg-safe-fork.md in TTSMLX.
        .package(url: "https://github.com/fil-technology/mlx-swift.git", .upToNextMinor(from: "0.31.5")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMajor(from: "2.30.3")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.8.1"))
    ],
    targets: [
        // MARK: - MLXAudioCore
        .target(
            name: "MLXAudioCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioCore",
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-warn-concurrency"], .when(configuration: .debug))
            ]
        ),

        // MARK: - MLXAudioCodecs
        .target(
            name: "MLXAudioCodecs",
            dependencies: [
                "MLXAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioCodecs"
        ),

        // MARK: - MLXAudioTTS
        .target(
            name: "MLXAudioTTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioTTS",
            resources: [
                // Pre-encoded MOSS-TTS-Nano reference clips; see MossVoicePack.
                .copy("Resources/MossVoices")
            ]
        ),

        // MARK: - MLXAudioSTT
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioSTT"
        ),

        // MARK: - MLXAudioVAD
        .target(
            name: "MLXAudioVAD",
            dependencies: [
                "MLXAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioVAD"
        ),

        // MARK: - MLXAudioLID
        .target(
            name: "MLXAudioLID",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioLID"
        ),

        // MARK: - MLXAudioSTS
        .target(
            name: "MLXAudioSTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioTTS",
                "MLXAudioSTT",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioSTS"
        ),

        // MARK: - MLXAudioUI
        .target(
            name: "MLXAudioUI",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioTTS",
                "MLXAudioSTS",
            ],
            path: "Sources/MLXAudioUI"
        ),
        
        .executableTarget(
            name: "mlx-audio-swift-tts",
            dependencies: ["MLXAudioCore", "MLXAudioTTS", "MLXAudioSTT"],
            path: "Sources/Tools/mlx-audio-swift-tts"
        ),
        .executableTarget(
            name: "mlx-audio-swift-codec",
            dependencies: ["MLXAudioCore", "MLXAudioCodecs"],
            path: "Sources/Tools/mlx-audio-swift-codec"
        ),
        .executableTarget(
            name: "mlx-audio-swift-sts",
            dependencies: ["MLXAudioCore", "MLXAudioSTS"],
            path: "Sources/Tools/mlx-audio-swift-sts"
        ),
        .executableTarget(
            name: "mlx-audio-swift-stt",
            dependencies: ["MLXAudioCore", "MLXAudioSTT"],
            path: "Sources/Tools/mlx-audio-swift-stt"
        ),
        .executableTarget(
            name: "mlx-audio-swift-lid",
            dependencies: ["MLXAudioCore", "MLXAudioLID"],
            path: "Sources/Tools/mlx-audio-swift-lid"
        ),

        // MARK: - Tests
        .testTarget(
            name: "MLXAudioTests",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioTTS",
                "MLXAudioSTT",
                "MLXAudioVAD",
                "MLXAudioSTS",
                "MLXAudioLID",
                "mlx-audio-swift-lid",
            ],
            path: "Tests",
            resources: [
                .copy("media")
            ]
        ),
    ]
)
