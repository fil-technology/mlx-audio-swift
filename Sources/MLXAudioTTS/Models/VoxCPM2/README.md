# VoxCPM2 TTS

Autoregressive MiniCPM backbone + CFM diffusion + AudioVAE decoder. Produces 48kHz speech with voice cloning from a reference audio clip. 2B parameters, supports 30+ languages.

## Supported Models

- [mlx-community/VoxCPM2-4bit](https://huggingface.co/mlx-community/VoxCPM2-4bit) (4-bit quantized, ~2.3GB)

## Swift Example

```swift
import MLXAudioTTS
import MLXAudioCore

// Load model
let model = try await VoxCPM2Model.fromPretrained("mlx-community/VoxCPM2-4bit")

// Voice cloning requires reference audio
let (_, refAudio) = try loadAudioArray(from: referenceAudioURL, sampleRate: model.sampleRate)
let audio = try await model.generate(
    text: "Hello, this is a test of VoxCPM2.",
    voice: nil, refAudio: refAudio, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 100, temperature: 1.0)
)
// audio is an MLXArray of Float32 samples at 48kHz
```

### Zero-Shot (No Reference)

```swift
let audio = try await model.generate(
    text: "Zero-shot generation without a reference voice.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 100, temperature: 1.0)
)
```

## Prompt modes and per-call parameters

`VoxCPM2Prompt` carries what the model hears before the text, matching the
upstream `build_prompt_cache` modes:

- `referenceAudio` alone: isolated reference (`ref_audio` tokens); the clip
  lends its timbre, a style description in parentheses at the front of the
  text steers delivery ("controllable cloning").
- `promptAudio` + `promptText`: continuation; the model continues the clip as
  if the text followed its exact transcript ("ultimate cloning").
- all three: continuation with a separate timbre reference.

`VoxCPM2GenerationParameters` sets `inferenceTimesteps` (checkpoint default
10), `cfgValue` (2.0), `seed`, `maxTokens` (also bounded to
`textTokens * 6 + 10` patches, upstream's bad-case guard) and the streaming
window. The protocol `generate`/`generateStream` map `refAudio` alone to the
reference mode and `refAudio` + `refText` to continuation; set
`callParameters` on the model for the other knobs.

```swift
let audio = try await model.generate(
    text: "(A calm, deep narrator)Hello, this is VoxCPM2.",
    prompt: VoxCPM2Prompt(promptAudio: clip, promptText: "Exactly what the clip says."),
    parameters: VoxCPM2GenerationParameters(inferenceTimesteps: 16, cfgValue: 2.0, seed: 42)
)
```

## Streaming Example

`generateStream` yields one patch of audio (160 ms at 48 kHz) per LM step.
Each new patch is decoded together with the previous `streamingWindow - 1`
patches (prompt patches at the start of a continuation) and only its own
samples are emitted, so the causal decoder has context and nothing overlaps.
First audio is one LM step after prefill instead of the whole take, and the
decoder never holds the full take, which lowers the peak memory.

```swift
for try await event in model.generateStream(
    text: "Streaming test.", prompt: VoxCPM2Prompt(referenceAudio: refAudio),
    parameters: VoxCPM2GenerationParameters()
) {
    switch event {
    case .audio(let samples):
        // 7 680 samples of 48 kHz Float32 per event
        break
    case .info(let info):
        print("first audio after \(info.prefillTime)s, done in \(info.generateTime)s")
    case .token, .progress:
        break
    }
}
```

## Output Format

- **Sample rate**: 48kHz
- **Format**: Mono Float32 PCM
- **Reference input rate**: 48kHz mono; pass `sampleRate: model.sampleRate` when loading a file.
- **VAE encoder rate**: 16kHz (internal; the model resamples reference input from 48kHz).

## Known Limitations

- The 4-bit quantized stop predictor can be unreliable; generation is bounded
  to `textTokens * 6 + 10` patches, and `maxTokens` can tighten it further.
- Short prompts (especially in Chinese) may have text-following issues, consistent with the upstream Python implementation.
- The streaming decode re-decodes a window of patches per step (about a
  quarter more work than one whole-take decode); a stateful causal-conv
  decoder would remove that.
- Text is fixed per call: there is no incremental text input.
