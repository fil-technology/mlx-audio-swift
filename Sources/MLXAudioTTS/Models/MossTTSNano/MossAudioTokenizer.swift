import Foundation
@preconcurrency import MLX
import MLXNN
import HuggingFace
import MLXAudioCore

/// MOSS-Audio-Tokenizer-Nano — **decode path**.
///
/// Ported from `mlx_audio/codec/models/moss_audio_tokenizer` (Apache-2.0).
/// This is the CAT ("causal audio tokenizer with transformer") codec: a
/// CNN-free stack of causal transformers interleaved with patch reshapes that
/// turn 12.5 Hz RVQ codes into 48 kHz stereo.
///
/// Only decoding is implemented. Encoding is what voice *cloning from new
/// audio* needs; named voices instead ship as pre-encoded codes (see
/// ``MossVoicePack``), so the encoder is not on the device critical path.

/// The MLX conversion of MOSS-Audio-Tokenizer-Nano.
///
/// The model's own `config.json` names the PyTorch build
/// (`OpenMOSS-Team/...`), whose remote-code weights this loader cannot read,
/// so that value is honoured only when it already points at an MLX
/// conversion — see the registry entry.
public let mossDefaultAudioTokenizerRepo = "mlx-community/MOSS-Audio-Tokenizer-Nano"

// MARK: - Configuration

public struct MossCodecStage: Decodable, Sendable {
    public var moduleType: String
    public var patchSize: Int?
    public var inputDimension: Int?
    public var outputDimension: Int?
    public var dModel: Int?
    public var numHeads: Int?
    public var numLayers: Int?
    public var dimFeedforward: Int?
    public var causal: Bool?
    public var contextDuration: Double?
    public var layerScale: Float?
    public var maxPeriod: Float?
    public var positionalEmbedding: String?
    public var gating: String?
    public var norm: String?

    enum CodingKeys: String, CodingKey {
        case moduleType = "module_type"
        case patchSize = "patch_size"
        case inputDimension = "input_dimension"
        case outputDimension = "output_dimension"
        case dModel = "d_model"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case dimFeedforward = "dim_feedforward"
        case causal
        case contextDuration = "context_duration"
        case layerScale = "layer_scale"
        case maxPeriod = "max_period"
        case positionalEmbedding = "positional_embedding"
        case gating
        case norm
    }

    /// How much this stage changes the frame rate.
    var downsampleRatio: Int { moduleType == "PatchedPretransform" ? (patchSize ?? 1) : 1 }
}

public struct MossQuantizerConfiguration: Decodable, Sendable {
    public var inputDim: Int = 768
    public var rvqDim: Int = 512
    public var outputDim: Int = 768
    public var numQuantizers: Int = 16
    public var codebookSize: Int = 1024
    public var codebookDim: Int = 8
    public var quantizerType: String = "rlfq"

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case rvqDim = "rvq_dim"
        case outputDim = "output_dim"
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case quantizerType = "quantizer_type"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MossQuantizerConfiguration()
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? d.inputDim
        rvqDim = try c.decodeIfPresent(Int.self, forKey: .rvqDim) ?? d.rvqDim
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? d.outputDim
        numQuantizers = try c.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? d.numQuantizers
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? d.codebookSize
        codebookDim = try c.decodeIfPresent(Int.self, forKey: .codebookDim) ?? d.codebookDim
        quantizerType = try c.decodeIfPresent(String.self, forKey: .quantizerType) ?? d.quantizerType
    }

    public init() {}
}

public struct MossAudioTokenizerConfiguration: Decodable, Sendable {
    public var sampleRate: Int = 48000
    public var downsampleRate: Int = 3840
    public var contextDuration: Double = 10.0
    public var numberChannels: Int = 2
    public var enableChannelInterleave: Bool = true
    public var encoderStages: [MossCodecStage] = []
    public var decoderStages: [MossCodecStage] = []
    public var quantizer: MossQuantizerConfiguration = .init()

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case samplingRate = "sampling_rate"
        case downsampleRate = "downsample_rate"
        case contextDuration = "causal_transformer_context_duration"
        case numberChannels = "number_channels"
        case enableChannelInterleave = "enable_channel_interleave"
        case encoderStages = "encoder_kwargs"
        case decoderStages = "decoder_kwargs"
        case quantizer = "quantizer_kwargs"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = MossAudioTokenizerConfiguration()
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate)
            ?? c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? d.sampleRate
        downsampleRate = try c.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? d.downsampleRate
        contextDuration = try c.decodeIfPresent(Double.self, forKey: .contextDuration) ?? d.contextDuration
        numberChannels = try c.decodeIfPresent(Int.self, forKey: .numberChannels) ?? d.numberChannels
        enableChannelInterleave = try c.decodeIfPresent(Bool.self, forKey: .enableChannelInterleave) ?? d.enableChannelInterleave
        encoderStages = try c.decodeIfPresent([MossCodecStage].self, forKey: .encoderStages) ?? []
        decoderStages = try c.decodeIfPresent([MossCodecStage].self, forKey: .decoderStages) ?? []
        quantizer = try c.decodeIfPresent(MossQuantizerConfiguration.self, forKey: .quantizer) ?? d.quantizer
    }

    public init() {}
}

// MARK: - Layers

/// A 1-D convolution whose weight-norm parametrisation has already been folded
/// at load time (see `MossAudioTokenizerModel.sanitize`), so this is a plain
/// conv. Every instance in this codec uses `kernel_size == 1`.
final class MossWNConv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray  // [out, kernel, in]
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 1) {
        self._weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }

    /// - Parameter x: `[batch, channels, length]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1d(x.transposed(0, 2, 1), weight, stride: 1, padding: 0, dilation: 1, groups: 1)
        return (y + bias).transposed(0, 2, 1)
    }
}

final class MossLayerScale: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(channels: Int, initialValue: Float) {
        self._scale.wrappedValue = MLXArray.full([channels], values: MLXArray(initialValue))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { scale * x }
}

/// The exact (erf-based) GELU. The codec uses this, not the tanh approximation
/// the TTS backbone uses — mixing them up shifts every decoded sample slightly.
@inline(__always)
func mossExactGELU(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1.0 + MLX.erf(x / 1.4142135623730951))
}

final class MossCodecAttention: Module {
    let numHeads: Int
    let headDim: Int
    let embedDim: Int
    let causal: Bool
    let context: Int?
    let useRoPE: Bool
    let rope: RoPE?

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(
        embedDim: Int, numHeads: Int, causal: Bool, context: Int?,
        maxPeriod: Float, useRoPE: Bool
    ) {
        precondition(embedDim % numHeads == 0, "embed_dim must be divisible by num_heads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.causal = causal
        self.context = context
        self.useRoPE = useRoPE
        self._inProj.wrappedValue = Linear(embedDim, 3 * embedDim, bias: false)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: false)
        // The codec's `_apply_rope` rotates adjacent pairs, i.e. traditional.
        self.rope = useRoPE
            ? RoPE(dimensions: embedDim / numHeads, traditional: true, base: maxPeriod)
            : nil
        super.init()
    }

    /// Boolean mask of shape `[1, 1, time, time]`.
    ///
    /// Built as `Bool` rather than an additive float mask: at the deepest
    /// decoder stage `time` runs into the thousands, and a float32 mask there
    /// costs 4x the memory for no benefit. Batch is always 1 for decode, so the
    /// reference's `valid_k` term (padding) is uniformly true and drops out.
    private func attentionMask(time: Int) -> MLXArray? {
        guard causal || context != nil else { return nil }
        let positions = MLXArray(Int32(0) ..< Int32(time))
        let delta = positions[0..., .newAxis] - positions[.newAxis, 0...]
        var allowed = MLXArray(true)
        if causal {
            allowed = delta .>= MLXArray(Int32(0))
        }
        if let context {
            let within = delta .< MLXArray(Int32(context))
            allowed = causal ? (allowed & within) : within
        }
        return allowed.reshaped(1, 1, time, time)
    }

    /// - Parameter x: `[batch, time, embedDim]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, time) = (x.dim(0), x.dim(1))
        // in_proj packs [3, heads, headDim] per position, in that order.
        let qkv = inProj(x).reshaped(batch, time, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)

        if let rope {
            q = rope(q)
            k = rope(k)
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask = attentionMask(time: time) {
            maskMode = .array(mask)
        } else {
            maskMode = .none
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: powf(Float(headDim), -0.5), mask: maskMode
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, time, embedDim)
        return outProj(out)
    }
}

final class MossCodecTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MossCodecAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: [Linear]
    @ModuleInfo(key: "layer_scale_1") var layerScale1: MossLayerScale?
    @ModuleInfo(key: "layer_scale_2") var layerScale2: MossLayerScale?

    init(
        dModel: Int, numHeads: Int, dimFeedforward: Int, causal: Bool, context: Int?,
        positionalEmbedding: String, maxPeriod: Float, layerScale: Float?
    ) {
        self._selfAttn.wrappedValue = MossCodecAttention(
            embedDim: dModel, numHeads: numHeads, causal: causal, context: context,
            maxPeriod: maxPeriod,
            useRoPE: positionalEmbedding == "rope" || positionalEmbedding == "sin_rope"
        )
        self._norm1.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        // Index 1 is an Identity in the reference; keeping the two Linears at
        // indices 0 and 2 preserves the checkpoint's `ffn.0` / `ffn.2` keys.
        self._ffn.wrappedValue = [
            Linear(dModel, dimFeedforward, bias: false),
            Linear(dimFeedforward, dModel, bias: false),
        ]
        self._layerScale1.wrappedValue = layerScale.map { MossLayerScale(channels: dModel, initialValue: $0) }
        self._layerScale2.wrappedValue = layerScale.map { MossLayerScale(channels: dModel, initialValue: $0) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = norm1(x)
        let attended = selfAttn(h)
        h = residual + (layerScale1.map { $0(attended) } ?? attended)

        residual = h
        var f = ffn[1](mossExactGELU(ffn[0](norm2(h))))
        f = layerScale2.map { $0(f) } ?? f
        return residual + f
    }
}

final class MossCodecTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MossCodecTransformerLayer]

    init(
        dModel: Int, numHeads: Int, numLayers: Int, dimFeedforward: Int,
        causal: Bool, context: Int?, positionalEmbedding: String,
        maxPeriod: Float, layerScale: Float?
    ) {
        self._layers.wrappedValue = (0 ..< numLayers).map { _ in
            MossCodecTransformerLayer(
                dModel: dModel, numHeads: numHeads, dimFeedforward: dimFeedforward,
                causal: causal, context: context, positionalEmbedding: positionalEmbedding,
                maxPeriod: maxPeriod, layerScale: layerScale
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }
}

/// Channel-first in, channel-first out; the transformer itself works in
/// `[batch, time, channels]`.
final class MossProjectedTransformer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "transformer") var transformer: MossCodecTransformer
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

    init(
        inputDimension: Int, outputDimension: Int, dModel: Int, numHeads: Int,
        numLayers: Int, dimFeedforward: Int, causal: Bool, context: Int?,
        positionalEmbedding: String, maxPeriod: Float, layerScale: Float?,
        forceInputProjection: Bool, forceOutputProjection: Bool
    ) {
        self._inputProj.wrappedValue = (forceInputProjection || inputDimension != dModel)
            ? Linear(inputDimension, dModel, bias: false) : nil
        self._transformer.wrappedValue = MossCodecTransformer(
            dModel: dModel, numHeads: numHeads, numLayers: numLayers,
            dimFeedforward: dimFeedforward, causal: causal, context: context,
            positionalEmbedding: positionalEmbedding, maxPeriod: maxPeriod,
            layerScale: layerScale
        )
        self._outputProj.wrappedValue = (forceOutputProjection || outputDimension != dModel)
            ? Linear(dModel, outputDimension, bias: false) : nil
        super.init()
    }

    /// - Parameter x: `[batch, channels, time]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        if let inputProj { h = inputProj(h) }
        h = transformer(h)
        if let outputProj { h = outputProj(h) }
        return h.transposed(0, 2, 1)
    }
}

// MARK: - Quantizer (decode only)

final class MossLFQ: Module {
    @ModuleInfo(key: "out_proj") var outProj: MossWNConv1d
    @ModuleInfo(key: "codebook") var codebook: Embedding
    // `in_proj` exists in the checkpoint but is encode-only; it is dropped in
    // sanitize so the module tree stays minimal.

    init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self._outProj.wrappedValue = MossWNConv1d(inChannels: codebookDim, outChannels: inputDim)
        self._codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        super.init()
    }

    /// - Parameter embedID: `[batch, time]` code indices.
    func decodeCode(_ embedID: MLXArray) -> MLXArray {
        outProj(codebook(embedID).transposed(0, 2, 1).asType(.float32))
    }
}

final class MossResidualLFQ: Module {
    let rvqDim: Int
    @ModuleInfo(key: "quantizers") var quantizers: [MossLFQ]
    @ModuleInfo(key: "output_proj") var outputProj: MossWNConv1d
    // `input_proj` is encode-only and dropped in sanitize.

    init(_ config: MossQuantizerConfiguration) {
        self.rvqDim = config.rvqDim
        self._quantizers.wrappedValue = (0 ..< config.numQuantizers).map { _ in
            MossLFQ(
                inputDim: config.rvqDim,
                codebookSize: config.codebookSize,
                codebookDim: config.codebookDim
            )
        }
        self._outputProj.wrappedValue = MossWNConv1d(
            inChannels: config.rvqDim, outChannels: config.outputDim
        )
        super.init()
    }

    /// Sums the residual stages back together.
    /// - Parameter codes: `[nq, batch, time]`
    func decodeCodes(_ codes: MLXArray) -> MLXArray {
        let nq = min(codes.dim(0), quantizers.count)
        var embedding = MLXArray.zeros([codes.dim(1), rvqDim, codes.dim(2)], type: Float.self)
        for index in 0 ..< nq {
            embedding = embedding + quantizers[index].decodeCode(codes[index]).asType(.float32)
        }
        return outputProj(embedding).asType(.float32)
    }
}

// MARK: - Model

public final class MossAudioTokenizerModel: Module, MossAudioDecoding, @unchecked Sendable {
    public let config: MossAudioTokenizerConfiguration
    public var sampleRate: Int { config.sampleRate }

    @ModuleInfo(key: "quantizer") var quantizer: MossResidualLFQ
    @ModuleInfo(key: "decoder") var decoder: [MossProjectedTransformer]

    /// Interleaved with `decoder`: patch sizes for the reshape stages, keyed by
    /// their index in the original stage list.
    private let decoderStages: [MossDecoderStage]

    enum MossDecoderStage {
        case patch(Int)
        case transformer(Int)  // index into `decoder`
    }

    public init(_ config: MossAudioTokenizerConfiguration, projectionKeys: Set<String>) {
        self.config = config
        self._quantizer.wrappedValue = MossResidualLFQ(config.quantizer)

        // Frame rate is threaded through the encoder first, because each
        // transformer's attention context is `contextDuration` seconds
        // expressed in frames *at that stage*.
        let channelFactor = (config.enableChannelInterleave && config.numberChannels > 1)
            ? config.numberChannels : 1
        var frameRate = Double(config.sampleRate * channelFactor)
        for stage in config.encoderStages {
            frameRate /= Double(stage.downsampleRatio)
        }

        var transformers: [MossProjectedTransformer] = []
        var stages: [MossDecoderStage] = []
        for (index, stage) in config.decoderStages.enumerated() {
            switch stage.moduleType {
            case "PatchedPretransform":
                stages.append(.patch(stage.patchSize ?? 1))
            case "Transformer":
                let context = Int((frameRate * (stage.contextDuration ?? config.contextDuration)).rounded())
                transformers.append(
                    MossProjectedTransformer(
                        inputDimension: stage.inputDimension ?? 0,
                        outputDimension: stage.outputDimension ?? 0,
                        dModel: stage.dModel ?? 256,
                        numHeads: stage.numHeads ?? 4,
                        numLayers: stage.numLayers ?? 1,
                        dimFeedforward: stage.dimFeedforward ?? 1024,
                        causal: stage.causal ?? true,
                        context: context,
                        positionalEmbedding: stage.positionalEmbedding ?? "rope",
                        maxPeriod: stage.maxPeriod ?? 10000,
                        layerScale: stage.layerScale,
                        forceInputProjection: projectionKeys.contains("decoder.\(index).input_proj.weight"),
                        forceOutputProjection: projectionKeys.contains("decoder.\(index).output_proj.weight")
                    )
                )
                stages.append(.transformer(transformers.count - 1))
            default:
                break
            }
            frameRate *= Double(stage.downsampleRatio)
        }
        self._decoder.wrappedValue = transformers
        self.decoderStages = stages
        super.init()
    }

    // MARK: Decoding

    /// - Parameter audioCodes: `[batch, frames, nq]` or `[frames, nq]`.
    /// - Returns: `[samples, channels]` float waveform.
    public func decodeAudioCodes(_ audioCodes: MLXArray, numQuantizers: Int) throws -> MLXArray {
        var codes = audioCodes.asType(.int32)
        if codes.ndim == 3 {
            guard codes.dim(0) == 1 else {
                throw MossTTSNanoError.audioTokenizerUnavailable("batched decode is not implemented")
            }
            codes = codes[0]
        }
        guard codes.ndim == 2 else {
            throw MossTTSNanoError.invalidPromptCodes("expected [frames, nq], got \(codes.shape)")
        }
        guard codes.dim(0) > 0 else {
            return MLXArray.zeros([0, config.numberChannels], type: Float.self)
        }

        let effectiveNQ = min(numQuantizers, codes.dim(1))
        // [nq, batch=1, time]
        let batched = codes[0..., 0 ..< effectiveNQ].transposed(1, 0).expandedDimensions(axis: 1)

        var hidden = quantizer.decodeCodes(batched)
        for stage in decoderStages {
            switch stage {
            case .patch(let patchSize):
                hidden = Self.patchDecode(hidden, patchSize: patchSize)
            case .transformer(let index):
                hidden = decoder[index](hidden)
            }
        }
        return restoreChannels(hidden)
    }

    /// Inverse of the patch reshape: folds `patch` channels back into time.
    static func patchDecode(_ x: MLXArray, patchSize: Int) -> MLXArray {
        let (batch, channelsPatch, length) = (x.dim(0), x.dim(1), x.dim(2))
        let channels = channelsPatch / patchSize
        return x.reshaped(batch, channels, patchSize, length)
            .transposed(0, 1, 3, 2)
            .reshaped(batch, channels, length * patchSize)
    }

    /// De-interleaves the single codec channel back into stereo.
    private func restoreChannels(_ output: MLXArray) -> MLXArray {
        let channels = config.numberChannels
        guard channels > 1, config.enableChannelInterleave else {
            return output[0].transposed(1, 0).asType(.float32)
        }
        let batch = output.dim(0)
        let deinterleaved = output[0..., 0, 0...].reshaped(batch, -1, channels)
        return deinterleaved[0].asType(.float32)
    }

    // MARK: Loading

    /// Folds the weight-norm parametrisation into a plain kernel and drops the
    /// encode-only tensors.
    ///
    /// `weight = g * v / ||v||` where the norm is taken over every axis except
    /// the output-channel axis. Doing this once at load beats recomputing it on
    /// every forward pass.
    /// Maps a decoder stage's index in the *config* stage list to its index
    /// in our `decoder` module array. The checkpoint numbers all nine stages
    /// (patch reshapes included); we only keep the four transformers.
    static func decoderIndexMap(_ config: MossAudioTokenizerConfiguration) -> [Int: Int] {
        var map: [Int: Int] = [:]
        var next = 0
        for (index, stage) in config.decoderStages.enumerated() where stage.moduleType == "Transformer" {
            map[index] = next
            next += 1
        }
        return map
    }

    public static func sanitize(
        weights: [String: MLXArray],
        decoderIndexMap: [Int: Int] = [:]
    ) -> (weights: [String: MLXArray], projectionKeys: Set<String>) {
        var renamed: [String: MLXArray] = [:]
        for (rawKey, value) in weights {
            var key = rawKey
            key = key.replacingOccurrences(of: ".linear1.weight", with: ".ffn.0.weight")
            key = key.replacingOccurrences(of: ".linear2.weight", with: ".ffn.2.weight")
            key = key.replacingOccurrences(of: ".self_attn.in_projs.0.weight", with: ".self_attn.in_proj.weight")
            key = key.replacingOccurrences(of: ".self_attn.out_projs.0.weight", with: ".self_attn.out_proj.weight")
            renamed[key] = value
        }

        let projectionKeys = Set(renamed.keys.filter {
            $0.hasSuffix(".input_proj.weight") || $0.hasSuffix(".output_proj.weight")
        })

        // Collect weight-norm pairs.
        var folded: [String: MLXArray] = [:]
        var handled = Set<String>()
        for (key, original1) in renamed where key.hasSuffix(".parametrizations.weight.original1") {
            let base = String(key.dropLast(".parametrizations.weight.original1".count))
            let gKey = base + ".parametrizations.weight.original0"
            guard let original0 = renamed[gKey] else { continue }
            // original1 is [out, in, kernel]; norm over all axes but 0.
            let squared = original1.asType(.float32).square()
            let norm = MLX.sqrt(squared.sum(axes: [1, 2], keepDims: true))
            let weight = original0.asType(.float32) * original1.asType(.float32) / norm
            // MLX conv1d wants [out, kernel, in].
            folded[base + ".weight"] = weight.transposed(0, 2, 1)
            handled.insert(key)
            handled.insert(gKey)
        }

        var result: [String: MLXArray] = [:]
        for (key, value) in renamed where !handled.contains(key) {
            result[key] = value
        }
        result.merge(folded) { _, new in new }

        // Drop everything the decode path does not use: the whole encoder, and
        // the quantizer's encode-side projections.
        result = result.filter { key, _ in
            if key.hasPrefix("encoder.") { return false }
            if key.hasPrefix("quantizer.input_proj.") { return false }
            if key.contains(".in_proj.") && key.hasPrefix("quantizer.quantizers.") { return false }
            return true
        }
        // `ffn.2` in the checkpoint is index 1 in our two-element array, and
        // `decoder.<configIndex>` becomes `decoder.<transformerIndex>`.
        var remapped: [String: MLXArray] = [:]
        for (key, value) in result {
            var newKey = key.replacingOccurrences(of: ".ffn.2.", with: ".ffn.1.")
            if newKey.hasPrefix("decoder.") {
                let parts = newKey.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
                if parts.count == 3, let configIndex = Int(parts[1]),
                   let moduleIndex = decoderIndexMap[configIndex] {
                    newKey = "decoder.\(moduleIndex).\(parts[2])"
                }
            }
            remapped[newKey] = value
        }
        return (remapped, projectionKeys)
    }

    public static func fromPretrained(
        _ modelRepo: String = mossDefaultAudioTokenizerRepo,
        cache: HubCache = .default
    ) async throws -> MossAudioTokenizerModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw MossTTSNanoError.audioTokenizerUnavailable("invalid repository ID: \(modelRepo)")
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", hfToken: hfToken, cache: cache
        )

        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MossAudioTokenizerConfiguration.self, from: configData)

        var raw: [String: MLXArray] = [:]
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        for file in contents where file.pathExtension == "safetensors" {
            raw.merge(try MLX.loadArrays(url: file)) { current, _ in current }
        }
        guard !raw.isEmpty else {
            throw MossTTSNanoError.audioTokenizerUnavailable("no weights in \(modelDir.path)")
        }

        let (weights, projectionKeys) = sanitize(
            weights: raw, decoderIndexMap: decoderIndexMap(config)
        )
        let model = MossAudioTokenizerModel(config, projectionKeys: projectionKeys)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
