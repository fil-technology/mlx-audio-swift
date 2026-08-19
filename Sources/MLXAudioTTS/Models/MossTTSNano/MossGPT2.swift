import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon

/// GPT-2 backbone used by both the global and local MOSS-TTS-Nano transformers.
///
/// Ported from `mlx_audio/tts/models/moss_tts_nano/gpt2.py` (Apache-2.0).
/// Differences from stock GPT-2 that matter for weight loading:
///   * `c_attn` / `c_proj` are `Linear` (the MLX conversion already transposed
///     HF's `Conv1D`), so no transpose is needed at load time.
///   * The MLP is named `fc_in` / `fc_out`, not `c_fc` / `c_proj`.
///   * Positions are RoPE by default, so `wpe` is absent.

/// `gelu_new` — the tanh approximation. `MLXNN.geluApproximate` uses the same
/// formula, but is spelled out here to keep parity with the reference obvious.
@inline(__always)
func mossGELUNew(_ x: MLXArray) -> MLXArray {
    let c: Float = 0.7978845608028654  // sqrt(2/pi)
    return 0.5 * x * (1.0 + MLX.tanh(c * (x + 0.044715 * (x * x * x))))
}

final class MossGPT2Attention: Module {
    let numHeads: Int
    let headDim: Int
    let embedDim: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    let rope: RoPE?

    init(_ config: MossGPT2Configuration, layerIndex: Int) {
        precondition(config.nEmbd % config.nHead == 0,
                     "n_embd=\(config.nEmbd) must be divisible by n_head=\(config.nHead)")
        self.embedDim = config.nEmbd
        self.numHeads = config.nHead
        self.headDim = config.nEmbd / config.nHead

        var scale = config.scaleAttnWeights ? powf(Float(headDim), -0.5) : 1.0
        if config.scaleAttnByInverseLayerIdx {
            scale /= Float(layerIndex + 1)
        }
        self.scale = scale

        self._cAttn.wrappedValue = Linear(config.nEmbd, 3 * config.nEmbd, bias: true)
        self._cProj.wrappedValue = Linear(config.nEmbd, config.nEmbd, bias: true)

        // `traditional: true` is the interleaved (even/odd) rotation that the
        // reference `rotate_half` + `repeat(..., 2)` pair implements.
        self.rope = config.usesRoPE
            ? RoPE(dimensions: config.nEmbd / config.nHead, traditional: true, base: config.ropeBase)
            : nil
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, queryLength) = (x.dim(0), x.dim(1))

        let qkv = cAttn(x)
        let parts = MLX.split(qkv, parts: 3, axis: -1)
        var queries = parts[0].reshaped(batch, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(batch, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(batch, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        let (k, v): (MLXArray, MLXArray)
        if let cache {
            (k, v) = cache.update(keys: keys, values: values)
        } else {
            (k, v) = (keys, values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, queryLength, embedDim)

        return cProj(output)
    }
}

final class MossGPT2MLP: Module {
    @ModuleInfo(key: "fc_in") var fcIn: Linear
    @ModuleInfo(key: "fc_out") var fcOut: Linear
    let activation: String

    init(_ config: MossGPT2Configuration) {
        let inner = config.nInner > 0 ? config.nInner : 4 * config.nEmbd
        self._fcIn.wrappedValue = Linear(config.nEmbd, inner, bias: true)
        self._fcOut.wrappedValue = Linear(inner, config.nEmbd, bias: true)
        self.activation = config.activationFunction
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fcIn(x)
        switch activation {
        case "gelu_new": h = mossGELUNew(h)
        case "silu": h = MLXNN.silu(h)
        default: h = MLXNN.gelu(h)
        }
        return fcOut(h)
    }
}

final class MossGPT2Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "attn") var attn: MossGPT2Attention
    @ModuleInfo(key: "mlp") var mlp: MossGPT2MLP

    init(_ config: MossGPT2Configuration, layerIndex: Int) {
        self._ln1.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
        self._ln2.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
        self._attn.wrappedValue = MossGPT2Attention(config, layerIndex: layerIndex)
        self._mlp.wrappedValue = MossGPT2MLP(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + attn(ln1(x), mask: mask, cache: cache)
        h = h + mlp(ln2(h))
        return h
    }
}

final class MossGPT2Model: Module {
    let config: MossGPT2Configuration
    let usesTokenEmbedding: Bool

    @ModuleInfo(key: "wte") var wte: Embedding?
    @ModuleInfo(key: "wpe") var wpe: Embedding?
    @ModuleInfo(key: "h") var blocks: [MossGPT2Block]
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    init(_ config: MossGPT2Configuration, usesTokenEmbedding: Bool = true) {
        self.config = config
        self.usesTokenEmbedding = usesTokenEmbedding
        self._wte.wrappedValue = usesTokenEmbedding
            ? Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
            : nil
        self._wpe.wrappedValue = config.usesAbsolutePositions
            ? Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbd)
            : nil
        self._blocks.wrappedValue = (0 ..< config.nLayer).map { MossGPT2Block(config, layerIndex: $0) }
        self._lnF.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
        super.init()
    }

    func makeCache() -> [KVCache] {
        (0 ..< config.nLayer).map { _ in KVCacheSimple() }
    }

    /// - Parameter inputEmbeddings: pre-built `[batch, seq, nEmbd]` embeddings.
    ///   MOSS always supplies these because a row mixes one text token with
    ///   `n_vq` audio-channel tokens.
    func callAsFunction(
        inputEmbeddings: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = inputEmbeddings

        if let wpe {
            let offset = cache?.first?.offset ?? 0
            let positions = MLXArray(Int32(offset) ..< Int32(offset + h.dim(1)))
            h = h + wpe(positions)
        }

        // The reference builds an explicit additive causal mask. With a single
        // un-padded sequence that is exactly `.causal`; a one-token decode step
        // sees every cached key, which is `.none`.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = h.dim(1) > 1 ? .causal : .none

        for (index, block) in blocks.enumerated() {
            h = block(h, mask: mask, cache: cache?[index])
        }
        return lnF(h)
    }
}
