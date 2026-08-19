import Foundation

/// A self-contained SentencePiece **BPE** tokenizer.
///
/// MOSS-TTS-Nano ships a raw SentencePiece `tokenizer.model` protobuf rather
/// than a `tokenizer.json`, so neither `Tokenizers` nor the Unigram reader in
/// `Models/PocketTTS` can load it. This reads the protobuf directly and
/// implements SentencePiece's BPE merge loop.
///
/// Only the fields MOSS needs are parsed; everything else in the proto is
/// skipped by wire type.
public final class SentencePieceBPETokenizer: @unchecked Sendable {
    public enum PieceType: Int {
        case normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6
    }

    struct Piece {
        let text: String
        let score: Float
        let type: PieceType
    }

    private let pieces: [Piece]
    private let idForPiece: [String: Int]
    private let unknownID: Int
    private let addDummyPrefix: Bool
    private let removeExtraWhitespaces: Bool
    private let byteFallback: Bool
    /// `<0xNN>` piece ids, indexed by byte value, when byte fallback is on.
    private let byteFallbackIDs: [Int]
    /// USER_DEFINED pieces (e.g. `<user_inst>`), longest first. SentencePiece
    /// matches these literally *before* the BPE merge loop runs, so they must
    /// be carved out of the normalized text rather than merged from characters.
    private let userDefinedPieces: [(text: String, id: Int)]

    public var vocabularySize: Int { pieces.count }

    public init(modelPath: URL) throws {
        let data = try Data(contentsOf: modelPath)
        var reader = ProtobufReader(data: data)

        var parsedPieces: [Piece] = []
        var addDummyPrefix = true
        var removeExtraWhitespaces = true
        var modelType: Int? = nil
        var unknownID = 0

        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == 2:  // repeated SentencePiece pieces
                parsedPieces.append(try Self.parsePiece(field.data))
            case 2 where field.wireType == 2:  // TrainerSpec
                var t = ProtobufReader(data: field.data)
                while let tf = try t.nextField() {
                    switch tf.number {
                    case 3 where tf.wireType == 0: modelType = Int(tf.varint)
                    case 40 where tf.wireType == 0: unknownID = Int(tf.varint)
                    default: break
                    }
                }
            case 3 where field.wireType == 2:  // NormalizerSpec
                var n = ProtobufReader(data: field.data)
                while let nf = try n.nextField() {
                    switch nf.number {
                    case 3 where nf.wireType == 0: addDummyPrefix = nf.varint != 0
                    case 4 where nf.wireType == 0: removeExtraWhitespaces = nf.varint != 0
                    default: break
                    }
                }
            default:
                break
            }
        }

        guard !parsedPieces.isEmpty else {
            throw MossTTSNanoError.tokenizerUnavailable("no pieces found in \(modelPath.lastPathComponent)")
        }
        // TrainerSpec.ModelType: UNIGRAM=1, BPE=2, WORD=3, CHAR=4.
        if let modelType, modelType != 2 {
            throw MossTTSNanoError.tokenizerUnavailable(
                "expected a BPE SentencePiece model, got model_type=\(modelType)"
            )
        }

        self.pieces = parsedPieces
        self.addDummyPrefix = addDummyPrefix
        self.removeExtraWhitespaces = removeExtraWhitespaces
        self.unknownID = unknownID

        var map: [String: Int] = [:]
        map.reserveCapacity(parsedPieces.count)
        for (index, piece) in parsedPieces.enumerated() where map[piece.text] == nil {
            map[piece.text] = index
        }
        self.idForPiece = map

        // Byte fallback pieces are spelled `<0x00>` … `<0xFF>`.
        var byteIDs = [Int](repeating: -1, count: 256)
        var sawByte = false
        for (index, piece) in parsedPieces.enumerated() where piece.type == .byte {
            let text = piece.text
            guard text.count == 6, text.hasPrefix("<0x"), text.hasSuffix(">"),
                  let value = UInt8(text.dropFirst(3).dropLast(), radix: 16) else { continue }
            byteIDs[Int(value)] = index
            sawByte = true
        }
        self.byteFallback = sawByte
        self.byteFallbackIDs = byteIDs

        self.userDefinedPieces = parsedPieces.enumerated()
            .filter { $0.element.type == .userDefined && !$0.element.text.isEmpty }
            .map { (text: $0.element.text, id: $0.offset) }
            .sorted { $0.text.count > $1.text.count }
    }

    private static func parsePiece(_ data: Data) throws -> Piece {
        var reader = ProtobufReader(data: data)
        var text = ""
        var score: Float = 0
        var type = PieceType.normal
        while let field = try reader.nextField() {
            switch field.number {
            case 1 where field.wireType == 2:
                text = String(decoding: field.data, as: UTF8.self)
            case 2 where field.wireType == 5:
                score = Float(bitPattern: field.fixed32)
            case 3 where field.wireType == 0:
                type = PieceType(rawValue: Int(field.varint)) ?? .normal
            default:
                break
            }
        }
        return Piece(text: text, score: score, type: type)
    }

    // MARK: - Normalization

    /// Approximates SentencePiece's `nmt_nfkc` normalizer: strip control
    /// characters, fold every whitespace variant to U+0020, apply NFKC, then
    /// optionally collapse whitespace runs and prepend the dummy prefix.
    func normalize(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            // NMT drops these outright rather than mapping them to space.
            if scalar.value == 0x0000 || scalar.value == 0x000B || scalar.value == 0x000C
                || scalar.value == 0x0085 || (scalar.value >= 0x200B && scalar.value <= 0x200F)
                || scalar.value == 0xFEFF || scalar.value == 0x2028 || scalar.value == 0x2029 {
                continue
            }
            if scalar.properties.isWhitespace || scalar.value == 0x09 || scalar.value == 0x0A || scalar.value == 0x0D {
                scalars.append(" ")
            } else {
                scalars.append(scalar)
            }
        }
        var normalized = String(scalars).precomposedStringWithCompatibilityMapping

        if removeExtraWhitespaces {
            while normalized.contains("  ") {
                normalized = normalized.replacingOccurrences(of: "  ", with: " ")
            }
            normalized = normalized.trimmingCharacters(in: .whitespaces)
        }

        // Whitespace-only input yields no pieces at all in SentencePiece —
        // the dummy prefix is not applied to an otherwise empty string. MOSS's
        // prompt template relies on this: its "\n" separator contributes zero
        // tokens, and emitting a stray "▁" there shifts the whole prompt.
        guard !normalized.isEmpty else { return "" }

        if addDummyPrefix, !normalized.hasPrefix(" ") {
            normalized = " " + normalized
        }
        // SentencePiece escapes spaces as U+2581 LOWER ONE EIGHTH BLOCK.
        return normalized.replacingOccurrences(of: " ", with: "\u{2581}")
    }

    // MARK: - Encoding

    public func encode(_ text: String) -> [Int] {
        let normalized = normalize(text)
        guard !normalized.isEmpty else { return [] }
        guard !userDefinedPieces.isEmpty else { return encodeBPE(normalized) }

        // Carve out user-defined symbols (longest match wins), BPE the rest.
        var ids: [Int] = []
        var pending = ""
        var cursor = normalized.startIndex
        outer: while cursor < normalized.endIndex {
            for piece in userDefinedPieces {
                if normalized[cursor...].hasPrefix(piece.text) {
                    if !pending.isEmpty {
                        ids.append(contentsOf: encodeBPE(pending))
                        pending = ""
                    }
                    ids.append(piece.id)
                    cursor = normalized.index(cursor, offsetBy: piece.text.count)
                    continue outer
                }
            }
            pending.append(normalized[cursor])
            cursor = normalized.index(after: cursor)
        }
        if !pending.isEmpty {
            ids.append(contentsOf: encodeBPE(pending))
        }
        return ids
    }

    /// SentencePiece BPE: seed with characters, then repeatedly merge the
    /// adjacent pair whose combined piece scores highest, ties broken by
    /// left-most position (matching `std::priority_queue` in the reference).
    private func encodeBPE(_ normalized: String) -> [Int] {
        guard !normalized.isEmpty else { return [] }

        // Doubly linked list over the surviving symbols.
        var symbols: [(text: String, prev: Int, next: Int, alive: Bool)] =
            normalized.map { (String($0), -1, -1, true) }
        guard !symbols.isEmpty else { return [] }
        for index in symbols.indices {
            symbols[index].prev = index - 1
            symbols[index].next = index + 1 < symbols.count ? index + 1 : -1
        }

        struct Candidate {
            let left: Int
            let right: Int
            let score: Float
            let length: Int
        }
        var queue: [Candidate] = []

        func pushCandidate(_ left: Int, _ right: Int) {
            guard left >= 0, right >= 0, symbols[left].alive, symbols[right].alive else { return }
            let merged = symbols[left].text + symbols[right].text
            guard let id = idForPiece[merged] else { return }
            queue.append(Candidate(left: left, right: right, score: pieces[id].score, length: merged.count))
        }

        for index in symbols.indices where symbols[index].next != -1 {
            pushCandidate(index, symbols[index].next)
        }

        while !queue.isEmpty {
            // Pick the best live candidate: highest score, then left-most.
            var bestIndex = -1
            for (index, candidate) in queue.enumerated() {
                guard symbols[candidate.left].alive, symbols[candidate.right].alive,
                      symbols[candidate.left].next == candidate.right else { continue }
                // Stale if either side has since been merged into something else.
                let expected = symbols[candidate.left].text + symbols[candidate.right].text
                guard expected.count == candidate.length else { continue }
                if bestIndex < 0 {
                    bestIndex = index
                } else {
                    let best = queue[bestIndex]
                    if candidate.score > best.score
                        || (candidate.score == best.score && candidate.left < best.left) {
                        bestIndex = index
                    }
                }
            }
            guard bestIndex >= 0 else { break }

            let candidate = queue.remove(at: bestIndex)
            let (left, right) = (candidate.left, candidate.right)
            symbols[left].text += symbols[right].text
            symbols[right].alive = false
            let after = symbols[right].next
            symbols[left].next = after
            if after != -1 { symbols[after].prev = left }

            pushCandidate(symbols[left].prev, left)
            pushCandidate(left, after)
        }

        var ids: [Int] = []
        var cursor: Int = 0
        while cursor >= 0, cursor < symbols.count {
            let symbol = symbols[cursor]
            cursor = symbol.next
            guard symbol.alive else { continue }
            if let id = idForPiece[symbol.text] {
                ids.append(id)
            } else if byteFallback {
                for byte in Array(symbol.text.utf8) {
                    let id = byteFallbackIDs[Int(byte)]
                    ids.append(id >= 0 ? id : unknownID)
                }
            } else {
                ids.append(unknownID)
            }
        }
        return ids
    }

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            let piece = pieces[id]
            switch piece.type {
            case .byte:
                if let value = UInt8(piece.text.dropFirst(3).dropLast(), radix: 16) { bytes.append(value) }
            case .control:
                continue
            default:
                bytes.append(contentsOf: Array(piece.text.utf8))
            }
        }
        return String(decoding: bytes, as: UTF8.self)
            .replacingOccurrences(of: "\u{2581}", with: " ")
    }
}

// MARK: - Minimal protobuf wire reader

struct ProtobufReader {
    struct Field {
        let number: Int
        let wireType: Int
        var varint: UInt64 = 0
        var fixed32: UInt32 = 0
        var data: Data = Data()
    }

    private let data: Data
    private var offset: Int

    init(data: Data) {
        self.data = data
        self.offset = data.startIndex
    }

    private mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while offset < data.endIndex {
            let byte = data[offset]
            offset += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
            if shift > 63 { break }
        }
        throw MossTTSNanoError.tokenizerUnavailable("malformed varint in tokenizer protobuf")
    }

    mutating func nextField() throws -> Field? {
        guard offset < data.endIndex else { return nil }
        let key = try readVarint()
        let number = Int(key >> 3)
        let wireType = Int(key & 0x7)
        var field = Field(number: number, wireType: wireType)

        switch wireType {
        case 0:
            field.varint = try readVarint()
        case 1:
            guard offset + 8 <= data.endIndex else {
                throw MossTTSNanoError.tokenizerUnavailable("truncated 64-bit field")
            }
            offset += 8
        case 2:
            let length = Int(try readVarint())
            guard length >= 0, offset + length <= data.endIndex else {
                throw MossTTSNanoError.tokenizerUnavailable("truncated length-delimited field")
            }
            field.data = data.subdata(in: offset ..< offset + length)
            offset += length
        case 5:
            guard offset + 4 <= data.endIndex else {
                throw MossTTSNanoError.tokenizerUnavailable("truncated 32-bit field")
            }
            var value: UInt32 = 0
            for index in 0 ..< 4 {
                value |= UInt32(data[offset + index]) << (8 * UInt32(index))
            }
            field.fixed32 = value
            offset += 4
        default:
            throw MossTTSNanoError.tokenizerUnavailable("unsupported protobuf wire type \(wireType)")
        }
        return field
    }
}
