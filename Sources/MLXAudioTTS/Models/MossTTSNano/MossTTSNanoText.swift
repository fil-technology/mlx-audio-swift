import Foundation

/// Prompt assembly and sentence chunking for MOSS-TTS-Nano.
///
/// Ported from `mlx_audio/tts/models/moss_tts_nano/text.py` (Apache-2.0).
/// The prompt template itself originates in OpenMOSS/MOSS-TTS-Nano and must be
/// reproduced verbatim — the model was trained against these exact strings.
enum MossTTSNanoText {
    static let userRolePrefix = "user\n"
    static let userTemplateReferencePrefix = "<user_inst>\n- Reference(s):\n"
    static let userTemplateAfterReference = """

    - Instruction:
    None
    - Tokens:
    None
    - Quality:
    None
    - Sound Event:
    None
    - Ambient Sound:
    None
    - Language:
    None
    - Text:

    """
    static let userTemplateSuffix = "\n</user_inst>"
    static let assistantTurnPrefix = "\n"
    static let assistantRolePrefix = "assistant\n"

    static let sentenceEndPunctuation: Set<Character> = Set(".!?。！？；;")
    static let clauseSplitPunctuation: Set<Character> = Set(",，、；;：:")
    static let closingPunctuation: Set<Character> = Set("\"'”’)]}）】》」』")

    // MARK: - Prompt sections

    static func userPromptPrefix(
        tokenizer: SentencePieceBPETokenizer,
        config: MossTTSNanoConfiguration
    ) -> [Int] {
        [config.imStartTokenId]
            + tokenizer.encode(userRolePrefix)
            + tokenizer.encode(userTemplateReferencePrefix)
    }

    static func userPromptAfterReference(tokenizer: SentencePieceBPETokenizer) -> [Int] {
        tokenizer.encode(userTemplateAfterReference)
    }

    static func assistantPromptPrefix(
        tokenizer: SentencePieceBPETokenizer,
        config: MossTTSNanoConfiguration
    ) -> [Int] {
        tokenizer.encode(userTemplateSuffix)
            + [config.imEndTokenId]
            + tokenizer.encode(assistantTurnPrefix)
            + [config.imStartTokenId]
            + tokenizer.encode(assistantRolePrefix)
    }

    // MARK: - Normalization

    static func lightweightNormalize(_ text: String) -> String {
        let collapsed = text
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return collapsed.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression
        )
    }

    static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
                || (0x3400...0x4DBF).contains(scalar.value)
                || (0x3040...0x30FF).contains(scalar.value)
                || (0xAC00...0xD7AF).contains(scalar.value)
        }
    }

    /// Mirrors `prepare_text_for_sentence_chunking`. The short-text indent is
    /// deliberate: the reference pads very short prompts with leading spaces,
    /// which measurably steadies prosody at the start of an utterance.
    static func prepareForChunking(_ text: String) throws -> String {
        var normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            throw MossTTSNanoError.invalidConfiguration("Text prompt cannot be empty.")
        }
        normalized = normalized
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        while normalized.contains("  ") {
            normalized = normalized.replacingOccurrences(of: "  ", with: " ")
        }

        if containsCJK(normalized) {
            if let last = normalized.last, !sentenceEndPunctuation.contains(last) {
                normalized += "。"
            }
            return normalized
        }

        if let first = normalized.first, first.isLowercase {
            normalized = first.uppercased() + normalized.dropFirst()
        }
        if let last = normalized.last, last.isLetter || last.isNumber {
            normalized += "."
        }
        if normalized.split(separator: " ").filter({ !$0.isEmpty }).count < 5 {
            normalized = "        " + normalized
        }
        return normalized
    }

    // MARK: - Chunking

    static func splitByPunctuation(_ text: String, punctuation: Set<Character>) -> [String] {
        var sentences: [String] = []
        var current: [Character] = []
        let characters = Array(text)
        var index = 0

        while index < characters.count {
            let character = characters[index]
            current.append(character)
            if punctuation.contains(character) {
                var lookahead = index + 1
                while lookahead < characters.count, closingPunctuation.contains(characters[lookahead]) {
                    current.append(characters[lookahead])
                    lookahead += 1
                }
                let sentence = String(current).trimmingCharacters(in: .whitespaces)
                if !sentence.isEmpty { sentences.append(sentence) }
                current.removeAll()
                while lookahead < characters.count, characters[lookahead].isWhitespace {
                    lookahead += 1
                }
                index = lookahead
                continue
            }
            index += 1
        }

        let tail = String(current).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { sentences.append(tail) }
        return sentences
    }

    static func joinParts(_ left: String, _ right: String) -> String {
        if left.isEmpty { return right }
        if right.isEmpty { return left }
        if containsCJK(left) || containsCJK(right) { return left + right }
        return "\(left) \(right)"
    }

    /// Binary-searches the longest prefix that fits the token budget, then
    /// backs up to a nearby clause boundary when one is within 25 characters.
    static func splitByTokenBudget(
        tokenizer: SentencePieceBPETokenizer,
        text: String,
        maxTokens: Int
    ) -> [String] {
        var remaining = text.trimmingCharacters(in: .whitespaces)
        guard !remaining.isEmpty else { return [] }

        var pieces: [String] = []
        let boundaryCharacters = clauseSplitPunctuation.union(sentenceEndPunctuation).union([" "])

        while !remaining.isEmpty {
            if tokenizer.encode(remaining).count <= maxTokens {
                pieces.append(remaining)
                break
            }

            let characters = Array(remaining)
            var low = 1
            var high = characters.count
            var bestPrefixLength = 1
            while low <= high {
                let middle = (low + high) / 2
                let candidate = String(characters[0 ..< middle]).trimmingCharacters(in: .whitespaces)
                if candidate.isEmpty {
                    low = middle + 1
                    continue
                }
                if tokenizer.encode(candidate).count <= maxTokens {
                    bestPrefixLength = middle
                    low = middle + 1
                } else {
                    high = middle - 1
                }
            }

            var cutIndex = bestPrefixLength
            let scanMin = max(-1, bestPrefixLength - 25)
            var scanIndex = bestPrefixLength - 1
            while scanIndex > scanMin {
                if boundaryCharacters.contains(characters[scanIndex]) {
                    cutIndex = scanIndex + 1
                    break
                }
                scanIndex -= 1
            }

            var piece = String(characters[0 ..< cutIndex]).trimmingCharacters(in: .whitespaces)
            if piece.isEmpty {
                piece = String(characters[0 ..< bestPrefixLength]).trimmingCharacters(in: .whitespaces)
                cutIndex = bestPrefixLength
            }
            pieces.append(piece)
            remaining = String(characters[cutIndex...]).trimmingCharacters(in: .whitespaces)
        }
        return pieces
    }

    /// Splits into sentence-sized chunks, then repacks them up to `maxTokens`.
    static func splitIntoBestSentences(
        tokenizer: SentencePieceBPETokenizer,
        text: String,
        maxTokens: Int = 75
    ) throws -> [String] {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return [] }
        let safeMaxTokens = max(1, maxTokens)
        let prepared = try prepareForChunking(normalized)

        var candidates = splitByPunctuation(prepared, punctuation: sentenceEndPunctuation)
        if candidates.isEmpty { candidates = [prepared.trimmingCharacters(in: .whitespaces)] }

        var slices: [(count: Int, text: String)] = []
        for candidate in candidates {
            let sentence = candidate.trimmingCharacters(in: .whitespaces)
            guard !sentence.isEmpty else { continue }
            let sentenceTokens = tokenizer.encode(sentence).count
            if sentenceTokens <= safeMaxTokens {
                slices.append((sentenceTokens, sentence))
                continue
            }

            var clauses = splitByPunctuation(sentence, punctuation: clauseSplitPunctuation)
            if clauses.count <= 1 { clauses = [sentence] }
            for clause in clauses {
                let normalizedClause = clause.trimmingCharacters(in: .whitespaces)
                guard !normalizedClause.isEmpty else { continue }
                let clauseTokens = tokenizer.encode(normalizedClause).count
                if clauseTokens <= safeMaxTokens {
                    slices.append((clauseTokens, normalizedClause))
                    continue
                }
                for piece in splitByTokenBudget(
                    tokenizer: tokenizer, text: normalizedClause, maxTokens: safeMaxTokens
                ) {
                    let normalizedPiece = piece.trimmingCharacters(in: .whitespaces)
                    guard !normalizedPiece.isEmpty else { continue }
                    slices.append((tokenizer.encode(normalizedPiece).count, normalizedPiece))
                }
            }
        }

        var chunks: [String] = []
        var currentChunk = ""
        var currentCount = 0
        for slice in slices {
            if currentChunk.isEmpty {
                currentChunk = slice.text
                currentCount = slice.count
                continue
            }
            if currentCount + slice.count > safeMaxTokens {
                chunks.append(currentChunk.trimmingCharacters(in: .whitespaces))
                currentChunk = slice.text
                currentCount = slice.count
            } else {
                currentChunk = joinParts(currentChunk, slice.text)
                currentCount = tokenizer.encode(currentChunk).count
            }
        }
        if !currentChunk.isEmpty {
            chunks.append(currentChunk.trimmingCharacters(in: .whitespaces))
        }
        // The reference falls back to the raw text when chunking produced a
        // single piece, so a lone sentence keeps its original spacing.
        return chunks.count > 1 ? chunks : [normalized]
    }
}
