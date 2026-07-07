/// Splits [text] into overlapping chunks suitable for embedding and retrieval.
///
/// Chunks hold at most [maxChars] characters and carry [overlap] characters of
/// trailing context into the next chunk, so a passage that straddles a
/// boundary is still retrievable. Splitting happens on whitespace, so words are
/// never cut mid-way — a single word longer than [maxChars] becomes its own
/// (oversized) chunk rather than being broken.
///
/// To budget by tokens instead of characters, pass
/// `maxChars: maxTokens * Model2Vec.medianTokenLength`, mirroring how the model
/// itself truncates input.
///
/// Returns an empty list for blank input. Throws [ArgumentError] if [maxChars]
/// is below 1 or [overlap] is outside `[0, maxChars)`.
List<String> chunkText(String text, {int maxChars = 1000, int overlap = 100}) {
  if (maxChars < 1) {
    throw ArgumentError.value(maxChars, 'maxChars', 'must be >= 1');
  }
  if (overlap < 0 || overlap >= maxChars) {
    throw ArgumentError.value(overlap, 'overlap', 'must be in [0, maxChars)');
  }

  final words = text.trim().split(RegExp(r'\s+')).where((w) => w.isNotEmpty);
  if (words.isEmpty) {
    return [];
  }

  final chunks = <String>[];
  var current = StringBuffer();

  for (final word in words) {
    if (current.isEmpty) {
      current.write(word);
    } else if (current.length + 1 + word.length <= maxChars) {
      current
        ..write(' ')
        ..write(word);
    } else {
      final chunk = current.toString();
      chunks.add(chunk);
      current = StringBuffer();
      final tail = _overlapTail(chunk, overlap);
      // Only carry the overlap when it still leaves room for this word, so a
      // chunk never exceeds maxChars (except a single oversized word).
      if (tail.isNotEmpty && tail.length + 1 + word.length <= maxChars) {
        current
          ..write(tail)
          ..write(' ');
      }
      current.write(word);
    }
  }

  if (current.isNotEmpty) {
    chunks.add(current.toString());
  }
  return chunks;
}

/// The last [overlap] characters of [chunk], advanced to the next word
/// boundary so the tail starts on a whole word.
String _overlapTail(String chunk, int overlap) {
  if (overlap <= 0 || chunk.length <= overlap) {
    return '';
  }

  var start = chunk.length - overlap;
  final space = chunk.indexOf(' ', start);
  if (space == -1) {
    return ''; // the tail would be a single partial word; skip it
  }

  start = space + 1;

  return chunk.substring(start);
}
