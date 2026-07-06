/// A snapshot of the active model's metadata, read in one call.
final class ModelInfo {
  /// Creates a metadata snapshot.
  const ModelInfo({
    required this.dimension,
    required this.vocabularySize,
    required this.isNormalized,
    required this.medianTokenLength,
  });

  /// Length of the embedding vectors the model produces.
  final int dimension;

  /// Number of unique tokens in the model's vocabulary.
  final int vocabularySize;

  /// Whether the model L2-normalizes its output embeddings.
  final bool isNormalized;

  /// Median token length, in characters, across the vocabulary.
  final int medianTokenLength;

  @override
  bool operator ==(Object other) =>
      other is ModelInfo &&
      other.dimension == dimension &&
      other.vocabularySize == vocabularySize &&
      other.isNormalized == isNormalized &&
      other.medianTokenLength == medianTokenLength;

  @override
  int get hashCode =>
      Object.hash(dimension, vocabularySize, isNormalized, medianTokenLength);

  @override
  String toString() =>
      'ModelInfo(dimension: $dimension, vocabularySize: $vocabularySize, '
      'isNormalized: $isNormalized, medianTokenLength: $medianTokenLength)';
}
