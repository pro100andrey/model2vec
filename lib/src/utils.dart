import 'dart:math' as math;
import 'dart:typed_data';

/// 
typedef _IndexedSimilarity = ({int i, double s});

/// Utilities for working with embedding vectors.
// ignore: avoid_classes_with_only_static_members
final class Model2VecUtils {
  /// Calculates the cosine similarity between two vectors.
  ///
  /// Returns a value between -1.0 and 1.0, where 1.0 means the vectors are
  /// identical.
  ///
  /// If the vectors are already L2-normalized (which is the default for most
  /// Potion models), this is mathematically equivalent to the [dotProduct].
  static double cosineSimilarity(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError(
        'Vectors must have the same length (a: ${a.length}, b: ${b.length})',
      );
    }

    var dotProduct = 0.0;
    var normA = 0.0;
    var normB = 0.0;

    for (var i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0.0 || normB == 0.0) {
      return 0;
    }

    return dotProduct / (math.sqrt(normA) * math.sqrt(normB));
  }

  /// Calculates the dot product of two vectors.
  ///
  /// For L2-normalized vectors, the dot product is equal to the cosine
  /// similarity.
  static double dotProduct(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('Vectors must have the same length');
    }

    var result = 0.0;
    for (var i = 0; i < a.length; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  /// Calculates the Euclidean distance between two vectors.
  static double euclideanDistance(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('Vectors must have the same length');
    }

    var sum = 0.0;
    for (var i = 0; i < a.length; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return math.sqrt(sum);
  }

  /// Finds the indices of the top [topK] most similar vectors in [candidates]

  /// to the [query] vector.
  ///
  /// Returns a list of indices sorted by similarity (descending).
  static List<int> similaritySearch(
    Float32List query,
    List<Float32List> candidates, {
    int topK = 5,
  }) {
    if (candidates.isEmpty) {
      return [];
    }

    final similarities = <_IndexedSimilarity>[];
    for (var i = 0; i < candidates.length; i++) {
      similarities.add(
        (i: i, s: cosineSimilarity(query, candidates[i])),
      );
    }

    similarities.sort((a, b) => b.s.compareTo(a.s));

    return similarities
        .take(math.min(topK, similarities.length))
        .map((s) => s.i)
        .toList(growable: false);
  }
}
