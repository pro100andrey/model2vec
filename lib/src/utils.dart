import 'dart:convert';
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
  /// If either vector is a zero vector, this method safely returns `0.0`
  /// instead of `NaN`.
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

  /// Finds indices of all vectors in [candidates] that have a similarity
  /// score greater than or equal to [threshold] with the [query].
  ///
  /// Returns a list of indices sorted by similarity (descending).
  static List<int> similaritySearchWithThreshold(
    Float32List query,
    List<Float32List> candidates, {
    required double threshold,
  }) {
    if (candidates.isEmpty) {
      return [];
    }

    final similarities = <_IndexedSimilarity>[];
    for (var i = 0; i < candidates.length; i++) {
      final sim = cosineSimilarity(query, candidates[i]);
      if (sim >= threshold) {
        similarities.add((i: i, s: sim));
      }
    }

    similarities.sort((a, b) => b.s.compareTo(a.s));
    return similarities.map((s) => s.i).toList(growable: false);
  }

  /// Calculates the cosine distance between two vectors.
  /// Distance is `1.0 - cosineSimilarity`. Range is 0.0 to 2.0.
  static double cosineDistance(Float32List a, Float32List b) =>
      1.0 - cosineSimilarity(a, b);

  /// Applies L2 normalization to a vector in-place
  /// (or returns a normalized copy).
  ///
  /// If the input vector is a zero vector (norm equals 0), this method
  /// safely returns a copy of the zero vector without throwing or
  /// introducing `NaN` values.
  static Float32List normalize(Float32List vector) {
    var normSq = 0.0;
    for (var i = 0; i < vector.length; i++) {
      normSq += vector[i] * vector[i];
    }
    if (normSq == 0.0) {
      return Float32List.fromList(vector);
    }
    final norm = math.sqrt(normSq);
    final result = Float32List(vector.length);
    for (var i = 0; i < vector.length; i++) {
      result[i] = vector[i] / norm;
    }
    return result;
  }

  /// Computes the mean (average) of multiple vectors.
  /// Useful for pooling token embeddings into a sentence embedding.
  static Float32List meanPooling(List<Float32List> vectors) {
    if (vectors.isEmpty) {
      throw ArgumentError('Cannot pool an empty list of vectors');
    }
    final dim = vectors.first.length;
    final result = Float32List(dim);

    for (final v in vectors) {
      if (v.length != dim) {
        throw ArgumentError('All vectors must have the same dimension');
      }
      for (var i = 0; i < dim; i++) {
        result[i] += v[i];
      }
    }

    final count = vectors.length.toDouble();
    for (var i = 0; i < dim; i++) {
      result[i] /= count;
    }

    return result;
  }

  /// Quantizes a normalized [Float32List] to an [Int8List].
  /// This saves 4x memory with minimal loss of accuracy for search.
  static Int8List quantizeToInt8(Float32List vector) {
    final result = Int8List(vector.length);

    for (var i = 0; i < vector.length; i++) {
      result[i] = (vector[i] * 127.0).round().clamp(-128, 127);
    }

    return result;
  }

  /// Serializes a [Float32List] to a Base64 string for easy storage.
  static String toBase64(Float32List vector) =>
      base64Encode(vector.buffer.asUint8List());

  /// Deserializes a [Float32List] from a Base64 string.
  static Float32List fromBase64(String base64String) {
    final bytes = base64Decode(base64String);
    final buffer = bytes.buffer;

    return Float32List.view(
      buffer,
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ Float32List.bytesPerElement,
    );
  }

  /// Computes pairwise similarities between all vectors in [listA] and [listB].
  /// Returns a 2D list where `result[i][j]` is the similarity between
  /// `listA[i]` and `listB[j]`.
  static List<List<double>> pairwiseSimilarity(
    List<Float32List> listA,
    List<Float32List> listB,
  ) {
    final result = <List<double>>[];
    for (final a in listA) {
      final row = <double>[];
      for (final b in listB) {
        row.add(cosineSimilarity(a, b));
      }

      result.add(row);
    }

    return result;
  }
}
