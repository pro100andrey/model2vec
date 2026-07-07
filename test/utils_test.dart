import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('Model2VecUtils', () {
    group('cosineSimilarity', () {
      test('scores orthogonal, identical, and 45-degree vectors', () {
        final a = Float32List.fromList([1.0, 0.0, 0.0]);
        final b = Float32List.fromList([0.0, 1.0, 0.0]);
        final c = Float32List.fromList([1.0, 1.0, 0.0]);

        expect(Model2VecUtils.cosineSimilarity(a, b), closeTo(0.0, 1e-6));
        expect(Model2VecUtils.cosineSimilarity(a, a), closeTo(1.0, 1e-6));
        expect(Model2VecUtils.cosineSimilarity(a, c), closeTo(0.707106, 1e-4));
      });

      test('returns 0.0 for a zero vector instead of NaN', () {
        final zero = Float32List.fromList([0.0, 0.0, 0.0]);
        final a = Float32List.fromList([1.0, 0.0, 0.0]);

        final score = Model2VecUtils.cosineSimilarity(zero, a);
        expect(score, equals(0.0));
        expect(score.isNaN, isFalse);
      });

      test('throws on mismatched lengths', () {
        final a = Float32List.fromList([1.0, 2.0]);
        final b = Float32List.fromList([1.0, 2.0, 3.0]);

        expect(
          () => Model2VecUtils.cosineSimilarity(a, b),
          throwsArgumentError,
        );
      });
    });

    group('dotProduct', () {
      test('multiplies and sums element-wise', () {
        final a = Float32List.fromList([1.0, 2.0, 3.0]);
        final b = Float32List.fromList([4.0, 5.0, 6.0]);

        // 1*4 + 2*5 + 3*6 = 32
        expect(Model2VecUtils.dotProduct(a, b), equals(32.0));
      });

      test('throws on mismatched lengths', () {
        final a = Float32List.fromList([1.0, 2.0]);
        final b = Float32List.fromList([1.0, 2.0, 3.0]);

        expect(() => Model2VecUtils.dotProduct(a, b), throwsArgumentError);
      });
    });

    group('euclideanDistance', () {
      test('computes the straight-line distance', () {
        final a = Float32List.fromList([0.0, 0.0]);
        final b = Float32List.fromList([3.0, 4.0]);

        // sqrt(3^2 + 4^2) = 5
        expect(Model2VecUtils.euclideanDistance(a, b), equals(5.0));
      });

      test('throws on mismatched lengths', () {
        final a = Float32List.fromList([1.0, 2.0]);
        final b = Float32List.fromList([1.0, 2.0, 3.0]);

        expect(
          () => Model2VecUtils.euclideanDistance(a, b),
          throwsArgumentError,
        );
      });
    });

    group('cosineDistance', () {
      test('is 1 minus the cosine similarity', () {
        final a = Float32List.fromList([1.0, 0.0]);
        final b = Float32List.fromList([0.0, 1.0]);
        expect(Model2VecUtils.cosineDistance(a, a), closeTo(0.0, 1e-6));
        expect(Model2VecUtils.cosineDistance(a, b), closeTo(1.0, 1e-6));
      });
    });

    group('normalize', () {
      test('scales a vector to unit length', () {
        final a = Float32List.fromList([3.0, 4.0]);
        final normalized = Model2VecUtils.normalize(a);

        expect(normalized[0], closeTo(0.6, 1e-6));
        expect(normalized[1], closeTo(0.8, 1e-6));
        expect(
          Model2VecUtils.euclideanDistance(
            normalized,
            Float32List.fromList([0, 0]),
          ),
          closeTo(1.0, 1e-6),
        );
      });

      test('returns all-zeros for a zero vector with no NaN', () {
        final zero = Float32List.fromList([0.0, 0.0, 0.0]);
        final normalized = Model2VecUtils.normalize(zero);

        expect(normalized, [0.0, 0.0, 0.0]);
        expect(normalized.any((e) => e.isNaN), isFalse);
      });
    });

    group('meanPooling', () {
      test('averages vectors element-wise', () {
        final a = Float32List.fromList([1.0, 2.0]);
        final b = Float32List.fromList([3.0, 4.0]);
        final c = Float32List.fromList([5.0, 6.0]);

        final mean = Model2VecUtils.meanPooling([a, b, c]);
        expect(mean[0], closeTo(3.0, 1e-6));
        expect(mean[1], closeTo(4.0, 1e-6));
      });

      test('throws on an empty list', () {
        expect(
          () => Model2VecUtils.meanPooling(const []),
          throwsArgumentError,
        );
      });

      test('throws on mismatched dimensions', () {
        final a = Float32List.fromList([1.0, 2.0]);
        final b = Float32List.fromList([3.0, 4.0, 5.0]);
        expect(
          () => Model2VecUtils.meanPooling([a, b]),
          throwsArgumentError,
        );
      });
    });

    group('quantizeToInt8', () {
      test('scales values into the int8 range', () {
        final vector = Float32List.fromList([1.0, 0.5, -0.5, -1.0]);
        final quantized = Model2VecUtils.quantizeToInt8(vector);

        expect(quantized[0], equals(127));
        expect(quantized[1], equals(64));
        expect(quantized[2], equals(-64)); // -0.5 * 127 = -63.5 -> -64
        expect(quantized[3], equals(-127));
      });

      test('clamps out-of-range inputs to [-128, 127]', () {
        final vector = Float32List.fromList([1.5, -1.5, 2.0]);
        final quantized = Model2VecUtils.quantizeToInt8(vector);

        // 1.5*127=190.5 -> 191 -> clamp 127; -1.5*127=-190.5 -> -191 ->
        // clamp -128; 2.0*127=254 -> clamp 127.
        expect(quantized, [127, -128, 127]);
      });
    });

    group('dequantizeInt8', () {
      test('approximately inverts quantizeToInt8', () {
        final vector = Float32List.fromList([0.5, -0.25, 1.0, -1.0, 0.0]);
        final restored = Model2VecUtils.dequantizeInt8(
          Model2VecUtils.quantizeToInt8(vector),
        );

        expect(restored.length, equals(vector.length));
        for (var i = 0; i < vector.length; i++) {
          expect(restored[i], closeTo(vector[i], 0.01));
        }
      });
    });

    group('similaritySearchWithScores', () {
      test('returns sorted index+score pairs', () {
        final query = Float32List.fromList([1, 0, 0]);
        final candidates = [
          Float32List.fromList([1, 0.2, 0]),
          Float32List.fromList([1, 0.25, 0]),
          Float32List.fromList([0.6, 0, 0.8]),
        ];

        final results = Model2VecUtils.similaritySearchWithScores(
          query,
          candidates,
          topK: 2,
        );
        expect(results.map((r) => r.index).toList(), [0, 1]);
        expect(results.first.score, greaterThan(results[1].score));
      });

      test('returns empty for no candidates', () {
        final query = Float32List.fromList([1, 0, 0]);
        expect(
          Model2VecUtils.similaritySearchWithScores(query, const []),
          isEmpty,
        );
      });

      test('drops results below the threshold', () {
        final query = Float32List.fromList([1.0, 0.0]);
        final candidates = [
          Float32List.fromList([0.0, 1.0]), // sim: 0.0
          Float32List.fromList([0.9, 0.1]), // sim: ~0.994
          Float32List.fromList([-1.0, 0.0]), // sim: -1.0
          Float32List.fromList([0.5, 0.5]), // sim: ~0.707
        ];

        final results = Model2VecUtils.similaritySearchWithScores(
          query,
          candidates,
          topK: 10,
          threshold: 0.8,
        );

        expect(results.map((r) => r.index).toList(), [1]);
        expect(results.single.score, greaterThan(0.8));
      });

      test('keeps a candidate scoring exactly the threshold', () {
        final query = Float32List.fromList([1.0, 0.0]);
        final candidates = [
          Float32List.fromList([1.0, 0.0]), // identical -> cosine 1.0
          Float32List.fromList([0.0, 1.0]), // orthogonal -> cosine 0.0
        ];

        final results = Model2VecUtils.similaritySearchWithScores(
          query,
          candidates,
          topK: 10,
          threshold: 1, // boundary is inclusive (>=)
        );

        expect(results.map((r) => r.index).toList(), [0]);
        expect(results.single.score, closeTo(1.0, 1e-6));
      });

      test('caps threshold matches at topK', () {
        final query = Float32List.fromList([1.0, 0.0]);
        final candidates = [
          Float32List.fromList([1.0, 0.0]), // sim 1.0
          Float32List.fromList([0.99, 0.01]), // sim ~0.9999
          Float32List.fromList([0.98, 0.02]), // sim ~0.9998
        ];

        final results = Model2VecUtils.similaritySearchWithScores(
          query,
          candidates,
          topK: 2,
          threshold: 0.5, // all three clear the floor, but topK caps to 2
        );

        expect(results, hasLength(2));
        expect(results.map((r) => r.index).toList(), [0, 1]);
      });
    });

    group('maximalMarginalRelevance', () {
      test('prefers a diverse result over a near-duplicate', () {
        final query = Float32List.fromList([1, 0, 0]);
        final candidates = [
          Float32List.fromList([1, 0.2, 0]), // 0: most relevant
          Float32List.fromList([1, 0.25, 0]), // 1: near-duplicate of 0
          Float32List.fromList([0.6, 0, 0.8]), // 2: relevant-ish but diverse
        ];

        final selected = Model2VecUtils.maximalMarginalRelevance(
          query,
          candidates,
          topK: 2,
        );
        expect(selected.first, 0); // most relevant first
        expect(selected[1], 2); // diversity beats the near-duplicate
      });

      test('with lambda 1.0 is pure relevance order', () {
        final query = Float32List.fromList([1, 0, 0]);
        final candidates = [
          Float32List.fromList([0.6, 0, 0.8]),
          Float32List.fromList([1, 0.2, 0]),
          Float32List.fromList([1, 0.25, 0]),
        ];

        final selected = Model2VecUtils.maximalMarginalRelevance(
          query,
          candidates,
          topK: 3,
          lambda: 1,
        );
        expect(selected, [1, 2, 0]);
      });

      test('with lambda 0.0 selects the maximally-dissimilar spread', () {
        final query = Float32List.fromList([1, 0, 0]);
        final candidates = [
          Float32List.fromList([1, 0, 0]), // 0: identical to the query
          Float32List.fromList([0.9, 0.1, 0]), // 1: near-duplicate of 0
          Float32List.fromList([-1, 0, 0]), // 2: opposite of 0
        ];

        final selected = Model2VecUtils.maximalMarginalRelevance(
          query,
          candidates,
          topK: 2,
          lambda: 0,
        );
        // Pure diversity: after the first pick, the maximally-dissimilar
        // candidate (2) wins over the near-duplicate (1).
        expect(selected, [0, 2]);
      });

      test('returns empty for no candidates', () {
        final query = Float32List.fromList([1, 0, 0]);
        expect(
          Model2VecUtils.maximalMarginalRelevance(query, const []),
          isEmpty,
        );
      });

      test('rejects lambda outside [0, 1]', () {
        final query = Float32List.fromList([1, 0]);
        final candidates = [
          Float32List.fromList([1, 0]),
        ];
        expect(
          () => Model2VecUtils.maximalMarginalRelevance(
            query,
            candidates,
            lambda: 1.5,
          ),
          throwsArgumentError,
        );
        expect(
          () => Model2VecUtils.maximalMarginalRelevance(
            query,
            candidates,
            lambda: -0.1,
          ),
          throwsArgumentError,
        );
      });
    });

    group('base64', () {
      test('round-trips a vector bidirectionally', () {
        final vector = Float32List.fromList([0.1, 0.2, -0.3, 100.5]);
        final base64 = Model2VecUtils.toBase64(vector);
        final restored = Model2VecUtils.fromBase64(base64);

        expect(restored.length, equals(vector.length));
        for (var i = 0; i < vector.length; i++) {
          expect(restored[i], equals(vector[i]));
        }
      });
    });

    group('pairwiseSimilarity', () {
      test('computes the full similarity matrix', () {
        final listA = [
          Float32List.fromList([1.0, 0.0]),
          Float32List.fromList([0.0, 1.0]),
        ];
        final listB = [
          Float32List.fromList([1.0, 0.0]),
          Float32List.fromList([0.0, 1.0]),
          Float32List.fromList([0.707, 0.707]),
        ];

        final result = Model2VecUtils.pairwiseSimilarity(listA, listB);
        expect(result.length, 2);
        expect(result[0].length, 3);

        // listA[0] vs listB
        expect(result[0][0], closeTo(1.0, 1e-6));
        expect(result[0][1], closeTo(0.0, 1e-6));
        expect(result[0][2], closeTo(0.707106, 1e-4));

        // listA[1] vs listB
        expect(result[1][0], closeTo(0.0, 1e-6));
        expect(result[1][1], closeTo(1.0, 1e-6));
        expect(result[1][2], closeTo(0.707106, 1e-4));
      });
    });
  });
}
