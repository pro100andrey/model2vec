import 'dart:io';
import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('Model2VecUtils - Math', () {
    test('cosineSimilarity calculates correctly', () {
      final a = Float32List.fromList([1.0, 0.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0, 0.0]);
      final c = Float32List.fromList([1.0, 1.0, 0.0]);

      // Orthogonal vectors
      expect(Model2VecUtils.cosineSimilarity(a, b), closeTo(0.0, 1e-6));

      // Identical vectors
      expect(Model2VecUtils.cosineSimilarity(a, a), closeTo(1.0, 1e-6));

      // 45 degree angle
      expect(Model2VecUtils.cosineSimilarity(a, c), closeTo(0.707106, 1e-4));
    });

    test('dotProduct calculates correctly', () {
      final a = Float32List.fromList([1.0, 2.0, 3.0]);
      final b = Float32List.fromList([4.0, 5.0, 6.0]);

      // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
      expect(Model2VecUtils.dotProduct(a, b), equals(32.0));
    });

    test('euclideanDistance calculates correctly', () {
      final a = Float32List.fromList([0.0, 0.0]);
      final b = Float32List.fromList([3.0, 4.0]);

      // sqrt(3^2 + 4^2) = 5
      expect(Model2VecUtils.euclideanDistance(a, b), equals(5.0));
    });

    test('throws ArgumentError on mismatched lengths', () {
      final a = Float32List.fromList([1.0, 2.0]);
      final b = Float32List.fromList([1.0, 2.0, 3.0]);

      expect(() => Model2VecUtils.cosineSimilarity(a, b), throwsArgumentError);
    });

    test('similaritySearch finds most similar vectors', () {
      final query = Float32List.fromList([1.0, 0.0]);
      final candidates = [
        Float32List.fromList([0.0, 1.0]), // orthogonal
        Float32List.fromList([0.9, 0.1]), // very similar
        Float32List.fromList([-1.0, 0.0]), // opposite
        Float32List.fromList([0.5, 0.5]), // somewhat similar
      ];

      final results = Model2VecUtils.similaritySearch(
        query,
        candidates,
        topK: 2,
      );

      expect(results, hasLength(2));
      expect(results[0], equals(1)); // [0.9, 0.1]
      expect(results[1], equals(3)); // [0.5, 0.5]
    });

    test('similaritySearchWithThreshold works correctly', () {
      final query = Float32List.fromList([1.0, 0.0]);
      final candidates = [
        Float32List.fromList([0.0, 1.0]), // sim: 0.0
        Float32List.fromList([0.9, 0.1]), // sim: ~0.99
        Float32List.fromList([-1.0, 0.0]), // sim: -1.0
        Float32List.fromList([0.5, 0.5]), // sim: 0.707
      ];

      final results = Model2VecUtils.similaritySearchWithThreshold(
        query,
        candidates,
        threshold: 0.8,
      );

      expect(results, hasLength(1));
      expect(results[0], equals(1)); // only [0.9, 0.1] passes 0.8
    });

    test('cosineDistance calculates correctly', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([0.0, 1.0]);
      expect(Model2VecUtils.cosineDistance(a, a), closeTo(0.0, 1e-6));
      expect(Model2VecUtils.cosineDistance(a, b), closeTo(1.0, 1e-6));
    });

    test('normalize works correctly', () {
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

    test('meanPooling averages correctly', () {
      final a = Float32List.fromList([1.0, 2.0]);
      final b = Float32List.fromList([3.0, 4.0]);
      final c = Float32List.fromList([5.0, 6.0]);

      final mean = Model2VecUtils.meanPooling([a, b, c]);
      expect(mean[0], closeTo(3.0, 1e-6));
      expect(mean[1], closeTo(4.0, 1e-6));
    });

    test('quantizeToInt8 scales correctly', () {
      final vector = Float32List.fromList([1.0, 0.5, -0.5, -1.0]);
      final quantized = Model2VecUtils.quantizeToInt8(vector);

      expect(quantized[0], equals(127));
      expect(quantized[1], equals(64));
      expect(quantized[2], equals(-64)); // -0.5 * 127 = -63.5 -> -64
      expect(quantized[3], equals(-127));
    });

    test('Base64 serialization works bidirectionally', () {
      final vector = Float32List.fromList([0.1, 0.2, -0.3, 100.5]);
      final base64 = Model2VecUtils.toBase64(vector);
      final restored = Model2VecUtils.fromBase64(base64);

      expect(restored.length, equals(vector.length));
      for (var i = 0; i < vector.length; i++) {
        expect(restored[i], equals(vector[i]));
      }
    });

    test('pairwiseSimilarity calculates matrix correctly', () {
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

  group('Model2VecUtils - Real World', () {
    setUpAll(() {
      Model2Vec.initEmbedder('minishlab/potion-base-2M');
    });

    test('semantic similarity makes sense', () {
      final vCat = Model2Vec.generateEmbedding('A small cute cat');
      final vKitten = Model2Vec.generateEmbedding('A young little kitten');
      final vSpace = Model2Vec.generateEmbedding(
        'The exploration of outer space and planets',
      );

      final simCatKitten = Model2VecUtils.cosineSimilarity(vCat, vKitten);
      final simCatSpace = Model2VecUtils.cosineSimilarity(vCat, vSpace);

      stdout
        ..writeln(
          'Similarity (Cat vs Kitten): '
          '${(simCatKitten * 100).toStringAsFixed(2)}%',
        )
        ..writeln(
          'Similarity (Cat vs Space):  '
          '${(simCatSpace * 100).toStringAsFixed(2)}%',
        );

      // Semantically related things should be more similar
      expect(simCatKitten, greaterThan(simCatSpace));
      expect(simCatKitten, greaterThan(0.5)); // High similarity
      expect(simCatSpace, lessThan(0.5)); // Low similarity
    });

    test('task similarity example', () {
      final vTask1 = Model2Vec.generateEmbedding(
        'Fix the login bug in production',
      );
      final vTask2 = Model2Vec.generateEmbedding(
        'Resolve authentication issue on server',
      );
      final vTask3 = Model2Vec.generateEmbedding(
        'Order pizza for the team lunch',
      );

      final sim12 = Model2VecUtils.cosineSimilarity(vTask1, vTask2);
      final sim13 = Model2VecUtils.cosineSimilarity(vTask1, vTask3);

      stdout
        ..writeln(
          'Similarity (Login vs Auth): ${(sim12 * 100).toStringAsFixed(2)}%',
        )
        ..writeln(
          'Similarity (Login vs Pizza): ${(sim13 * 100).toStringAsFixed(2)}%',
        );

      expect(sim12, greaterThan(sim13));
    });
  });
}
