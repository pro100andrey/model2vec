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
  });

  group('Model2VecUtils - Real World', () {
    late Model2Vec m2v;

    setUpAll(() {
      m2v = Model2Vec.instance..initEmbedder('minishlab/potion-base-2M');
    });

    test('semantic similarity makes sense', () {
      final vCat = m2v.generateEmbedding('A small cute cat');
      final vKitten = m2v.generateEmbedding('A young little kitten');
      final vSpace = m2v.generateEmbedding(
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
      final vTask1 = m2v.generateEmbedding('Fix the login bug in production');
      final vTask2 = m2v.generateEmbedding(
        'Resolve authentication issue on server',
      );
      final vTask3 = m2v.generateEmbedding('Order pizza for the team lunch');

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
