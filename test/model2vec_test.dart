import 'dart:io';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  late Model2Vec m2v;

  setUpAll(() {
    m2v = Model2Vec.instance;
  });

  group('Model2Vec Production Tests', () {
    test('Successful initialization (online)', () {
      // This MUST be the first init call in the process due to Rust's OnceCell
      expect(
        () => m2v.initEmbedder('minishlab/potion-base-2M'),
        returnsNormally,
      );
    });

    group('Model Metadata', () {
      test('returns valid dimension and metadata', () {
        expect(m2v.embeddingDimension, equals(64));
        expect(m2v.vocabularySize, greaterThan(20000));
        expect(m2v.isNormalized, isTrue);
        expect(m2v.medianTokenLength, isPositive);
      });
    });

    group('Tokenization', () {
      test('breaks text into tokens correctly', () {
        final tokens = m2v.tokenize('Dart FFI is powerful');
        expect(tokens, isNotEmpty);
        expect(tokens, contains('dart'));
      });

      test('handles empty string tokenization', () {
        final tokens = m2v.tokenize('');
        expect(tokens, isEmpty);
      });
    });

    group('Embeddings', () {
      test(
        'generates vector of correct length and exact values (snapshot)',
        () {
          final vector = m2v.generateEmbedding('Hello world');
          expect(vector.length, equals(m2v.embeddingDimension));

          // Exact values for Potion Base 2M on "Hello world"
          // Using closeTo to handle minor floating point precision differences
          expect(vector[0], closeTo(-0.06458017975091934, 1e-6));
          expect(vector[1], closeTo(-0.15089768171310425, 1e-6));
          expect(vector[2], closeTo(-0.27086204290390015, 1e-6));
        },
      );

      test('batch embedding is consistent with single embedding', () {
        const text = 'Consistency check';
        final single = m2v.generateEmbedding(text);
        final batch = m2v.generateBatchEmbeddings([text]);

        expect(batch.length, 1);
        expect(batch[0], orderedEquals(single));
      });

      test('handles batch with multiple diverse strings', () {
        final texts = [
          'Short',
          'A much longer sentence to test truncation logic in the underlying ',
          '12345 !? @#',
          'Теж має працювати',
        ];
        final results = m2v.generateBatchEmbeddings(texts);
        expect(results.length, equals(texts.length));
        for (final v in results) {
          expect(v.length, equals(m2v.embeddingDimension));
        }
      });

      test('batch with empty list returns empty list', () {
        expect(m2v.generateBatchEmbeddings([]), isEmpty);
      });
    });

    group('Asynchronous API', () {
      test('generates embedding asynchronously in isolate', () async {
        final vector = await m2v.generateEmbeddingAsync('Async test');
        expect(vector.length, equals(m2v.embeddingDimension));
      });

      test('generates batch embeddings asynchronously in isolate', () async {
        final results = await m2v.generateBatchEmbeddingsAsync([
          'Async 1',
          'Async 2',
        ]);
        expect(results.length, 2);
        expect(results[0].length, m2v.embeddingDimension);
      });
    });

    group('Model Switching', () {
      test('can switch between different models successfully', () {
        // Switch to 8M model (dimension 256)
        m2v.initEmbedder('minishlab/potion-base-8M');
        expect(m2v.embeddingDimension, equals(256));

        // Switch back to 2M model (dimension 64)
        m2v.initEmbedder('minishlab/potion-base-2M');
        expect(m2v.embeddingDimension, equals(64));
      });
    });

    group('Advanced Features', () {
      test('supports custom cache directory', () {
        final tempDir = Directory.systemTemp.createTempSync('m2v_cache_');
        try {
          // Now that switching is enabled, this should return normally
          expect(
            () => m2v.initEmbedderAdvanced(
              modelPath: 'minishlab/potion-base-2M',
              cacheDirectory: tempDir.path,
            ),
            returnsNormally,
          );
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      });
    });

    group('Recommended Models', () {
      test('returns a non-empty list of maps', () {
        final models = m2v.getRecommendedModels();
        expect(models, isNotEmpty);
        expect(models.first, contains('id'));
      });
    });
  });
}
