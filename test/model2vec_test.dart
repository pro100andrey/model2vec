import 'dart:io';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('Model2Vec Production Tests', () {
    test('Successful initialization (online)', () {
      // Must be the first init call so later tests have a loaded model.
      expect(
        () => Model2Vec.initEmbedder('minishlab/potion-base-2M'),
        returnsNormally,
      );
    });

    group('Model Metadata', () {
      test('returns valid dimension and metadata', () {
        expect(Model2Vec.embeddingDimension, equals(64));
        expect(Model2Vec.vocabularySize, greaterThan(20000));
        expect(Model2Vec.isNormalized, isTrue);
        expect(Model2Vec.medianTokenLength, isPositive);
      });
    });

    group('Tokenization', () {
      test('breaks text into tokens correctly', () {
        final tokens = Model2Vec.tokenize('Dart FFI is powerful');
        expect(tokens, isNotEmpty);
        expect(tokens, contains('dart'));
      });

      test('handles empty string tokenization', () {
        final tokens = Model2Vec.tokenize('');
        expect(tokens, isEmpty);
      });
    });

    group('Embeddings', () {
      test(
        'generates vector of correct length and exact values (snapshot)',
        () {
          final vector = Model2Vec.generateEmbedding('Hello world');
          expect(vector.length, equals(Model2Vec.embeddingDimension));

          // Exact values for Potion Base 2M on "Hello world"
          // Using closeTo to handle minor floating point precision differences
          expect(vector[0], closeTo(-0.06458017975091934, 1e-6));
          expect(vector[1], closeTo(-0.15089768171310425, 1e-6));
          expect(vector[2], closeTo(-0.27086204290390015, 1e-6));
        },
      );

      test('batch embedding is consistent with single embedding', () {
        const text = 'Consistency check';
        final single = Model2Vec.generateEmbedding(text);
        final batch = Model2Vec.generateBatchEmbeddings([text]);

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
        final results = Model2Vec.generateBatchEmbeddings(texts);
        expect(results.length, equals(texts.length));
        for (final v in results) {
          expect(v.length, equals(Model2Vec.embeddingDimension));
        }
      });

      test('batch with empty list returns empty list', () {
        expect(Model2Vec.generateBatchEmbeddings([]), isEmpty);
      });
    });

    group('Asynchronous API', () {
      test('generates embedding asynchronously in isolate', () async {
        final vector = await Model2Vec.generateEmbeddingAsync('Async test');
        expect(vector.length, equals(Model2Vec.embeddingDimension));
      });

      test('generates batch embeddings asynchronously in isolate', () async {
        final results = await Model2Vec.generateBatchEmbeddingsAsync([
          'Async 1',
          'Async 2',
        ]);
        expect(results.length, 2);
        expect(results[0].length, Model2Vec.embeddingDimension);
      });
    });

    group('Streaming API', () {
      test('processes a stream of texts and yields embeddings', () async {
        // Create a synthetic stream of 2500 strings
        const totalItems = 2500;
        final stream = Stream.fromIterable(
          Iterable.generate(totalItems, (i) => 'This is test string number $i'),
        );

        final resultStream = Model2Vec.generateEmbeddingStream(
          stream,
          batchSize: 1000,
        );

        final results = await resultStream.toList();

        expect(results.length, equals(totalItems));
        expect(results.first.length, equals(Model2Vec.embeddingDimension));
      });

      test('works correctly with useIsolate: false', () async {
        final stream = Stream.fromIterable(['Text 1', 'Text 2', 'Text 3']);
        final results = await Model2Vec.generateEmbeddingStream(
          stream,
          batchSize: 2,
          useIsolate: false,
        ).toList();

        expect(results.length, equals(3));
      });

      test('canceling stream early works cleanly', () async {
        final stream = Stream.fromIterable([
          'Text 1',
          'Text 2',
          'Text 3',
          'Text 4',
        ]);
        final resultStream = Model2Vec.generateEmbeddingStream(
          stream,
          batchSize: 2,
        );

        // Take only 2 elements, which will cancel the subscription early
        final firstTwo = await resultStream.take(2).toList();
        expect(firstTwo.length, 2);
      });
    });

    group('Model Switching', () {
      test('can switch between different models successfully', () {
        // Switch to 8M model (dimension 256)
        Model2Vec.initEmbedder('minishlab/potion-base-8M');
        expect(Model2Vec.embeddingDimension, equals(256));

        // Switch back to 2M model (dimension 64)
        Model2Vec.initEmbedder('minishlab/potion-base-2M');
        expect(Model2Vec.embeddingDimension, equals(64));
      });
    });

    group('Advanced Features', () {
      test('supports advanced parameters maxLength and batchSize', () {
        const text =
            'A very long sentence to test truncation with maxLength parameter';
        // With very short maxLength, the embedding should be different from
        // default
        final defaultEmbedding = Model2Vec.generateEmbedding(text);
        final shortEmbedding = Model2Vec.generateEmbedding(text, maxLength: 2);

        expect(defaultEmbedding, isNot(orderedEquals(shortEmbedding)));

        final batchEmbedding = Model2Vec.generateBatchEmbeddings(
          [text, text],
          maxLength: 2,
          batchSize: 1,
        );
        expect(batchEmbedding.length, 2);
        expect(batchEmbedding[0], orderedEquals(shortEmbedding));
        expect(batchEmbedding[1], orderedEquals(shortEmbedding));
      });

      test('supports custom cache directory', () {
        final tempDir = Directory.systemTemp.createTempSync('m2v_cache_');
        try {
          // Now that switching is enabled, this should return normally
          expect(
            () => Model2Vec.initEmbedderAdvanced(
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
      test('exposes a non-empty typed catalog', () {
        expect(Model2Vec.recommendedModels, isNotEmpty);
        expect(
          Model2Vec.recommendedModels.first.id,
          startsWith('minishlab/'),
        );
      });
    });
  });
}
