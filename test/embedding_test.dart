import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

import 'support.dart';

void main() {
  setUpAll(() => Model2Vec.loadModel(testModelId));

  group('tokenize', () {
    test('breaks text into lowercased tokens', () {
      final tokens = Model2Vec.tokenize('Dart FFI is powerful');
      expect(tokens, isNotEmpty);
      expect(tokens, contains('dart'));
    });

    test('returns an empty list for empty input', () {
      expect(Model2Vec.tokenize(''), isEmpty);
    });
  });

  group('generateEmbedding', () {
    test('returns a model-dimension vector with the expected snapshot', () {
      final vector = Model2Vec.generateEmbedding('Hello world');
      expect(vector.length, Model2Vec.embeddingDimension);
      expect(vector.length, testModelDim);

      // Exact values for Potion Base 2M on "Hello world"; closeTo absorbs
      // minor floating-point differences across platforms.
      expect(vector[0], closeTo(-0.06458017975091934, 1e-6));
      expect(vector[1], closeTo(-0.15089768171310425, 1e-6));
      expect(vector[2], closeTo(-0.27086204290390015, 1e-6));
    });

    test('empty input returns a full-length vector rather than throwing', () {
      // Pinned contract: an empty string yields a dimension-length embedding;
      // it does not raise emptyResult.
      final vector = Model2Vec.generateEmbedding('');
      expect(vector.length, testModelDim);
    });
  });

  group('generateBatchEmbeddings', () {
    test('an empty list returns an empty list', () {
      expect(Model2Vec.generateBatchEmbeddings([]), isEmpty);
    });

    test('is consistent with a single embedding', () {
      const text = 'Consistency check';
      final single = Model2Vec.generateEmbedding(text);
      final batch = Model2Vec.generateBatchEmbeddings([text]);

      expect(batch.length, 1);
      expect(batch[0], orderedEquals(single));
    });

    test('handles multiple diverse strings', () {
      final texts = [
        'Short',
        'A much longer sentence to test truncation logic in the underlying ',
        '12345 !? @#',
        'Теж має працювати',
      ];
      final results = Model2Vec.generateBatchEmbeddings(texts);
      expect(results.length, texts.length);
      for (final v in results) {
        expect(v.length, testModelDim);
      }
    });

    test('stitches native chunks for more than 1024 texts', () {
      // 2000 exceeds the internal FFI chunk size (1024), so this exercises the
      // native multi-chunk stitching.
      final texts = List.generate(2000, (i) => 'row number $i');
      final results = Model2Vec.generateBatchEmbeddings(texts);
      expect(results.length, 2000);
      for (final v in results) {
        expect(v.length, testModelDim);
      }
    });
  });

  group('async embeddings', () {
    test('generateEmbeddingAsync runs in a background isolate', () async {
      final vector = await Model2Vec.generateEmbeddingAsync('Async test');
      expect(vector.length, testModelDim);
    });

    test('generateBatchEmbeddingsAsync runs in a background isolate', () async {
      final results = await Model2Vec.generateBatchEmbeddingsAsync([
        'Async 1',
        'Async 2',
      ]);
      expect(results.length, 2);
      expect(results[0].length, testModelDim);
    });
  });

  group('generateEmbeddingStream', () {
    test('processes a stream of texts across multiple batches', () async {
      const totalItems = 2500;
      final stream = Stream.fromIterable(
        Iterable.generate(totalItems, (i) => 'test string number $i'),
      );

      final results = await Model2Vec.generateEmbeddingStream(
        stream,
        batchSize: 1000,
      ).toList();

      expect(results.length, totalItems);
      expect(results.first.length, testModelDim);
    });

    test('works with useIsolate: false', () async {
      final stream = Stream.fromIterable(['Text 1', 'Text 2', 'Text 3']);
      final results = await Model2Vec.generateEmbeddingStream(
        stream,
        batchSize: 2,
        useIsolate: false,
      ).toList();
      expect(results.length, 3);
    });

    test('cancels the subscription cleanly when taking a prefix', () async {
      final stream = Stream.fromIterable([
        'Text 1',
        'Text 2',
        'Text 3',
        'Text 4',
      ]);
      // take(2) cancels the subscription early.
      final firstTwo = await Model2Vec.generateEmbeddingStream(
        stream,
        batchSize: 2,
      ).take(2).toList();
      expect(firstTwo.length, 2);
    });

    test('an empty stream yields nothing and completes (isolate)', () async {
      final results = await Model2Vec.generateEmbeddingStream(
        const Stream<String>.empty(),
      ).toList();
      expect(results, isEmpty);
    });

    test('an empty stream yields nothing and completes (no isolate)', () async {
      final results = await Model2Vec.generateEmbeddingStream(
        const Stream<String>.empty(),
        useIsolate: false,
      ).toList();
      expect(results, isEmpty);
    });
  });

  group('maxLength', () {
    test('truncation applies to single and batch embeddings', () {
      const text =
          'A very long sentence to test truncation with maxLength parameter';
      // With a very short maxLength the embedding differs from the default.
      final defaultEmbedding = Model2Vec.generateEmbedding(text);
      final shortEmbedding = Model2Vec.generateEmbedding(text, maxLength: 2);
      expect(defaultEmbedding, isNot(orderedEquals(shortEmbedding)));

      final batch = Model2Vec.generateBatchEmbeddings(
        [text, text],
        maxLength: 2,
      );
      expect(batch.length, 2);
      expect(batch[0], orderedEquals(shortEmbedding));
      expect(batch[1], orderedEquals(shortEmbedding));
    });
  });

  group('semantic sanity', () {
    test('related sentences embed closer than unrelated ones', () {
      final cat = Model2Vec.generateEmbedding('A small cute cat');
      final kitten = Model2Vec.generateEmbedding('A young little kitten');
      final space = Model2Vec.generateEmbedding(
        'The exploration of outer space and planets',
      );

      final catKitten = Model2VecUtils.cosineSimilarity(cat, kitten);
      final catSpace = Model2VecUtils.cosineSimilarity(cat, space);

      // Semantically related things should be more similar.
      expect(catKitten, greaterThan(catSpace));
      expect(catKitten, greaterThan(0.5));
      expect(catSpace, lessThan(0.5));
    });
  });

  group('notInitialized', () {
    test('generateEmbedding and tokenize throw after unloadModel', () {
      // Restore the model for the rest of the suite, even if an expect fails.
      addTearDown(() => Model2Vec.loadModel(testModelId));

      Model2Vec.unloadModel();
      expect(Model2Vec.isInitialized, isFalse);

      expect(
        () => Model2Vec.generateEmbedding('x'),
        throwsA(
          isA<Model2VecException>().having(
            (e) => e.kind,
            'kind',
            Model2VecErrorKind.notInitialized,
          ),
        ),
      );
      expect(
        () => Model2Vec.tokenize('x'),
        throwsA(
          isA<Model2VecException>().having(
            (e) => e.kind,
            'kind',
            Model2VecErrorKind.notInitialized,
          ),
        ),
      );
    });
  });
}
