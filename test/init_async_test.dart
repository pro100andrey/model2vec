import 'dart:io';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('async model loading', () {
    test(
      'loadModelAsync loads the process-global model, visible on the '
      'calling isolate',
      () async {
        await Model2Vec.loadModelAsync('minishlab/potion-base-2M');

        // The load ran on a background isolate; the native model is a
        // process-global, so a synchronous call on this isolate sees it.
        expect(Model2Vec.isInitialized, isTrue);
        expect(Model2Vec.embeddingDimension, equals(64));
        expect(
          Model2Vec.generateEmbedding('async load').length,
          equals(64),
        );
      },
    );

    test('loadModelAdvancedAsync honors a custom cache directory', () async {
      final tempDir = Directory.systemTemp.createTempSync('m2v_async_cache_');
      try {
        await Model2Vec.loadModelAdvancedAsync(
          modelPath: 'minishlab/potion-base-2M',
          cacheDirectory: tempDir.path,
        );
        expect(Model2Vec.isInitialized, isTrue);
        expect(Model2Vec.embeddingDimension, equals(64));
      } finally {
        tempDir.deleteSync(recursive: true);
      }
    });

    test('a failed async load surfaces a typed Model2VecException', () {
      expect(
        Model2Vec.loadModelAsync('definitely/not-a-real-model-xyz'),
        throwsA(isA<Model2VecException>()),
      );
    });
  });
}
