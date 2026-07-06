import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('model lifecycle', () {
    test('isInitialized / modelInfo / unloadModel round-trip', () {
      Model2Vec.initEmbedder('minishlab/potion-base-2M');
      expect(Model2Vec.isInitialized, isTrue);

      final info = Model2Vec.modelInfo;
      expect(info.dimension, equals(64));
      expect(info.vocabularySize, greaterThan(20000));
      expect(info.isNormalized, isTrue);
      expect(info.medianTokenLength, isPositive);

      Model2Vec.unloadModel();
      expect(Model2Vec.isInitialized, isFalse);
      expect(
        () => Model2Vec.embeddingDimension,
        throwsA(
          isA<Model2VecException>().having(
            (e) => e.kind,
            'kind',
            Model2VecErrorKind.notInitialized,
          ),
        ),
      );

      // Idempotent: unloading again is a no-op.
      expect(Model2Vec.unloadModel, returnsNormally);

      // Re-initialize so later suites in this process find a model.
      Model2Vec.initEmbedder('minishlab/potion-base-2M');
      expect(Model2Vec.isInitialized, isTrue);
    });
  });
}
