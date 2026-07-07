import 'dart:io';
import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

import 'support.dart';

void main() {
  // Load the default fixture once before the suite, and restore it after every
  // test. The native model is a single process-global, so tests that unload or
  // switch it must not leak that state to their neighbours; reloading the
  // (cached) model after each test keeps every test starting from a clean 2M.
  setUpAll(() => Model2Vec.loadModel(testModelId));
  tearDown(() => Model2Vec.loadModel(testModelId));

  group('loadModel', () {
    test('loads a model by repo id and reports it initialized', () {
      Model2Vec.loadModel(testModelId);
      expect(Model2Vec.isInitialized, isTrue);
    });
  });

  group('loadModelAdvanced', () {
    test('populates a fresh custom cache directory with the model files', () {
      final cacheDir = Directory.systemTemp.createTempSync('m2v_cache_');
      addTearDown(() => cacheDir.deleteSync(recursive: true));

      Model2Vec.loadModelAdvanced(
        modelPath: testModelId,
        cacheDirectory: cacheDir.path,
      );

      expect(Model2Vec.isInitialized, isTrue);
      // The download actually landed in the directory we handed it.
      expect(cacheDir.listSync(), isNotEmpty);
    });

    test('with normalize:false disables output normalization', () {
      Model2Vec.loadModelAdvanced(modelPath: testModelId, normalize: false);
      expect(Model2Vec.isInitialized, isTrue);
      expect(Model2Vec.isNormalized, isFalse);
      // tearDown restores the normal (normalized) model for later tests.
    });
  });

  group('loadModelFromBytes', () {
    test('loads a model from the cached tokenizer/model/config bytes', () {
      final bytes = _cachedModelBytes(testModelId);
      if (bytes == null) {
        markTestSkipped('cached files for $testModelId not found in HF cache');
        return;
      }

      Model2Vec.loadModelFromBytes(
        tokenizerBytes: bytes.tokenizer,
        modelBytes: bytes.model,
        configBytes: bytes.config,
      );

      expect(Model2Vec.isInitialized, isTrue);
      expect(Model2Vec.embeddingDimension, testModelDim);
    });

    test('with garbage bytes throws a typed initFromBytesFailed', () {
      final garbage = Uint8List.fromList([0, 1, 2, 3]);
      expect(
        () => Model2Vec.loadModelFromBytes(
          tokenizerBytes: garbage,
          modelBytes: garbage,
          configBytes: garbage,
        ),
        throwsA(
          isA<Model2VecException>()
              .having(
                (e) => e.kind,
                'kind',
                Model2VecErrorKind.initFromBytesFailed,
              )
              .having((e) => e.message, 'message', isNotEmpty),
        ),
      );
    });
  });

  group('async loading', () {
    test('loadModelAsync loads the process-global model', () async {
      await Model2Vec.loadModelAsync(testModelId);

      // The load ran on a background isolate; the native model is a
      // process-global, so a synchronous call on this isolate sees it.
      expect(Model2Vec.isInitialized, isTrue);
      expect(Model2Vec.embeddingDimension, testModelDim);
      expect(Model2Vec.generateEmbedding('async load').length, testModelDim);
    });

    test('loadModelAdvancedAsync honors a custom cache directory', () async {
      final cacheDir = Directory.systemTemp.createTempSync('m2v_async_cache_');
      addTearDown(() => cacheDir.deleteSync(recursive: true));

      await Model2Vec.loadModelAdvancedAsync(
        modelPath: testModelId,
        cacheDirectory: cacheDir.path,
      );

      expect(Model2Vec.isInitialized, isTrue);
      expect(Model2Vec.embeddingDimension, testModelDim);
    });

    test('a failed async load surfaces a typed Model2VecException', () async {
      // Awaited so the rejected future is asserted here, not left to surface
      // later as an unhandled async error.
      await expectLater(
        Model2Vec.loadModelAsync('bad/nope'),
        throwsA(isA<Model2VecException>()),
      );
    });
  });

  group('loadModelWithProgress', () {
    test('downloads and reaches done on a fresh cache', () async {
      // A private temp cache guarantees a cache miss, so the weights actually
      // download and the downloading phase is exercised.
      final tmp = Directory.systemTemp.createTempSync('m2v_progress_');
      addTearDown(() => tmp.deleteSync(recursive: true));

      final events = <LoadProgress>[];
      await for (final p in Model2Vec.loadModelWithProgress(
        testModelId,
        cacheDirectory: tmp.path,
        pollInterval: const Duration(milliseconds: 1),
      )) {
        events.add(p);
      }

      // The load succeeded and the stream ended on `done`.
      expect(Model2Vec.isInitialized, isTrue);
      expect(events, isNotEmpty);
      expect(events.last.phase, LoadPhase.done);

      // `done` is only ever the terminal event.
      final nonTerminal = events.sublist(0, events.length - 1);
      expect(
        nonTerminal,
        everyElement(
          isNot(predicate<LoadProgress>((e) => e.phase == LoadPhase.done)),
        ),
      );

      // A fresh cache must download the weights: we should have caught at least
      // one downloading snapshot with a known total and a real fraction.
      final downloading = events
          .where((e) => e.phase == LoadPhase.downloading)
          .toList();
      expect(
        downloading,
        isNotEmpty,
        reason: 'a fresh cache should download the weights',
      );
      final last = downloading.last;
      expect(last.totalBytes, greaterThan(0));
      expect(last.bytesDownloaded, greaterThan(0));
      expect(last.fraction, isNotNull);
      expect(last.fraction, inInclusiveRange(0.0, 1.0));
    });

    test('a cached model streams straight to done', () async {
      // The prior test cached the model in a temp dir, but the default cache
      // may already hold it too; either way this must terminate on `done`.
      final events = <LoadProgress>[];
      await for (final p in Model2Vec.loadModelWithProgress(testModelId)) {
        events.add(p);
      }

      expect(Model2Vec.isInitialized, isTrue);
      expect(events.last.phase, LoadPhase.done);
    });

    test('a failed load surfaces a typed exception on the stream', () async {
      await expectLater(
        Model2Vec.loadModelWithProgress('definitely/not-a-real-model').toList(),
        throwsA(isA<Model2VecException>()),
      );
    });
  });

  group('isInitialized and unloadModel', () {
    test('unloadModel clears the model; metadata getters then throw', () {
      expect(Model2Vec.isInitialized, isTrue);

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
      // tearDown reloads the model for later tests in this process.
    });
  });

  group('modelInfo', () {
    test('reports dimension, vocab, normalization and median token length', () {
      final info = Model2Vec.modelInfo;
      expect(info.dimension, testModelDim);
      expect(info.vocabularySize, greaterThan(testModelVocabMin));
      expect(info.isNormalized, isTrue);
      expect(info.medianTokenLength, isPositive);
    });

    test('the four metadata getters agree with modelInfo', () {
      expect(Model2Vec.embeddingDimension, testModelDim);
      expect(Model2Vec.vocabularySize, greaterThan(testModelVocabMin));
      expect(Model2Vec.isNormalized, isTrue);
      expect(Model2Vec.medianTokenLength, isPositive);
    });
  });

  group('model switching', () {
    test('switches between the 2M and 8M models', () {
      Model2Vec.loadModel(largeModelId);
      expect(Model2Vec.embeddingDimension, largeModelDim);

      Model2Vec.loadModel(testModelId);
      expect(Model2Vec.embeddingDimension, testModelDim);
    });
  });
}

/// Reads the cached `tokenizer.json`, `model.safetensors` and `config.json`
/// bytes for [modelId] from the local Hugging Face cache, or `null` when the
/// model is not cached. Used by the `loadModelFromBytes` happy-path test.
({Uint8List tokenizer, Uint8List model, Uint8List config})? _cachedModelBytes(
  String modelId,
) {
  final env = Platform.environment;
  final hfHome = env['HF_HOME'] ?? '${env['HOME']}/.cache/huggingface';
  final repoDir = 'models--${modelId.replaceAll('/', '--')}';
  final snapshots = Directory('$hfHome/hub/$repoDir/snapshots');
  if (!snapshots.existsSync()) {
    return null;
  }

  for (final entry in snapshots.listSync().whereType<Directory>()) {
    final tokenizer = File('${entry.path}/tokenizer.json');
    final model = File('${entry.path}/model.safetensors');
    final config = File('${entry.path}/config.json');
    if (tokenizer.existsSync() && model.existsSync() && config.existsSync()) {
      return (
        tokenizer: tokenizer.readAsBytesSync(),
        model: model.readAsBytesSync(),
        config: config.readAsBytesSync(),
      );
    }
  }
  return null;
}
