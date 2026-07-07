import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'batcher.dart';
import 'embedding_worker.dart';
import 'exception.dart';
import 'load_progress.dart';
import 'model2vec_bindings.g.dart' as native;
import 'model_info.dart';
import 'recommended_model.dart';

/// The main entry point for the Model2Vec library.
///
/// The native layer keeps a single active model per process, so this type is a
/// stateless namespace of static operations rather than something you
/// instantiate. The native library is located automatically through the Dart
/// SDK's code-asset resolver — there is nothing to boot or inject.
// ignore: avoid_classes_with_only_static_members
abstract final class Model2Vec {
  /// Whether a model is currently loaded. Unlike the metadata getters, this
  /// never throws — use it to guard calls before a model is initialized.
  static bool get isInitialized => native.is_model_loaded() == 1;

  /// Returns the embedding dimension of the currently loaded model.
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
  static int get embeddingDimension => _readInt(native.get_embedding_dimension);

  /// Returns the total number of unique tokens in the model's vocabulary.
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
  static int get vocabularySize => _readInt(native.get_vocabulary_size);

  /// Returns `true` if the model L2-normalizes its output embeddings.
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
  static bool get isNormalized => _readInt(native.is_normalized) == 1;

  /// Returns the median length (in characters) of tokens in the vocabulary.
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
  static int get medianTokenLength => _readInt(native.get_median_token_length);

  /// Reads the current model's metadata into a [ModelInfo].
  ///
  /// Reflects the model loaded at call time; do not switch or unload the model
  /// concurrently. Throws a [Model2VecException] if no model is loaded.
  static ModelInfo get modelInfo => ModelInfo(
    dimension: embeddingDimension,
    vocabularySize: vocabularySize,
    isNormalized: isNormalized,
    medianTokenLength: medianTokenLength,
  );

  static int _readInt(
    int Function(Pointer<Int>, Pointer<Pointer<Char>>) fn,
  ) => using((arena) {
    final outValue = arena<Int>();
    final outError = arena<Pointer<Char>>();
    _check(fn(outValue, outError), outError);
    return outValue.value;
  });

  /// Tokenizes the input [text] into a list of strings.
  ///
  /// Throws a [Model2VecException] if tokenization fails or no model is loaded.
  static List<String> tokenize(String text) => using((arena) {
    final textPtr = text.toNativeUtf8(allocator: arena).cast<Char>();
    final outJson = arena<Pointer<Char>>();
    final outError = arena<Pointer<Char>>();

    _check(native.tokenize(textPtr, outJson, outError), outError);

    final jsonPtr = outJson.value;
    try {
      final jsonString = jsonPtr.cast<Utf8>().toDartString();
      return List<String>.from(json.decode(jsonString) as List);
    } finally {
      native.free_string(jsonPtr);
    }
  });

  /// Loads a model from a Hugging Face repo id or a local directory path,
  /// replacing any currently active model.
  ///
  /// [modelPath] can be a repo id like `minishlab/potion-base-8M` or a path to
  /// a directory containing `model.safetensors`, `config.json` and
  /// `tokenizer.json`.
  ///
  /// This is synchronous and blocks the calling isolate for the whole load,
  /// including the first download. Use [loadModelAsync] to load off-thread.
  static void loadModel(String modelPath) =>
      loadModelAdvanced(modelPath: modelPath);

  /// Advanced model loading with additional options.
  ///
  /// - [modelPath]: Repo id or local path.
  /// - [hfToken]: Optional Hugging Face API token for private repos.
  /// - [cacheDirectory]: Optional path to store downloaded models.
  /// - [normalize]: Whether to L2-normalize output embeddings.
  /// - [subfolder]: Optional subfolder within the repo/path.
  static void loadModelAdvanced({
    required String modelPath,
    String? hfToken,
    String? cacheDirectory,
    bool? normalize,
    String? subfolder,
  }) => using((arena) {
    final pathPtr = modelPath.toNativeUtf8(allocator: arena).cast<Char>();
    final tokenPtr = hfToken == null
        ? nullptr
        : hfToken.toNativeUtf8(allocator: arena).cast<Char>();
    final cachePtr = cacheDirectory == null
        ? nullptr
        : cacheDirectory.toNativeUtf8(allocator: arena).cast<Char>();
    final subPtr = subfolder == null
        ? nullptr
        : subfolder.toNativeUtf8(allocator: arena).cast<Char>();
    final normInt = normalize == null ? -1 : (normalize ? 1 : 0);
    final outError = arena<Pointer<Char>>();

    _check(
      native.init_embedder_advanced(
        pathPtr,
        tokenPtr,
        cachePtr,
        normInt,
        subPtr,
        outError,
      ),
      outError,
    );
  });

  /// Loads a model from raw bytes in memory.
  ///
  /// - [tokenizerBytes]: Content of `tokenizer.json`.
  /// - [modelBytes]: Content of `model.safetensors`.
  /// - [configBytes]: Content of `config.json`.
  static void loadModelFromBytes({
    required Uint8List tokenizerBytes,
    required Uint8List modelBytes,
    required Uint8List configBytes,
  }) => using((arena) {
    final tPtr = arena<Uint8>(tokenizerBytes.length);
    final mPtr = arena<Uint8>(modelBytes.length);
    final cPtr = arena<Uint8>(configBytes.length);

    tPtr.asTypedList(tokenizerBytes.length).setAll(0, tokenizerBytes);
    mPtr.asTypedList(modelBytes.length).setAll(0, modelBytes);
    cPtr.asTypedList(configBytes.length).setAll(0, configBytes);

    final outError = arena<Pointer<Char>>();
    _check(
      native.init_embedder_from_bytes(
        tPtr.cast<UnsignedChar>(),
        tokenizerBytes.length,
        mPtr.cast<UnsignedChar>(),
        modelBytes.length,
        cPtr.cast<UnsignedChar>(),
        configBytes.length,
        outError,
      ),
      outError,
    );
  });

  /// Loads a model asynchronously on a background isolate.
  ///
  /// Prefer this over [loadModel] when the model may be downloaded from
  /// Hugging Face for the first time (tens to hundreds of MB): the synchronous
  /// [loadModel] blocks the calling isolate for the entire download, which
  /// freezes a Flutter UI. The native model is a single process-global, so once
  /// the background isolate has loaded it the model is visible to every isolate
  /// — including the one that awaited this call.
  static Future<void> loadModelAsync(String modelPath) =>
      Isolate.run(() => loadModel(modelPath));

  /// Advanced counterpart to [loadModelAsync].
  ///
  /// Takes the same options as [loadModelAdvanced] and loads them on a
  /// background isolate; see [loadModelAsync] for why loading off-thread
  /// matters and why the loaded model is visible on every isolate.
  static Future<void> loadModelAdvancedAsync({
    required String modelPath,
    String? hfToken,
    String? cacheDirectory,
    bool? normalize,
    String? subfolder,
  }) => Isolate.run(
    () => loadModelAdvanced(
      modelPath: modelPath,
      hfToken: hfToken,
      cacheDirectory: cacheDirectory,
      normalize: normalize,
      subfolder: subfolder,
    ),
  );

  /// Loads a model on a background isolate, reporting progress as a stream of
  /// [LoadProgress] snapshots.
  ///
  /// Like [loadModelAdvancedAsync] the load runs off-thread so the calling
  /// isolate stays responsive; this method additionally polls the native load
  /// state and yields a snapshot roughly every [pollInterval]. The stream's
  /// final event is always [LoadPhase.done]; if the load fails the stream emits
  /// that error instead. Options mirror [loadModelAdvanced].
  ///
  /// Meaningful byte progress only appears while downloading the weights on a
  /// first (uncached) load — an already-cached model or a local path moves
  /// straight to [LoadPhase.done] with no byte counts.
  ///
  /// Only one load may run at a time (the native model is a single process
  /// global); do not start another load, or switch/unload the model, while this
  /// stream is active.
  ///
  /// Cancelling the stream (e.g. `break`ing out of `await for`, or
  /// `subscription.cancel()`) stops progress events but does **not** cancel the
  /// load: the background isolate keeps downloading and will still swap in the
  /// new process-global model when it finishes. There is no way to abort a load
  /// in flight — only [unloadModel] afterwards.
  static Stream<LoadProgress> loadModelWithProgress(
    String modelPath, {
    String? hfToken,
    String? cacheDirectory,
    bool? normalize,
    String? subfolder,
    Duration pollInterval = const Duration(milliseconds: 100),
  }) async* {
    // Arm progress on this isolate before the load starts, so a previous load's
    // terminal state is never observed as this one's opening event.
    native.reset_load_progress();

    final load = loadModelAdvancedAsync(
      modelPath: modelPath,
      hfToken: hfToken,
      cacheDirectory: cacheDirectory,
      normalize: normalize,
      subfolder: subfolder,
    );

    // Mirror completion into a flag and a void future the poll loop can race
    // against. Errors are swallowed here (re-thrown via `await load` below), so
    // this future never completes with an error.
    var finished = false;
    final done = load.then<void>(
      (_) => finished = true,
      onError: (_, _) => finished = true,
    );

    while (!finished) {
      yield _readLoadProgress(terminal: false);
      // Wake on whichever comes first: the next poll tick or load completion.
      await Future.any<void>([Future<void>.delayed(pollInterval), done]);
    }

    // Surface a load failure to the stream consumer.
    await load;

    // Terminal event: the model is loaded and ready.
    yield _readLoadProgress(terminal: true);
  }

  /// Reads a [LoadProgress] snapshot from native load state. While polling
  /// ([terminal] false) a native `done` is clamped to [LoadPhase.parsing] so
  /// the only [LoadPhase.done] a consumer observes is the terminal event
  /// emitted once the load future has actually completed.
  static LoadProgress _readLoadProgress({required bool terminal}) =>
      using((arena) {
        final outPhase = arena<Int>();
        final outDownloaded = arena<Size>();
        final outTotal = arena<Size>();
        native.get_load_progress(outPhase, outDownloaded, outTotal);

        var phase = _loadPhaseFromCode(outPhase.value);
        if (!terminal && phase == LoadPhase.done) {
          phase = LoadPhase.parsing;
        }
        return LoadProgress(
          phase: phase,
          bytesDownloaded: outDownloaded.value,
          totalBytes: outTotal.value,
        );
      });

  /// Maps a native load-phase code (see `native/model2vec.h`) to a [LoadPhase].
  static LoadPhase _loadPhaseFromCode(int code) => switch (code) {
    2 => LoadPhase.downloading,
    3 => LoadPhase.parsing,
    4 => LoadPhase.done,
    _ => LoadPhase.resolving, // 0 idle, 1 resolving
  };

  /// Unloads the active model and frees its native memory.
  ///
  /// Safe to call when no model is loaded. After this, [isInitialized] is
  /// `false` and the metadata getters throw until a model is re-initialized.
  static void unloadModel() => using((arena) {
    final outError = arena<Pointer<Char>>();
    _check(native.free_embedder(outError), outError);
  });

  /// Officially recommended Potion models to start from.
  ///
  /// A curated, offline catalog — not fetched from Hugging Face (see
  /// `docs/adr/0003`), since the editorial fields here cannot be fetched.
  // ignore: omit_obvious_property_types  explicit type documents the public API
  static const List<RecommendedModel> recommendedModels = [
    RecommendedModel(
      id: 'minishlab/potion-base-2M',
      name: 'Potion Base 2M',
      lang: 'English',
      params: '1.8M',
      description: 'Smallest English model, very fast.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-base-4M',
      name: 'Potion Base 4M',
      lang: 'English',
      params: '3.7M',
      description: 'Small and efficient English model.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-base-8M',
      name: 'Potion Base 8M',
      lang: 'English',
      params: '7.5M',
      description: 'Balanced English model.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-base-32M',
      name: 'Potion Base 32M',
      lang: 'English',
      params: '32.3M',
      description: 'Large and accurate English model.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-retrieval-32M',
      name: 'Potion Retrieval 32M',
      lang: 'English',
      params: '32.3M',
      description: 'Optimized specifically for RAG and retrieval tasks.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-code-16M',
      name: 'Potion Code 16M',
      lang: 'Code',
      params: '16M',
      description: 'Optimized for code retrieval and analysis.',
    ),
    RecommendedModel(
      id: 'minishlab/potion-multilingual-128M',
      name: 'Potion Multilingual 128M',
      lang: 'Multilingual (101)',
      params: '128M',
      description: 'Best for multi-language tasks.',
    ),
  ];

  /// Generates a dense vector embedding for the provided [text].
  ///
  /// - [maxLength]: Maximum number of tokens to keep before truncating.
  ///
  /// Returns a [Float32List] representing the text embedding.
  /// Throws a [Model2VecException] if generation fails or no model is loaded.
  static Float32List generateEmbedding(String text, {int maxLength = 512}) =>
      using((arena) {
        final textPtr = text.toNativeUtf8(allocator: arena).cast<Char>();
        final outData = arena<Pointer<Float>>();
        final outDim = arena<Size>();
        final outError = arena<Pointer<Char>>();

        _check(
          native.generate_embedding(
            textPtr,
            maxLength,
            outData,
            outDim,
            outError,
          ),
          outError,
        );

        final dim = outDim.value;
        final dataPtr = outData.value;
        try {
          return Float32List.fromList(dataPtr.asTypedList(dim));
        } finally {
          native.free_floats(dataPtr, dim);
        }
      });

  /// How many texts the native layer processes per internal chunk. This is a
  /// throughput detail with no bearing on the result, so it is not exposed on
  /// [generateBatchEmbeddings].
  static const _ffiBatchSize = 1024;

  /// Generates embeddings for multiple [texts] in a single batch call.
  ///
  /// - [maxLength]: Maximum number of tokens to keep before truncating.
  ///
  /// Returns a list of [Float32List], one per input text.
  /// Throws a [Model2VecException] if generation fails.
  static List<Float32List> generateBatchEmbeddings(
    List<String> texts, {
    int maxLength = 512,
  }) {
    if (texts.isEmpty) {
      return [];
    }

    return using((arena) {
      final count = texts.length;
      final textPointers = arena<Pointer<Char>>(count);
      for (var i = 0; i < count; i++) {
        textPointers[i] = texts[i].toNativeUtf8(allocator: arena).cast<Char>();
      }

      final outData = arena<Pointer<Float>>();
      final outDim = arena<Size>();
      final outCount = arena<Size>();
      final outError = arena<Pointer<Char>>();

      _check(
        native.generate_batch_embeddings_advanced(
          textPointers,
          count,
          maxLength,
          _ffiBatchSize,
          outData,
          outDim,
          outCount,
          outError,
        ),
        outError,
      );

      final dim = outDim.value;
      final resultCount = outCount.value;
      final dataPtr = outData.value;
      try {
        final flatData = dataPtr.asTypedList(resultCount * dim);
        final results = <Float32List>[];
        for (var i = 0; i < resultCount; i++) {
          final start = i * dim;
          // sublist already copies out of native memory into a Dart-owned
          // Float32List, so it stays valid after free_floats below.
          results.add(flatData.sublist(start, start + dim));
        }
        return results;
      } finally {
        native.free_floats(dataPtr, resultCount * dim);
      }
    });
  }

  /// Generates an embedding vector asynchronously in a background Isolate.
  static Future<Float32List> generateEmbeddingAsync(
    String text, {
    int maxLength = 512,
  }) => Isolate.run(
    () => Model2Vec.generateEmbedding(text, maxLength: maxLength),
  );

  /// Generates batch embeddings asynchronously in a background Isolate.
  static Future<List<Float32List>> generateBatchEmbeddingsAsync(
    List<String> texts, {
    int maxLength = 512,
  }) => Isolate.run(
    () => Model2Vec.generateBatchEmbeddings(texts, maxLength: maxLength),
  );

  /// Consumes a stream of [texts] and yields a stream of embeddings.
  ///
  /// The stream is buffered into batches of [batchSize] to maximize throughput.
  ///
  /// By default, [useIsolate] is `true`, which runs the heavy FFI computation
  /// in a single background worker isolate, preventing the main thread from
  /// stuttering. Set `useIsolate: false` in a pure CLI/server context where
  /// blocking the main thread is acceptable, to avoid isolate IPC overhead.
  static Stream<Float32List> generateEmbeddingStream(
    Stream<String> texts, {
    int batchSize = 1024,
    int maxLength = 512,
    bool useIsolate = true,
  }) async* {
    final batches = batched(texts, batchSize);

    if (!useIsolate) {
      await for (final batch in batches) {
        final results = generateBatchEmbeddings(batch, maxLength: maxLength);
        for (final result in results) {
          yield result;
        }
      }
      return;
    }

    EmbeddingWorker? worker;
    try {
      worker = await EmbeddingWorker.start();
      await for (final batch in batches) {
        final results = await worker.embedBatch(batch, maxLength: maxLength);
        for (final result in results) {
          yield result;
        }
      }
    } finally {
      await worker?.close();
    }
  }

  /// Throws a typed [Model2VecException] when [code] is a native failure
  /// (non-zero), reading and freeing the message written to [outError].
  static void _check(int code, Pointer<Pointer<Char>> outError) {
    if (code != 0) {
      throw _nativeException(code, outError.value);
    }
  }

  /// Reads the native error message at [errPtr] (freeing it) and builds a
  /// typed [Model2VecException] for the given native [code].
  static Model2VecException _nativeException(int code, Pointer<Char> errPtr) {
    if (errPtr == nullptr) {
      return Model2VecException.fromNative(code, 'native error (code $code)');
    }
    try {
      return Model2VecException.fromNative(
        code,
        errPtr.cast<Utf8>().toDartString(),
      );
    } finally {
      native.free_string(errPtr);
    }
  }
}
