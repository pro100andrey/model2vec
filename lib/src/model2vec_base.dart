import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'batcher.dart';
import 'embedding_worker.dart';
import 'exception.dart';
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
  static int get embeddingDimension =>
      _readInt(native.get_embedding_dimension);

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
  static int get medianTokenLength =>
      _readInt(native.get_median_token_length);

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

  /// Initializes a model from a Hugging Face repo id or a local directory path.
  ///
  /// [modelPath] can be a repo id like `minishlab/potion-base-8M` or a path to
  /// a directory containing `model.safetensors`, `config.json` and
  /// `tokenizer.json`.
  static void initEmbedder(String modelPath) =>
      initEmbedderAdvanced(modelPath: modelPath);

  /// Advanced model initialization with additional options.
  ///
  /// - [modelPath]: Repo id or local path.
  /// - [hfToken]: Optional Hugging Face API token for private repos.
  /// - [cacheDirectory]: Optional path to store downloaded models.
  /// - [normalize]: Whether to L2-normalize output embeddings.
  /// - [subfolder]: Optional subfolder within the repo/path.
  static void initEmbedderAdvanced({
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

  /// Initializes a model using raw bytes from memory.
  ///
  /// - [tokenizerBytes]: Content of `tokenizer.json`.
  /// - [modelBytes]: Content of `model.safetensors`.
  /// - [configBytes]: Content of `config.json`.
  static void initEmbedderFromBytes({
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

  /// Generates embeddings for multiple [texts] in a single batch call.
  ///
  /// - [maxLength]: Maximum number of tokens to keep before truncating.
  /// - [batchSize]: Size of the internal batches sent to the model.
  ///
  /// Returns a list of [Float32List], one per input text.
  /// Throws a [Model2VecException] if generation fails.
  static List<Float32List> generateBatchEmbeddings(
    List<String> texts, {
    int maxLength = 512,
    int batchSize = 1024,
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
          batchSize,
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
    int batchSize = 1024,
  }) => Isolate.run(
    () => Model2Vec.generateBatchEmbeddings(
      texts,
      maxLength: maxLength,
      batchSize: batchSize,
    ),
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
        final results = generateBatchEmbeddings(
          batch,
          maxLength: maxLength,
          batchSize: batch.length,
        );
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
