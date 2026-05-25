import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;

import 'exception.dart';
import 'model2vec_bindings.g.dart';

/// The main entry point for the Model2Vec library.
class Model2Vec {
  /// Creates a new instance of [Model2Vec] using the provided [library].
  Model2Vec(DynamicLibrary library) : _bindings = Model2VecBindings(library);
  static Model2Vec? _instance;
  final Model2VecBindings _bindings;
  int? _cachedDimension;

  /// Manually initializes the shared [instance] with a specific [library].
  static void boot(DynamicLibrary library) {
    _instance = Model2Vec(library);
  }

  /// A shared singleton instance of [Model2Vec].
  static Model2Vec get instance {
    if (_instance != null) {
      return _instance!;
    }
    try {
      final library = DynamicLibrary.open(_resolveLibPath());
      _instance = Model2Vec(library);
      return _instance!;
    } catch (e) {
      throw Model2VecException(
        'Failed to auto-load Model2Vec native library: $e',
      );
    }
  }

  /// Returns the embedding dimension of the currently loaded model.
  int get embeddingDimension {
    if (_cachedDimension != null) {
      return _cachedDimension!;
    }
    final dim = _bindings.get_embedding_dimension();
    if (dim <= 0) {
      throw const Model2VecException(
        'No model initialized. Call initEmbedder() before accessing dimension.',
      );
    }
    _cachedDimension = dim;
    return dim;
  }

  /// Returns the total number of unique tokens in the model's vocabulary.
  int get vocabularySize {
    final size = _bindings.get_vocabulary_size();
    if (size < 0) {
      throw const Model2VecException(
        'Failed to get vocabulary size. Is a model initialized?',
      );
    }
    return size;
  }

  /// Returns `true` if the model is configured to L2-normalize output
  /// embeddings.
  bool get isNormalized => _bindings.is_normalized() == 1;

  /// Returns the median length (in characters) of tokens in the vocabulary.
  int get medianTokenLength {
    final length = _bindings.get_median_token_length();
    if (length < 0) {
      throw const Model2VecException('Failed to get median token length.');
    }
    return length;
  }

  /// Tokenizes the input [text].
  List<String> tokenize(String text) => using((arena) {
    final textPtr = text.toNativeUtf8(allocator: arena);
    final resPtr = _bindings.tokenize(textPtr.cast<Char>());
    if (resPtr == nullptr) {
      throw const Model2VecException('Tokenization failed.');
    }
    try {
      final jsonString = resPtr.cast<Utf8>().toDartString();
      return List<String>.from(json.decode(jsonString) as List);
    } finally {
      _bindings.free_string(resPtr);
    }
  });

  /// Initializes a model from a Hugging Face repo ID or a local directory path.
  void initEmbedder(String modelPath) {
    initEmbedderAdvanced(modelPath: modelPath);
  }

  /// Advanced model initialization.
  void initEmbedderAdvanced({
    required String modelPath,
    String? hfToken,
    String? cacheDirectory,
    bool? normalize,
    String? subfolder,
  }) {
    using((arena) {
      final pathPtr = modelPath.toNativeUtf8(allocator: arena);
      final tokenPtr = hfToken?.toNativeUtf8(allocator: arena) ?? nullptr;
      final cachePtr =
          cacheDirectory?.toNativeUtf8(allocator: arena) ?? nullptr;
      final subPtr = subfolder?.toNativeUtf8(allocator: arena) ?? nullptr;
      final normInt = normalize == null ? -1 : (normalize ? 1 : 0);

      final res = _bindings.init_embedder_advanced(
        pathPtr.cast<Char>(),
        tokenPtr.cast<Char>(),
        cachePtr.cast<Char>(),
        normInt,
        subPtr.cast<Char>(),
      );

      if (res != 0) {
        throw Model2VecException(
          'Failed to initialize model at "$modelPath".',
          res,
        );
      }
      _cachedDimension = null;
    });
  }

  /// Initializes a model using raw bytes from memory.
  void initEmbedderFromBytes({
    required Uint8List tokenizerBytes,
    required Uint8List modelBytes,
    required Uint8List configBytes,
  }) {
    using((arena) {
      final tPtr = arena<Uint8>(tokenizerBytes.length);
      final mPtr = arena<Uint8>(modelBytes.length);
      final cPtr = arena<Uint8>(configBytes.length);

      tPtr.asTypedList(tokenizerBytes.length).setAll(0, tokenizerBytes);
      mPtr.asTypedList(modelBytes.length).setAll(0, modelBytes);
      cPtr.asTypedList(configBytes.length).setAll(0, configBytes);

      final res = _bindings.init_embedder_from_bytes(
        tPtr.cast<UnsignedChar>(),
        tokenizerBytes.length,
        mPtr.cast<UnsignedChar>(),
        modelBytes.length,
        cPtr.cast<UnsignedChar>(),
        configBytes.length,
      );

      if (res != 0) {
        throw Model2VecException('Failed to initialize model from bytes.', res);
      }
      _cachedDimension = null;
    });
  }

  /// Returns a list of officially recommended Potion models.
  List<Map<String, dynamic>> getRecommendedModels() {
    final ptr = _bindings.get_model_list();
    if (ptr == nullptr) {
      return [];
    }
    try {
      final jsonString = ptr.cast<Utf8>().toDartString();
      return List<Map<String, dynamic>>.from(json.decode(jsonString) as List);
    } finally {
      _bindings.free_string(ptr);
    }
  }

  /// Generates a dense vector embedding for the provided [text].
  Float32List generateEmbedding(String text) {
    final dim = embeddingDimension;
    return using((arena) {
      final textPtr = text.toNativeUtf8(allocator: arena);
      final outVector = arena<Float>(dim);
      final res = _bindings.generate_embedding(
        textPtr.cast<Char>(),
        outVector,
        dim,
      );
      if (res != 0) {
        throw Model2VecException('Failed to generate embedding for text.', res);
      }
      return Float32List.fromList(outVector.asTypedList(dim));
    });
  }

  /// Generates embeddings for multiple [texts] in a highly optimized batch
  /// call.
  List<Float32List> generateBatchEmbeddings(List<String> texts) {
    if (texts.isEmpty) {
      return [];
    }
    final dim = embeddingDimension;
    final count = texts.length;

    return using((arena) {
      final outVectors = arena<Float>(count * dim);
      final textPointers = arena<Pointer<Char>>(count);

      for (var i = 0; i < count; i++) {
        textPointers[i] = texts[i].toNativeUtf8(allocator: arena).cast<Char>();
      }

      final res = _bindings.generate_batch_embeddings(
        textPointers,
        count,
        outVectors,
      );
      if (res != 0) {
        throw Model2VecException('Failed to generate batch embeddings.', res);
      }

      final results = <Float32List>[];
      final flatData = outVectors.asTypedList(count * dim);
      for (var i = 0; i < count; i++) {
        final start = i * dim;
        results.add(Float32List.fromList(flatData.sublist(start, start + dim)));
      }
      return results;
    });
  }

  /// Generates an embedding vector asynchronously in a background Isolate.
  Future<Float32List> generateEmbeddingAsync(String text) =>
      Isolate.run(() => Model2Vec.instance.generateEmbedding(text));

  /// Generates batch embeddings asynchronously in a background Isolate.
  Future<List<Float32List>> generateBatchEmbeddingsAsync(List<String> texts) =>
      Isolate.run(() => Model2Vec.instance.generateBatchEmbeddings(texts));

  static String _resolveLibPath() {
    final libName = Platform.isLinux
        ? 'libm2v_ffi.so'
        : Platform.isMacOS
        ? 'libm2v_ffi.dylib'
        : 'm2v_ffi.dll';

    final path = Directory.current.path;

    final searchPaths = [
      p.join(path, libName),
      p.join(path, 'lib', libName),
      p.join(path, '.dart_tool', 'lib', libName),
      p.join(path, 'packages', 'model2vec', '.dart_tool', 'lib', libName),
      p.join(path, 'native', 'target', 'release', libName),
      p.join(
        path,
        'packages',
        'model2vec',
        'native',
        'target',
        'release',
        libName,
      ),
    ];

    for (final path in searchPaths) {
      if (File(path).existsSync()) {
        return path;
      }
    }

    return libName;
  }
}
