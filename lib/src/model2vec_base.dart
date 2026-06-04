import 'dart:async';
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
///
/// This class provides methods to initialize the embedder, generate text
/// embeddings, and access model metadata like vocabulary size and dimensions.
class Model2Vec {
  /// Creates a new instance of [Model2Vec] using the provided [library].
  ///
  /// In most cases, you should use the shared [instance] instead of
  /// creating a new one manually.
  Model2Vec(DynamicLibrary library) : _bindings = Model2VecBindings(library);

  static Model2Vec? _instance;

  final Model2VecBindings _bindings;
  int? _cachedDimension;

  /// Manually initializes the shared [instance] with a specific [library].
  ///
  /// This is useful if you need to load the native library from a custom
  /// location or if automatic resolution fails.
  static void boot(DynamicLibrary library) {
    _instance = Model2Vec(library);
  }

  /// A shared singleton instance of [Model2Vec].
  ///
  /// When accessed for the first time, it attempts to automatically resolve
  /// and load the native library using [Platform.isLinux], [Platform.isMacOS],
  /// and [Platform.isWindows] to determine the correct library filename.
  ///
  /// Throws a [Model2VecException] if the library cannot be found.
  static Model2Vec get instance {
    if (_instance != null) {
      return _instance!;
    }

    final libPath = _resolveLibPath();
    try {
      final library = DynamicLibrary.open(libPath);
      _instance = Model2Vec(library);

      return _instance!;
    } catch (e) {
      throw Model2VecException(
        'Failed to auto-load Model2Vec native library "$libPath".\n'
        'Please ensure you have built the Rust library '
        '(cd native && cargo build --release)\n'
        'or that the shared object is placed in a system library path.\n'
        'Underlying error: $e',
      );
    }
  }

  /// Returns the embedding dimension of the currently loaded model.
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
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
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
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
  ///
  /// Throws a [Model2VecException] if no model has been initialized yet.
  int get medianTokenLength {
    final length = _bindings.get_median_token_length();

    if (length < 0) {
      throw const Model2VecException('Failed to get median token length.');
    }

    return length;
  }

  /// Tokenizes the input [text] into a list of strings.
  ///
  /// Throws a [Model2VecException] if tokenization fails or if no model is
  /// initialized.
  List<String> tokenize(String text) => using((arena) {
    final textPtr = text.toNativeUtf8(allocator: arena);
    final resPtr = _bindings.tokenize(textPtr.cast<Char>());

    if (resPtr == nullptr) {
      throw const Model2VecException('Tokenization failed.');
    }

    try {
      final jsonString = resPtr.cast<Utf8>().toDartString();
      return .from(json.decode(jsonString) as List);
    } finally {
      _bindings.free_string(resPtr);
    }
  });

  /// Initializes a model from a Hugging Face repo ID or a local directory path.
  ///
  /// [modelPath] can be either a repo ID like 'minishlab/potion-base-8M'
  /// or a path to a directory containing `model.safetensors`, `config.json`,
  /// and `tokenizer.json`.
  void initEmbedder(String modelPath) =>
      initEmbedderAdvanced(modelPath: modelPath);

  /// Advanced model initialization with additional options.
  ///
  /// - [modelPath]: Repo ID or local path.
  /// - [hfToken]: Optional Hugging Face API token for private repos.
  /// - [cacheDirectory]: Optional path to store downloaded models.
  /// - [normalize]: Whether to L2-normalize output embeddings.
  /// - [subfolder]: Optional subfolder within the repo/path.
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
        throw Model2VecException.fromCode(res, 'Initialization failed');
      }

      _cachedDimension = null;
    });
  }

  /// Initializes a model using raw bytes from memory.
  ///
  /// Requires the content of the three main configuration files:
  /// - [tokenizerBytes]: Content of `tokenizer.json`.
  /// - [modelBytes]: Content of `model.safetensors`.
  /// - [configBytes]: Content of `config.json`.
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
        throw Model2VecException.fromCode(
          res,
          'Initialization from bytes failed',
        );
      }

      _cachedDimension = null;
    });
  }

  /// Returns a list of officially recommended Potion models from Hugging Face.
  List<Map<String, dynamic>> getRecommendedModels() => [
    {
      'id': 'minishlab/potion-base-2M',
      'name': 'Potion Base 2M',
      'lang': 'English',
      'params': '1.8M',
      'description': 'Smallest English model, very fast.',
    },
    {
      'id': 'minishlab/potion-base-4M',
      'name': 'Potion Base 4M',
      'lang': 'English',
      'params': '3.7M',
      'description': 'Small and efficient English model.',
    },
    {
      'id': 'minishlab/potion-base-8M',
      'name': 'Potion Base 8M',
      'lang': 'English',
      'params': '7.5M',
      'description': 'Balanced English model.',
    },
    {
      'id': 'minishlab/potion-base-32M',
      'name': 'Potion Base 32M',
      'lang': 'English',
      'params': '32.3M',
      'description': 'Large and accurate English model.',
    },
    {
      'id': 'minishlab/potion-retrieval-32M',
      'name': 'Potion Retrieval 32M',
      'lang': 'English',
      'params': '32.3M',
      'description': 'Optimized specifically for RAG and retrieval tasks.',
    },
    {
      'id': 'minishlab/potion-code-16M',
      'name': 'Potion Code 16M',
      'lang': 'Code',
      'params': '16M',
      'description': 'Optimized for code retrieval and analysis.',
    },
    {
      'id': 'minishlab/potion-multilingual-128M',
      'name': 'Potion Multilingual 128M',
      'lang': 'Multilingual (101)',
      'params': '128M',
      'description': 'Best for multi-language tasks.',
    },
  ];

  /// Generates a dense vector embedding for the provided [text].
  ///
  /// - [maxLength]: Maximum number of tokens to keep before truncating.
  ///
  /// Returns a [Float32List] representing the text embedding.
  /// Throws a [Model2VecException] if generation fails or no model is
  /// initialized.
  Float32List generateEmbedding(String text, {int maxLength = 512}) {
    final dim = embeddingDimension;
    return using((arena) {
      final textPtr = text.toNativeUtf8(allocator: arena).cast<Char>();
      final outVector = arena<Float>(dim);

      final res = _bindings.generate_embedding(
        textPtr,
        outVector,
        maxLength,
      );

      if (res != 0) {
        throw Model2VecException.fromCode(res, 'Embedding generation failed');
      }

      return .fromList(outVector.asTypedList(dim));
    });
  }

  /// Generates embeddings for multiple [texts] in a highly optimized batch
  /// call.
  ///
  /// - [maxLength]: Maximum number of tokens to keep before truncating.
  /// - [batchSize]: Size of the internal batches sent to the model for
  /// inference.
  ///
  /// Returns a list of [Float32List], one for each input text.
  /// Throws a [Model2VecException] if generation fails.
  List<Float32List> generateBatchEmbeddings(
    List<String> texts, {
    int maxLength = 512,
    int batchSize = 1024,
  }) {
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

      final res = _bindings.generate_batch_embeddings_advanced(
        textPointers,
        count,
        outVectors,
        maxLength,
        batchSize,
      );

      if (res != 0) {
        throw Model2VecException.fromCode(
          res,
          'Batch embedding generation failed',
        );
      }

      final results = <Float32List>[];
      final flatData = outVectors.asTypedList(count * dim);
      for (var i = 0; i < count; i++) {
        final start = i * dim;
        results.add(.fromList(flatData.sublist(start, start + dim)));
      }

      return results;
    });
  }

  /// Generates an embedding vector asynchronously in a background Isolate.
  Future<Float32List> generateEmbeddingAsync(
    String text, {
    int maxLength = 512,
  }) => Isolate.run(
    () => Model2Vec.instance.generateEmbedding(
      text,
      maxLength: maxLength,
    ),
  );

  /// Generates batch embeddings asynchronously in a background Isolate.
  Future<List<Float32List>> generateBatchEmbeddingsAsync(
    List<String> texts, {
    int maxLength = 512,
    int batchSize = 1024,
  }) => Isolate.run(
    () => Model2Vec.instance.generateBatchEmbeddings(
      texts,
      maxLength: maxLength,
      batchSize: batchSize,
    ),
  );

  /// Consumes a stream of [texts] and yields a stream of embeddings.
  ///
  /// The stream is buffered into batches of [batchSize] to maximize throughput.
  ///
  /// By default, [useIsolate] is `true`, which runs the heavy FFI computations
  /// in a single background worker isolate, preventing the main thread from
  /// stuttering. This is ideal for Flutter applications or processing millions
  /// of rows.
  ///
  /// **Performance Warning for CLI / Server:**
  /// If you are writing a pure Dart CLI or server application where blocking
  /// the main thread is acceptable (or if you already manage your own isolate
  /// pool),  setting `useIsolate: false` will eliminate Inter-Process
  /// Communication (IPC) overhead, avoiding unnecessary serialization of
  /// strings and Float32Lists between isolates, making generation up to 2x
  /// faster for small batches.
  Stream<Float32List> generateEmbeddingStream(
    Stream<String> texts, {
    int batchSize = 1024,
    int maxLength = 512,
    bool useIsolate = true,
  }) async* {
    if (!useIsolate) {
      var buffer = <String>[];
      Stream<Float32List> flushBuffer(List<String> batch) async* {
        final results = generateBatchEmbeddings(
          batch,
          maxLength: maxLength,
          batchSize: batch.length,
        );

        for (final res in results) {
          yield res;
        }
      }

      await for (final text in texts) {
        buffer.add(text);

        if (buffer.length >= batchSize) {
          final currentBatch = buffer;
          buffer = <String>[];
          yield* flushBuffer(currentBatch);
        }
      }

      if (buffer.isNotEmpty) {
        yield* flushBuffer(buffer);
      }

      return;
    }

    final mainReceivePort = ReceivePort();
    Isolate? isolate;
    SendPort? workerSendPort;
    StreamSubscription<dynamic>? sub;

    try {
      isolate = await .spawn(_streamWorker, mainReceivePort.sendPort);
      var pendingRequest = Completer<dynamic>();

      sub = mainReceivePort.listen((message) {
        pendingRequest.complete(message);
      });

      final firstMessage = await pendingRequest.future;
      if (firstMessage is SendPort) {
        workerSendPort = firstMessage;
      } else {
        throw StateError('Worker isolate failed to initialize');
      }

      var buffer = <String>[];

      Stream<Float32List> flushBuffer(List<String> batch) async* {
        pendingRequest = Completer<dynamic>();
        workerSendPort!.send((
          batch: batch,
          maxLength: maxLength,
        ));

        final message = await pendingRequest.future;
        if (message is List<Float32List>) {
          for (final res in message) {
            yield res;
          }
        } else if (message is String && message.startsWith('ERROR:')) {
          throw Model2VecException(message);
        } else {
          throw StateError('Unexpected response from worker isolate: $message');
        }
      }

      await for (final text in texts) {
        buffer.add(text);

        if (buffer.length >= batchSize) {
          final currentBatch = buffer;
          buffer = <String>[];
          yield* flushBuffer(currentBatch);
        }
      }

      if (buffer.isNotEmpty) {
        yield* flushBuffer(buffer);
      }
    } finally {
      if (workerSendPort != null) {
        workerSendPort.send('close');
      }

      isolate?.kill();
      await sub?.cancel();
      mainReceivePort.close();
    }
  }

  static void _streamWorker(SendPort mainSendPort) {
    final workerReceivePort = ReceivePort();
    mainSendPort.send(workerReceivePort.sendPort);

    workerReceivePort.listen((message) {
      if (message == 'close') {
        workerReceivePort.close();
        return;
      }

      try {
        final data = message as ({List<String> batch, int maxLength});

        final results = Model2Vec.instance.generateBatchEmbeddings(
          data.batch,
          maxLength: data.maxLength,
          batchSize: data.batch.length,
        );
        mainSendPort.send(results);
      } on Object catch (e) {
        mainSendPort.send('ERROR: $e');
      }
    });
  }

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
