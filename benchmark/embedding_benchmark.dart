import 'dart:ffi';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:model2vec/model2vec.dart';
import 'package:path/path.dart' as p;

class EmbeddingBenchmark extends BenchmarkBase {
  EmbeddingBenchmark(this.modelId, this.text) : super('Embedding($modelId)');

  final String modelId;
  final String text;
  late Model2Vec model2vec;
  late DynamicLibrary lib;

  @override
  void setup() {
    final pkgRoot = Directory.current.path;
    String? libPath;

    // Search for library
    final dartToolPath = p.join(pkgRoot, '.dart_tool');
    if (Directory(dartToolPath).existsSync()) {
      final files = Directory(dartToolPath).listSync(recursive: true);
      for (final file in files) {
        if (file is File && file.path.endsWith('libm2v_ffi.so')) {
          libPath = file.path;
          break;
        }
      }
    }

    libPath ??= p.join(pkgRoot, 'native/target/release/libm2v_ffi.so');

    if (!File(libPath).existsSync()) {
      throw Exception('Library not found. Run "make build" first.');
    }

    lib = DynamicLibrary.open(libPath);
    model2vec = Model2Vec(lib);

    stdout.writeln('\n[Setup] Initializing $modelId...');
    final sw = Stopwatch()..start();
    model2vec.initEmbedder(modelId);
    sw.stop();

    stdout.writeln(
      '[Setup] $modelId initialized in ${sw.elapsedMilliseconds}ms. '
      'Dim: ${model2vec.embeddingDimension}',
    );
  }

  @override
  void run() {
    model2vec.generateEmbedding(text);
  }

  @override
  void teardown() {
    // Rust MODEL OnceCell can't be cleared easily in this FFI setup,
    // so we usually benchmark one model per process or just accept the shared
    // state if IDs were different.
    // But since our Rust side uses a static OnceCell, we can only test one
    // model per execution properly unless we modify Rust to support multiple
    // models.
  }
}

void main() {
  const testText =
      'The quick brown fox jumps over the lazy dog. '
      'Model2Vec is a fast and small model for text embeddings.';

  // Note: Due to OnceCell in Rust, we can only initialize ONE model per run in
  // current implementation.  To benchmark all models, we should run this script
  // multiple times with different arguments.

  final modelToBench =
      Platform.environment['BENCH_MODEL'] ?? 'minishlab/potion-base-2M';

  stdout
    ..writeln('=== Model2Vec Performance Benchmark ===')
    ..writeln('Target Model: $modelToBench')
    ..writeln('Input text length: ${testText.length} chars');

  EmbeddingBenchmark(modelToBench, testText).report();
}
