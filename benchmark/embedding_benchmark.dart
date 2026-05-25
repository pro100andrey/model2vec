import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:model2vec/model2vec.dart';

class EmbeddingBenchmark extends BenchmarkBase {
  EmbeddingBenchmark(this.modelId, this.text)
    : super('Single Embedding($modelId)');

  final String modelId;
  final String text;
  late Model2Vec m2v;

  @override
  void setup() {
    m2v = Model2Vec.instance;
    m2v.initEmbedder(modelId);
  }

  @override
  void run() {
    m2v.generateEmbedding(text);
  }
}

class BatchEmbeddingBenchmark extends BenchmarkBase {
  BatchEmbeddingBenchmark(this.modelId, this.texts)
    : super('Batch Embedding($modelId, size=${texts.length})');

  final String modelId;
  final List<String> texts;
  late Model2Vec m2v;

  @override
  void setup() {
    m2v = Model2Vec.instance;
    m2v.initEmbedder(modelId);
  }

  @override
  void run() {
    m2v.generateBatchEmbeddings(texts);
  }
}

void main() {
  const testText =
      'The quick brown fox jumps over the lazy dog. '
      'Model2Vec is a fast and small model for text embeddings.';

  final batchTexts = List.generate(32, (i) => '$testText #$i');

  final modelId =
      Platform.environment['BENCH_MODEL'] ?? 'minishlab/potion-base-2M';

  stdout
    ..writeln('=== Model2Vec Performance Benchmark ===')
    ..writeln('Target Model: $modelId')
    ..writeln('Input text length: ${testText.length} chars');

  EmbeddingBenchmark(modelId, testText).report();
  BatchEmbeddingBenchmark(modelId, batchTexts).report();
}
