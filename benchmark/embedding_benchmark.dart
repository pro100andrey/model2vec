import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:model2vec/model2vec.dart';

const testText =
    'The quick brown fox jumps over the lazy dog. Model2Vec is a fast and '
    'small model for text embeddings.';

late List<String> batchTexts;

class EmbeddingBenchmark extends BenchmarkBase {
  EmbeddingBenchmark() : super('Single');

  @override
  void run() {
    Model2Vec.generateEmbedding(testText);
  }
}

class BatchEmbeddingBenchmark extends BenchmarkBase {
  BatchEmbeddingBenchmark() : super('Batch (32)');

  @override
  void run() {
    Model2Vec.generateBatchEmbeddings(batchTexts);
  }
}

void main() {
  batchTexts = List.generate(32, (i) => '$testText #$i');

  final models = [
    'minishlab/potion-base-2M',
    'minishlab/potion-base-4M',
    'minishlab/potion-base-8M',
    'minishlab/potion-base-32M',
    'minishlab/potion-multilingual-128M',
  ];

  stdout
    ..writeln('=== Model2Vec Performance Benchmark ===')
    ..writeln(
      'Warming up models (Downloading to cache, this may take a while)...',
    );

  for (final model in models) {
    stdout.write('Warming up $model... ');
    final watch = Stopwatch()..start();
    Model2Vec.initEmbedder(model);
    watch.stop();
    stdout.writeln('Done in ${watch.elapsedMilliseconds} ms.');
  }

  stdout
    ..writeln('\n--- Benchmark Results ---')
    ..writeln(
      '| Model | Load Time (Cache) | Single Embedding | Batch (32) |',
    )
    ..writeln('|---|---|---|---|');

  for (final model in models) {
    // Measure Load Time (Already cached)
    final watch = Stopwatch()..start();
    Model2Vec.initEmbedder(model);
    watch.stop();
    final loadTime = '${watch.elapsedMilliseconds} ms';

    // Measure Single Embedding
    final singleBench = EmbeddingBenchmark();
    final singleScore = singleBench.measure(); // Microseconds per run
    final singleStr = '${singleScore.toStringAsFixed(1)} μs';

    // Measure Batch Embedding
    final batchBench = BatchEmbeddingBenchmark();
    final batchScore = batchBench.measure(); // Microseconds per run
    final batchStr = '${(batchScore / 1000).toStringAsFixed(2)} ms';

    stdout.writeln('| `$model` | $loadTime | $singleStr | $batchStr |');
  }
}
