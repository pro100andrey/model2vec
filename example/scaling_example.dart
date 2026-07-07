import 'dart:io';

import 'package:model2vec/model2vec.dart';

/// Production, at-scale patterns:
///  1. load with a live download progress bar,
///  2. batch embedding in a single native call,
///  3. parallel embedding across CPU cores with an [EmbeddingPool],
///  4. streaming a huge dataset with bounded memory.
///
/// Run with `dart run example/scaling_example.dart`.
Future<void> main() async {
  // 1. Load with a progress bar. On a cache hit this jumps straight to `Ready`;
  //    the byte counts only move while the weights actually download.
  const modelId = 'minishlab/potion-base-2M';
  stdout.writeln('Loading $modelId…');
  await for (final progress in Model2Vec.loadModelWithProgress(modelId)) {
    _renderProgress(progress);
  }
  stdout.writeln();

  // 2. Batch embedding — one FFI call, SIMD across the whole batch.
  final sentences = [
    'The quick brown fox.',
    'Jumps over the lazy dog.',
    'Embeddings are just vectors.',
  ];
  final batch = Model2Vec.generateBatchEmbeddings(sentences);
  stdout.writeln(
    'Batch: ${batch.length} vectors of dim ${batch.first.length}.',
  );

  // 3. Parallel embedding across CPU cores. The pool shares the already-loaded
  //    process-global model; each batch runs on the least-busy worker.
  final pool = await EmbeddingPool.start(size: 4);
  try {
    final results = await pool.embedBatches([
      ['alpha', 'beta', 'gamma'],
      ['delta', 'epsilon'],
      ['zeta'],
    ]);
    final total = results.fold<int>(0, (sum, b) => sum + b.length);
    stdout.writeln('Pool: embedded $total texts across ${pool.size} workers.');
  } finally {
    await pool.close();
  }

  // 4. Streaming — process a large Stream<String> in batches without holding
  //    the whole dataset (or all of its vectors) in memory at once.
  final huge = Stream.fromIterable(List.generate(1000, (i) => 'Item $i'));
  var streamed = 0;
  await for (final _
      in Model2Vec.generateEmbeddingStream(huge, batchSize: 200)) {
    streamed++;
  }
  stdout.writeln('Streamed and embedded $streamed items.');
}

/// Renders a single load-progress snapshot as an in-place status line.
void _renderProgress(LoadProgress p) {
  final line = switch (p.phase) {
    LoadPhase.resolving => 'Resolving model files…',
    LoadPhase.downloading => _downloadBar(p),
    LoadPhase.parsing => 'Parsing & building model…',
    LoadPhase.done => 'Ready',
  };
  // `\r` rewrites the same line; pad to clear leftovers from a longer one.
  stdout.write('\r  ${line.padRight(56)}');
}

/// A `[███░░░] 42%  3.1/7.5 MB` style bar, or an indeterminate label until the
/// total size is known.
String _downloadBar(LoadProgress p) {
  final fraction = p.fraction;
  if (fraction == null) {
    return 'Downloading…';
  }
  const width = 24;
  final filled = (fraction * width).round();
  final bar = '${'█' * filled}${'░' * (width - filled)}';
  final mb = (p.bytesDownloaded / (1024 * 1024)).toStringAsFixed(1);
  final totalMb = (p.totalBytes / (1024 * 1024)).toStringAsFixed(1);
  return '[$bar] ${(fraction * 100).toStringAsFixed(0).padLeft(3)}%  '
      '$mb/$totalMb MB';
}
