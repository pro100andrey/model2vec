import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'embedding_worker.dart';

/// A pool of worker isolates that embed batches concurrently.
///
/// The native model is a single process-global behind a read/write lock, and
/// the Rust engine allows concurrent readers — so N worker isolates genuinely
/// parallelize embedding across CPU cores. Use this for large workloads where a
/// single `Model2Vec.generateEmbeddingStream` worker is the bottleneck.
///
/// A model must be initialized (via `Model2Vec.initEmbedder`) before use, and
/// must not be switched while the pool is working (the one-model-per-run
/// contract, same as the single worker).
class EmbeddingPool {
  EmbeddingPool._(this._workers)
      : _inFlight = List<int>.filled(_workers.length, 0);

  final List<EmbeddingWorker> _workers;
  final List<int> _inFlight;

  /// Number of worker isolates in the pool.
  int get size => _workers.length;

  /// Spawns a pool of [size] workers (defaulting to the number of CPU cores).
  ///
  /// [entryPoint] defaults to the model-backed worker; tests inject a fake one.
  static Future<EmbeddingPool> start({
    int? size,
    void Function(SendPort) entryPoint = embeddingWorkerEntryPoint,
  }) async {
    final requested = size ?? Platform.numberOfProcessors;
    final count = requested < 1 ? 1 : requested;
    // Start all in parallel; if any worker fails to start, close the ones that
    // did so we never leak orphaned isolates.
    final outcomes = await Future.wait(
      List.generate(count, (_) async {
        try {
          return await EmbeddingWorker.start(entryPoint: entryPoint);
        } on Object {
          return null;
        }
      }),
    );
    final workers = outcomes.whereType<EmbeddingWorker>().toList();
    if (workers.length != count) {
      await Future.wait(workers.map((worker) => worker.close()));
      throw StateError('failed to start $count embedding workers');
    }
    return EmbeddingPool._(workers);
  }

  /// Embeds [batch] on the least-busy worker. Safe to call concurrently — each
  /// call runs on a (potentially) different worker, in parallel.
  Future<List<Float32List>> embedBatch(
    List<String> batch, {
    int maxLength = 512,
  }) {
    final worker = _leastBusyIndex();
    _inFlight[worker]++;
    return _workers[worker]
        .embedBatch(batch, maxLength: maxLength)
        .whenComplete(() => _inFlight[worker]--);
  }

  /// Embeds every batch in [batches] concurrently across the pool and returns
  /// the results in the same order as the input.
  ///
  /// All batches are dispatched at once; for very large inputs, feed them in
  /// windows to bound memory.
  Future<List<List<Float32List>>> embedBatches(
    List<List<String>> batches, {
    int maxLength = 512,
  }) => Future.wait(
    batches.map((batch) => embedBatch(batch, maxLength: maxLength)),
  );

  /// Tears down every worker in the pool.
  Future<void> close() async {
    await Future.wait(_workers.map((worker) => worker.close()));
  }

  int _leastBusyIndex() {
    var best = 0;
    for (var i = 1; i < _inFlight.length; i++) {
      if (_inFlight[i] < _inFlight[best]) {
        best = i;
      }
    }
    return best;
  }
}
