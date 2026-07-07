import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

import 'worker_support.dart';

void main() {
  group('EmbeddingPool', () {
    test('starts with the requested number of workers', () async {
      final pool = await EmbeddingPool.start(
        size: 3,
        entryPoint: echoEntryPoint,
      );
      addTearDown(pool.close);
      expect(pool.size, 3);
    });

    test('clamps a non-positive size to at least one worker', () async {
      final pool = await EmbeddingPool.start(
        size: 0,
        entryPoint: echoEntryPoint,
      );
      addTearDown(pool.close);
      expect(pool.size, 1);
    });

    test('embedBatch returns results', () async {
      final pool = await EmbeddingPool.start(
        size: 2,
        entryPoint: echoEntryPoint,
      );
      addTearDown(pool.close);
      final result = await pool.embedBatch(['a', 'bb']);
      expect(result.map((v) => v.first).toList(), [1.0, 2.0]);
    });

    test('embedBatches preserves input order under parallelism', () async {
      final pool = await EmbeddingPool.start(
        size: 3,
        entryPoint: echoEntryPoint,
      );
      addTearDown(pool.close);
      final results = await pool.embedBatches([
        ['a'],
        ['bb'],
        ['ccc'],
        ['dddd'],
        ['eeeee'],
      ]);
      expect(
        results.map((r) => r.first.first).toList(),
        [1.0, 2.0, 3.0, 4.0, 5.0],
      );
    });

    test('spreads concurrent batches across distinct workers', () async {
      final pool = await EmbeddingPool.start(
        size: 3,
        entryPoint: countingEntryPoint,
      );
      addTearDown(pool.close);

      // Fire one batch per worker at once. Least-busy dispatch should hand each
      // to a different worker, so every reply is that worker's *first* request.
      final results = await Future.wait([
        for (var i = 0; i < pool.size; i++) pool.embedBatch(['x']),
      ]);

      final counts = results.map((r) => r.single.first).toList();
      expect(counts, everyElement(1.0));
      expect(counts, hasLength(pool.size));
    });

    test(
      'close rejects in-flight and queued batches without hanging',
      () async {
        final pool = await EmbeddingPool.start(
          size: 2,
          entryPoint: countingEntryPoint,
        );

        // Launch more batches than workers so some are queued behind others, then
        // close before any of the (delayed) replies land.
        final batches = [
          for (var i = 0; i < 5; i++) pool.embedBatch(['a']),
        ];
        final expectations = [
          for (final batch in batches)
            expectLater(batch, throwsA(isA<Model2VecException>())),
        ];

        await pool.close();
        await Future.wait(expectations);
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test('recovers after a failing batch and serves later batches', () async {
      // A single worker fails its first request, then echoes. If the pool
      // failed to decrement its in-flight counter on the error path, later
      // batches would still have to land — assert they do.
      final pool = await EmbeddingPool.start(
        size: 1,
        entryPoint: failThenEchoEntryPoint,
      );
      addTearDown(pool.close);

      await expectLater(
        pool.embedBatch(['a']),
        throwsA(isA<Model2VecException>()),
      );

      final result = await pool.embedBatch(['abcd']);
      expect(result.single.first, 4.0);
    });

    test(
      'start throws and cleans up when a later worker fails to spawn',
      () async {
        resetPartialStartSlots();
        addTearDown(resetPartialStartSlots);

        // The pool is larger than the number of spawns the entry point lets
        // through, so some workers start and at least one fails. start() must
        // close the survivors and throw rather than leak orphaned isolates.
        await expectLater(
          EmbeddingPool.start(
            size: partialStartSuccessLimit + 1,
            entryPoint: flakyStartEntryPoint,
          ),
          throwsA(isA<StateError>()),
        );
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test('close tears down all workers', () async {
      final pool = await EmbeddingPool.start(
        size: 2,
        entryPoint: echoEntryPoint,
      );
      await pool.close();
      await expectLater(
        pool.embedBatch(['a']),
        throwsA(isA<Model2VecException>()),
      );
    });
  });
}
