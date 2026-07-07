import 'package:model2vec/src/embedding_worker.dart';
import 'package:model2vec/src/exception.dart';
import 'package:test/test.dart';

import 'worker_support.dart';

void main() {
  group('EmbeddingWorker', () {
    test('embedBatch returns results from the worker isolate', () async {
      final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
      addTearDown(worker.close);
      final result = await worker.embedBatch(['a', 'bb', 'ccc']);
      expect(result.map((v) => v.first).toList(), [1.0, 2.0, 3.0]);
    });

    test('multiple batches preserve order over the real isolate', () async {
      final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
      addTearDown(worker.close);
      final r1 = worker.embedBatch(['x']);
      final r2 = worker.embedBatch(['yy']);
      expect((await r1).first.first, 1.0);
      expect((await r2).first.first, 2.0);
    });

    test('an empty batch round-trips as an empty list', () async {
      final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
      addTearDown(worker.close);
      expect(await worker.embedBatch([]), isEmpty);
    });

    test('a typed error survives the isolate boundary', () async {
      final worker = await EmbeddingWorker.start(entryPoint: errorEntryPoint);
      addTearDown(worker.close);
      await expectLater(
        worker.embedBatch(['a']),
        throwsA(
          isA<Model2VecException>().having(
            (e) => e.kind,
            'kind',
            Model2VecErrorKind.tokenizationFailed,
          ),
        ),
      );
    });

    test('close tears the worker down and later calls fail fast', () async {
      final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
      await worker.close();
      await expectLater(
        worker.embedBatch(['a']),
        throwsA(isA<Model2VecException>()),
      );
    });

    test('close is idempotent', () async {
      final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
      await worker.close();
      await worker.close();
    });

    test('a request fails (does not hang) if the worker dies mid-request',
        () async {
      final worker = await EmbeddingWorker.start(entryPoint: dyingEntryPoint);
      addTearDown(worker.close);
      await expectLater(
        worker.embedBatch(['a']),
        throwsA(isA<Model2VecException>()),
      );
    }, timeout: const Timeout(Duration(seconds: 20)));

    test('start fails if the worker never completes the handshake', () async {
      await expectLater(
        EmbeddingWorker.start(entryPoint: noHandshakeEntryPoint),
        throwsA(isA<StateError>()),
      );
    }, timeout: const Timeout(Duration(seconds: 20)));
  });
}
