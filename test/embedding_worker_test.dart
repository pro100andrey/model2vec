import 'dart:isolate';
import 'dart:typed_data';

import 'package:model2vec/src/embedding_worker.dart';
import 'package:model2vec/src/exception.dart';
import 'package:model2vec/src/worker_protocol.dart';
import 'package:test/test.dart';

/// Fake worker entry point: replies with one vector per input text, whose only
/// element is the text's length. No model, no native calls.
void echoEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    final request = message as EmbedRequest;
    final results = [
      for (final text in request.batch)
        Float32List.fromList([text.length.toDouble()]),
    ];
    mainSendPort.send(results);
  });
}

/// Fake worker entry point that always replies with a typed error.
void errorEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    mainSendPort.send(
      const Model2VecException(
        Model2VecErrorKind.tokenizationFailed,
        'boom',
        6,
      ),
    );
  });
}

/// Fake entry point that dies (closes its port, exiting the isolate) on the
/// first request without ever replying.
void dyingEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    receivePort.close(); // exit without replying
  });
}

/// Fake entry point that exits immediately, never completing the handshake.
void noHandshakeEntryPoint(SendPort mainSendPort) {
  // Returns at once; the isolate exits before sending its SendPort.
}

void main() {
  test('embedBatch returns results from the worker isolate', () async {
    final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
    final result = await worker.embedBatch(['a', 'bb', 'ccc']);
    expect(result.map((v) => v.first).toList(), [1.0, 2.0, 3.0]);
    await worker.close();
  });

  test('multiple batches preserve order over the real isolate', () async {
    final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
    final r1 = worker.embedBatch(['x']);
    final r2 = worker.embedBatch(['yy']);
    expect((await r1).first.first, 1.0);
    expect((await r2).first.first, 2.0);
    await worker.close();
  });

  test('a typed error survives the isolate boundary', () async {
    final worker = await EmbeddingWorker.start(entryPoint: errorEntryPoint);
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
    await worker.close();
  });

  test('close tears the worker down and later calls fail fast', () async {
    final worker = await EmbeddingWorker.start(entryPoint: echoEntryPoint);
    await worker.close();
    await expectLater(
      worker.embedBatch(['a']),
      throwsA(isA<Model2VecException>()),
    );
  });

  test('a request fails (does not hang) if the worker dies mid-request',
      () async {
    final worker = await EmbeddingWorker.start(entryPoint: dyingEntryPoint);
    await expectLater(
      worker.embedBatch(['a']),
      throwsA(isA<Model2VecException>()),
    );
    await worker.close();
  });

  test('start fails if the worker never completes the handshake', () async {
    await expectLater(
      EmbeddingWorker.start(entryPoint: noHandshakeEntryPoint),
      throwsA(isA<StateError>()),
    );
  });
}
