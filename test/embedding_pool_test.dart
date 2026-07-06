import 'dart:isolate';
import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:model2vec/src/worker_protocol.dart';
import 'package:test/test.dart';

/// Fake worker entry point: one vector per input text, whose only element is
/// the text's length. No model, no native calls.
void echoEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    final request = message as EmbedRequest;
    mainSendPort.send([
      for (final text in request.batch)
        Float32List.fromList([text.length.toDouble()]),
    ]);
  });
}

void main() {
  test('starts with the requested number of workers', () async {
    final pool = await EmbeddingPool.start(size: 3, entryPoint: echoEntryPoint);
    expect(pool.size, 3);
    await pool.close();
  });

  test('clamps a non-positive size to at least one worker', () async {
    final pool = await EmbeddingPool.start(size: 0, entryPoint: echoEntryPoint);
    expect(pool.size, 1);
    await pool.close();
  });

  test('embedBatch returns results', () async {
    final pool = await EmbeddingPool.start(size: 2, entryPoint: echoEntryPoint);
    final result = await pool.embedBatch(['a', 'bb']);
    expect(result.map((v) => v.first).toList(), [1.0, 2.0]);
    await pool.close();
  });

  test('embedBatches preserves input order under parallelism', () async {
    final pool = await EmbeddingPool.start(size: 3, entryPoint: echoEntryPoint);
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
    await pool.close();
  });

  test('close tears down all workers', () async {
    final pool = await EmbeddingPool.start(size: 2, entryPoint: echoEntryPoint);
    await pool.close();
    await expectLater(
      pool.embedBatch(['a']),
      throwsA(isA<Model2VecException>()),
    );
  });
}
