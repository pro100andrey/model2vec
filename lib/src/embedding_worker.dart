import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'channel.dart';
import 'exception.dart';
import 'model2vec_base.dart';
import 'worker_protocol.dart';

/// A background worker that produces embeddings off the caller's thread, one
/// batch at a time.
///
/// Private to the package: callers reach it through
/// [Model2Vec.generateEmbeddingStream]. It wires an [IsolateChannel] to a
/// [WorkerProtocol]; the isolate lifecycle and the request protocol are the
/// only concerns here.
class EmbeddingWorker {
  EmbeddingWorker._(this._protocol);

  final WorkerProtocol _protocol;

  /// Spawns the worker isolate and completes its handshake.
  ///
  /// [entryPoint] defaults to the model-backed [embeddingWorkerEntryPoint];
  /// tests inject a fake top-level entry point to exercise the protocol
  /// without a model.
  static Future<EmbeddingWorker> start({
    void Function(SendPort) entryPoint = embeddingWorkerEntryPoint,
  }) async {
    final channel = await IsolateChannel.start(entryPoint);
    return EmbeddingWorker._(WorkerProtocol(channel));
  }

  /// Embeds [batch] on the worker isolate.
  Future<List<Float32List>> embedBatch(
    List<String> batch, {
    int maxLength = 512,
  }) => _protocol.embedBatch(batch, maxLength: maxLength);

  /// Tears the worker down, failing any outstanding requests.
  Future<void> close() => _protocol.close();
}

/// Default worker isolate entry point.
///
/// Embeds each requested batch against the process-global model and replies
/// with the vectors, or with a typed [Model2VecException] on failure so the
/// error's [Model2VecErrorKind] and code survive the isolate boundary.
void embeddingWorkerEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }

    final request = message as EmbedRequest;
    try {
      final results = Model2Vec.generateBatchEmbeddings(
        request.batch,
        maxLength: request.maxLength,
        batchSize: request.batch.length,
      );
      mainSendPort.send(results);
    } on Model2VecException catch (e) {
      mainSendPort.send(e);
    } on Object catch (e) {
      mainSendPort.send(Model2VecException(Model2VecErrorKind.unknown, '$e'));
    }
  });
}
