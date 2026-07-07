import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';

import 'channel.dart';
import 'exception.dart';

/// The request payload sent to an embedding worker over a [Channel].
typedef EmbedRequest = ({List<String> batch, int maxLength});

class _Request {
  _Request(this.batch, this.maxLength) : completer = Completer();

  final List<String> batch;
  final int maxLength;
  final Completer<List<Float32List>> completer;
}

/// A serialized request/response protocol over a [Channel].
///
/// One request is in flight at a time; concurrent [embedBatch] calls queue and
/// are serviced in submission order. This mirrors the single active native
/// model — there is nothing to gain from overlapping requests, and serializing
/// keeps any model switch deterministically ordered against embedding calls.
///
/// The protocol is pure with respect to transport: it only talks to a
/// [Channel], so it can be driven by an in-memory fake in tests with no
/// isolate. If the channel closes on its own (the worker died), every
/// outstanding request is failed rather than left hanging.
class WorkerProtocol {
  WorkerProtocol(this._channel) {
    _subscription = _channel.incoming.listen(
      _onMessage,
      onDone: _onChannelClosed,
    );
  }

  final Channel _channel;
  late final StreamSubscription<Object?> _subscription;
  final _queue = Queue<_Request>();

  var _inFlight = false;
  var _acceptingRequests = true;
  var _disposed = false;

  /// Submits [batch] for embedding. Safe to call concurrently; calls are
  /// serviced one at a time, in submission order.
  Future<List<Float32List>> embedBatch(
    List<String> batch, {
    required int maxLength,
  }) {
    if (!_acceptingRequests) {
      return .error(
        const Model2VecException(.unknown, 'embedding worker is closed'),
      );
    }

    final request = _Request(batch, maxLength);
    _queue.add(request);
    _pump();

    return request.completer.future;
  }

  void _pump() {
    if (_inFlight || _queue.isEmpty) {
      return;
    }

    _inFlight = true;
    final head = _queue.first;
    _channel.send((batch: head.batch, maxLength: head.maxLength));
  }

  void _onMessage(Object? message) {
    if (!_inFlight || _queue.isEmpty) {
      return; // stray reply; nothing is awaiting it
    }

    final request = _queue.removeFirst();
    _inFlight = false;

    if (message is List<Float32List>) {
      request.completer.complete(message);
    } else if (message is Model2VecException) {
      request.completer.completeError(message);
    } else {
      request.completer.completeError(
        StateError('unexpected worker reply: $message'),
      );
    }

    _pump();
  }

  /// The channel closed without us asking — the worker went away. Reject new
  /// requests and fail everything outstanding.
  void _onChannelClosed() {
    if (!_acceptingRequests) {
      return;
    }

    _acceptingRequests = false;
    _failAll('embedding worker terminated unexpectedly');
  }

  /// Closes the protocol, failing any queued or in-flight requests, then tears
  /// down the underlying [Channel]. Idempotent.
  Future<void> close() async {
    if (_disposed) {
      return;
    }

    _disposed = true;
    _acceptingRequests = false;
    await _subscription.cancel();
    _failAll('embedding worker closed before the request completed');
    await _channel.close();
  }

  void _failAll(String message) {
    final error = Model2VecException(.unknown, message);
    while (_queue.isNotEmpty) {
      _queue.removeFirst().completer.completeError(error);
    }
  }
}
