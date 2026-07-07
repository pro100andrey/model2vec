import 'dart:async';
import 'dart:isolate';

/// A bidirectional message channel — the transport a worker protocol runs on.
///
/// This is the seam that makes the worker testable: production uses
/// [IsolateChannel] over real isolate ports, while tests drive an in-memory
/// fake. The protocol on top of it never mentions isolates or ports.
///
/// [incoming] closes when the other end goes away — for [IsolateChannel] that
/// includes the worker isolate dying unexpectedly, so a protocol listening for
/// `onDone` can fail its pending requests instead of hanging forever.
abstract interface class Channel {
  /// Sends [message] to the other end.
  void send(Object? message);

  /// Messages arriving from the other end (excluding any handshake).
  Stream<Object?> get incoming;

  /// Tears the channel down, releasing the other end.
  Future<void> close();
}

/// A [Channel] backed by a dedicated worker [Isolate].
///
/// The worker's `entryPoint` receives a [SendPort] and must reply with its own
/// [SendPort] as the first message (the handshake); every later message is
/// surfaced on [incoming]. If the isolate exits — cleanly or by crashing —
/// [incoming] is closed so callers are never left waiting on a dead worker.
class IsolateChannel implements Channel {
  IsolateChannel._(
    this._isolate,
    this._workerSendPort,
    this._receivePort,
    this._exitPort,
    this._subscription,
    this._exitSubscription,
    this._incoming,
  );

  final Isolate _isolate;
  final SendPort _workerSendPort;
  final ReceivePort _receivePort;
  final ReceivePort _exitPort;
  final StreamSubscription<dynamic> _subscription;
  final StreamSubscription<dynamic> _exitSubscription;
  final StreamController<Object?> _incoming;

  var _closed = false;

  /// Spawns a worker running [entryPoint] and completes the handshake.
  static Future<IsolateChannel> start(
    void Function(SendPort) entryPoint,
  ) async {
    final receivePort = ReceivePort();
    final exitPort = ReceivePort();
    final isolate = await Isolate.spawn(
      entryPoint,
      receivePort.sendPort,
      onExit: exitPort.sendPort,
    );

    final incoming = StreamController<Object?>();
    final handshake = Completer<SendPort>();

    final subscription = receivePort.listen((message) {
      if (!handshake.isCompleted) {
        if (message is SendPort) {
          handshake.complete(message);
        } else {
          handshake.completeError(
            StateError('worker isolate did not hand back a SendPort'),
          );
        }
        return;
      }
      // The reply port and the exit port are independent; if the exit
      // notification already closed [incoming], drop this late message rather
      // than adding to a closed controller.
      if (!incoming.isClosed) {
        incoming.add(message);
      }
    });

    // The isolate exited. Fail a pending handshake, and close [incoming] so a
    // protocol awaiting a reply is released rather than hanging forever.
    final exitSubscription = exitPort.listen((_) {
      if (!handshake.isCompleted) {
        handshake.completeError(
          StateError('worker isolate exited during startup'),
        );
      }
      if (!incoming.isClosed) {
        unawaited(incoming.close());
      }
    });

    final SendPort workerSendPort;
    try {
      workerSendPort = await handshake.future;
    } on Object {
      await subscription.cancel();
      await exitSubscription.cancel();
      receivePort.close();
      exitPort.close();
      if (!incoming.isClosed) {
        // No one has listened to [incoming] yet (start hasn't returned), and a
        // single-subscription controller's close() future never completes
        // without a listener — so fire-and-forget instead of awaiting a hang.
        unawaited(incoming.close());
      }
      isolate.kill(priority: Isolate.immediate);
      rethrow;
    }

    return IsolateChannel._(
      isolate,
      workerSendPort,
      receivePort,
      exitPort,
      subscription,
      exitSubscription,
      incoming,
    );
  }

  @override
  void send(Object? message) => _workerSendPort.send(message);

  @override
  Stream<Object?> get incoming => _incoming.stream;

  @override
  Future<void> close() async {
    if (_closed) {
      return;
    }
    _closed = true;
    // Cancel the exit listener first so our own kill() below doesn't re-enter
    // the unexpected-exit path.
    await _exitSubscription.cancel();
    _workerSendPort.send('close');
    await _subscription.cancel();
    _isolate.kill();
    _receivePort.close();
    _exitPort.close();
    if (!_incoming.isClosed) {
      // Awaiting close() only completes once a listener drains the done event;
      // an unlistened single-subscription controller would hang here forever.
      if (_incoming.hasListener) {
        await _incoming.close();
      } else {
        unawaited(_incoming.close());
      }
    }
  }
}
