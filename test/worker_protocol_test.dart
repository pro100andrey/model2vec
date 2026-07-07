import 'dart:async';
import 'dart:typed_data';

import 'package:model2vec/src/channel.dart';
import 'package:model2vec/src/exception.dart';
import 'package:model2vec/src/worker_protocol.dart';
import 'package:test/test.dart';

/// An in-memory [Channel] the test drives directly — no isolate, no model.
class FakeChannel implements Channel {
  final sent = <Object?>[];
  final _incoming = StreamController<Object?>();
  // ignore: omit_obvious_property_types  public field needs the annotation
  bool closed = false;

  @override
  void send(Object? message) => sent.add(message);

  @override
  Stream<Object?> get incoming => _incoming.stream;

  @override
  Future<void> close() async {
    closed = true;
    await _incoming.close();
  }

  /// Simulates a reply arriving from the worker end.
  void deliver(Object? message) => _incoming.add(message);
}

Float32List _vec(List<double> xs) => Float32List.fromList(xs);

void main() {
  group('WorkerProtocol', () {
    late FakeChannel channel;
    late WorkerProtocol protocol;

    setUp(() {
      channel = FakeChannel();
      protocol = WorkerProtocol(channel);
    });

    test('sends the request and resolves with the delivered result', () async {
      final future = protocol.embedBatch(['a'], maxLength: 8);
      expect(channel.sent, hasLength(1)); // sent synchronously

      final result = [
        _vec([1, 2, 3]),
      ];
      channel.deliver(result);
      expect(await future, same(result));
    });

    test('serializes: the second request waits for the first reply', () async {
      final f1 = protocol.embedBatch(['a'], maxLength: 8);
      final f2 = protocol.embedBatch(['b'], maxLength: 8);

      // Only the first request is in flight.
      expect(channel.sent, hasLength(1));

      channel.deliver([
        _vec([1]),
      ]);
      await f1;

      // Now the second is sent.
      expect(channel.sent, hasLength(2));
      channel.deliver([
        _vec([2]),
      ]);
      await f2;
    });

    test('an empty batch resolves with an empty list', () async {
      final future = protocol.embedBatch([], maxLength: 8);
      expect(channel.sent, hasLength(1));
      channel.deliver(<Float32List>[]);
      expect(await future, isEmpty);
    });

    test('propagates a typed Model2VecException from the worker', () async {
      final future = protocol.embedBatch(['a'], maxLength: 8);
      channel.deliver(
        const Model2VecException(
          Model2VecErrorKind.notInitialized,
          'no model',
          1,
        ),
      );

      await expectLater(
        future,
        throwsA(
          isA<Model2VecException>().having(
            (e) => e.kind,
            'kind',
            Model2VecErrorKind.notInitialized,
          ),
        ),
      );
    });

    test('an unexpected reply shape becomes a StateError', () async {
      final future = protocol.embedBatch(['a'], maxLength: 8);
      channel.deliver('garbage');
      await expectLater(future, throwsStateError);
    });

    test('drops a stray reply when nothing is awaiting', () async {
      // A reply arrives with an empty queue — nothing awaits it, so it must be
      // dropped silently and leave the protocol able to serve real requests.
      channel.deliver([
        _vec([9]),
      ]);
      // Let the stray be processed (and dropped) before a real request exists;
      // otherwise it would land on the request submitted below.
      await pumpEventQueue();

      final future = protocol.embedBatch(['a'], maxLength: 8);
      channel.deliver([
        _vec([1]),
      ]);
      expect((await future).single.first, 1.0);
    });

    test(
      'close fails queued and in-flight requests and closes the channel',
      () async {
        final f1 = protocol.embedBatch(['a'], maxLength: 8);
        final f2 = protocol.embedBatch(['b'], maxLength: 8);

        // Attach listeners before close so the failed futures aren't reported
        // as unhandled (real callers await the future they submitted).
        final expect1 = expectLater(f1, throwsA(isA<Model2VecException>()));
        final expect2 = expectLater(f2, throwsA(isA<Model2VecException>()));

        await protocol.close();

        await expect1;
        await expect2;
        expect(channel.closed, isTrue);
      },
    );

    test('embedBatch after close fails fast', () async {
      await protocol.close();
      await expectLater(
        protocol.embedBatch(['a'], maxLength: 8),
        throwsA(isA<Model2VecException>()),
      );
    });

    test('close is idempotent', () async {
      await protocol.close();
      await protocol.close();
    });

    test(
      'fails a pending request when the channel closes on its own',
      () async {
        final future = protocol.embedBatch(['a'], maxLength: 8);
        final expectation = expectLater(
          future,
          throwsA(isA<Model2VecException>()),
        );
        // Simulates the worker dying: the channel closes while a request is in
        // flight, without protocol.close() first, so onDone reaches the
        // protocol.
        await channel.close();
        await expectation;
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test(
      'rejects new requests after the channel closes underneath',
      () async {
        // The channel closes with nothing outstanding; a *new* request must
        // then fail fast rather than wait for a reply that will never come.
        await channel.close();
        await expectLater(
          protocol.embedBatch(['a'], maxLength: 8),
          throwsA(isA<Model2VecException>()),
        );
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );
  });
}
