import 'dart:async';
import 'dart:typed_data';

import 'package:model2vec/src/channel.dart';
import 'package:test/test.dart';

import 'worker_support.dart';

void main() {
  group('IsolateChannel', () {
    test('delivers worker replies over incoming', () async {
      final channel = await IsolateChannel.start(echoEntryPoint);
      addTearDown(channel.close);

      final reply = Completer<Object?>();
      channel.incoming.listen(reply.complete);
      channel.send((batch: ['ab', 'c'], maxLength: 8));

      final vectors = (await reply.future)! as List<Float32List>;
      expect(vectors.map((v) => v.first).toList(), [2.0, 1.0]);
    });

    test(
      'start fails when the worker exits before the handshake',
      () async {
        await expectLater(
          IsolateChannel.start(noHandshakeEntryPoint),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('exited during startup'),
            ),
          ),
        );
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test(
      'start rejects a non-SendPort handshake instead of hanging',
      () async {
        await expectLater(
          IsolateChannel.start(nonSendPortHandshakeEntryPoint),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('did not hand back a SendPort'),
            ),
          ),
        );
      },
      // Regression guard: a bad-handshake message must reject, not hang.
      // start()'s cleanup used to await _incoming.close() on a single-
      // subscription controller that was never listened (its close future never
      // completes) — now fire-and-forget.
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test(
      'incoming closes when the worker exits',
      () async {
        final channel = await IsolateChannel.start(dyingEntryPoint);
        addTearDown(channel.close);

        final done = Completer<void>();
        channel.incoming.listen((_) {}, onDone: done.complete);
        // The dying worker closes its port (exiting) instead of replying; the
        // exit must surface as incoming's onDone so a protocol above can't
        // hang.
        channel.send((batch: ['a'], maxLength: 8));

        await done.future;
      },
      timeout: const Timeout(Duration(seconds: 20)),
    );

    test('close is idempotent', () async {
      final channel = await IsolateChannel.start(echoEntryPoint);
      // Listen as a real caller (WorkerProtocol) does; close() awaits the
      // incoming controller's own close, which only completes once listened.
      channel.incoming.listen((_) {});
      await channel.close();
      await channel.close();
    }, timeout: const Timeout(Duration(seconds: 20)));
  });
}
