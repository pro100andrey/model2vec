import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:model2vec/src/channel.dart';
import 'package:model2vec/src/exception.dart';
import 'package:model2vec/src/worker_protocol.dart';
import 'package:path/path.dart' as p;

/// Shared fake isolate entry points for the concurrency tests.
///
/// Isolate entry points must be top-level functions (they can't be closures),
/// so the fakes that several test files need live here instead of being
/// duplicated. None of them touch the native model — they exercise the
/// isolate/protocol wiring only.

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
    mainSendPort.send(<Float32List>[
      for (final text in request.batch)
        Float32List.fromList([text.length.toDouble()]),
    ]);
  });
}

/// Fake worker entry point that always replies with a typed error, so tests can
/// check that a [Model2VecException] survives the isolate boundary.
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

/// Fake entry point whose first message is a plain value rather than a
/// [SendPort], to drive the "bad handshake" branch of [IsolateChannel.start].
void nonSendPortHandshakeEntryPoint(SendPort mainSendPort) {
  mainSendPort.send('not a SendPort');
}

/// Fake worker that reports, as the single element of each reply vector, how
/// many requests this specific isolate has handled so far, after a short delay
/// so concurrent calls genuinely overlap.
///
/// Because each spawned isolate has its own counter, a pool that spreads N
/// concurrent batches across N distinct workers yields all-ones (each worker's
/// first request); a pool that funnelled them to one worker would yield
/// 1, 2, 3, … instead. That makes the dispatch spread observable.
void countingEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  var handled = 0;
  receivePort.listen((message) async {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    final request = message as EmbedRequest;
    handled++;
    final count = handled;
    await Future<void>.delayed(const Duration(milliseconds: 50));
    mainSendPort.send(<Float32List>[
      for (final _ in request.batch) Float32List.fromList([count.toDouble()]),
    ]);
  });
}

/// Fake worker that fails its first request with a typed error and then behaves
/// like [echoEntryPoint]. Used to check the pool/protocol recover after an
/// error rather than wedging.
void failThenEchoEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  var failed = false;
  receivePort.listen((message) {
    if (message == 'close') {
      receivePort.close();
      return;
    }
    final request = message as EmbedRequest;
    if (!failed) {
      failed = true;
      mainSendPort.send(
        const Model2VecException(Model2VecErrorKind.panic, 'boom'),
      );
      return;
    }
    mainSendPort.send(<Float32List>[
      for (final text in request.batch)
        Float32List.fromList([text.length.toDouble()]),
    ]);
  });
}

/// How many [flakyStartEntryPoint] spawns complete their handshake before the
/// rest deliberately fail. Start a pool larger than this to force a partial
/// start failure.
const partialStartSuccessLimit = 2;

final _partialStartSlots =
    Directory(p.join(Directory.systemTemp.path, 'm2v_partial_start_slots'));

/// Fake entry point that succeeds for the first [partialStartSuccessLimit]
/// spawns and then exits before the handshake, so starting a larger pool leaves
/// some workers running and one (or more) failed.
///
/// The spawns race in parallel, so they claim a sequence number through
/// exclusive file creation — an atomic cross-isolate counter (isolates share no
/// memory). Call [resetPartialStartSlots] before and after use.
void flakyStartEntryPoint(SendPort mainSendPort) {
  final sequence = _claimStartSequence();
  if (sequence >= partialStartSuccessLimit) {
    return; // exit before the handshake — this spawn fails
  }
  echoEntryPoint(mainSendPort);
}

int _claimStartSequence() {
  _partialStartSlots.createSync(recursive: true);
  var sequence = 0;
  while (true) {
    try {
      File(p.join(_partialStartSlots.path, 'slot_$sequence'))
          .createSync(exclusive: true);
      return sequence;
    } on FileSystemException {
      sequence++;
    }
  }
}

/// Clears the [flakyStartEntryPoint] sequence counter so each test run starts
/// fresh regardless of leftovers from a previous run.
void resetPartialStartSlots() {
  if (_partialStartSlots.existsSync()) {
    _partialStartSlots.deleteSync(recursive: true);
  }
}
