import 'dart:io';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('loadModelWithProgress', () {
    test('reports download progress and reaches done on a fresh cache',
        () async {
      // A private temp cache guarantees a cache miss, so the weights actually
      // download and the downloading phase is exercised.
      final tmp = Directory.systemTemp.createTempSync('m2v_progress_');
      addTearDown(() => tmp.deleteSync(recursive: true));

      final events = <LoadProgress>[];
      await for (final p in Model2Vec.loadModelWithProgress(
        'minishlab/potion-base-2M',
        cacheDirectory: tmp.path,
        pollInterval: const Duration(milliseconds: 1),
      )) {
        events.add(p);
      }

      // The load succeeded and the stream ended on `done`.
      expect(Model2Vec.isInitialized, isTrue);
      expect(events, isNotEmpty);
      expect(events.last.phase, LoadPhase.done);

      // `done` is only ever the terminal event.
      final nonTerminal = events.sublist(0, events.length - 1);
      expect(
        nonTerminal,
        everyElement(
          isNot(
            predicate<LoadProgress>((e) => e.phase == LoadPhase.done),
          ),
        ),
      );

      // A fresh cache must download the weights: we should have caught at least
      // one downloading snapshot with a known total and a real fraction.
      final downloading =
          events.where((e) => e.phase == LoadPhase.downloading).toList();
      expect(
        downloading,
        isNotEmpty,
        reason: 'a fresh cache should download the weights',
      );
      final last = downloading.last;
      expect(last.totalBytes, greaterThan(0));
      expect(last.bytesDownloaded, greaterThan(0));
      expect(last.fraction, isNotNull);
      expect(last.fraction, inInclusiveRange(0.0, 1.0));

      // Print the shape of the run for eyeballing.
      final byPhase = <LoadPhase, int>{};
      for (final e in events) {
        byPhase[e.phase] = (byPhase[e.phase] ?? 0) + 1;
      }
      // Diagnostic output for eyeballing a real run.
      // ignore: avoid_print
      print('phase counts: $byPhase; last downloading: $last');
    });

    test('a cached model streams straight to done', () async {
      // The prior test cached the model in a temp dir, but the default cache
      // may already hold it too; either way this must terminate on `done`.
      final events = <LoadProgress>[];
      await for (final p in Model2Vec.loadModelWithProgress(
        'minishlab/potion-base-2M',
      )) {
        events.add(p);
      }

      expect(Model2Vec.isInitialized, isTrue);
      expect(events.last.phase, LoadPhase.done);
    });
  });
}
