import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('Model2VecException.fromNative', () {
    // Mirrors the CODE_* constants in native/src/lib.rs.
    const cases = {
      1: Model2VecErrorKind.notInitialized,
      2: Model2VecErrorKind.modelLoadFailed,
      3: Model2VecErrorKind.initFromBytesFailed,
      4: Model2VecErrorKind.lockPoisoned,
      5: Model2VecErrorKind.nullArgument,
      6: Model2VecErrorKind.tokenizationFailed,
      7: Model2VecErrorKind.emptyResult,
      8: Model2VecErrorKind.panic,
    };

    for (final entry in cases.entries) {
      test('code ${entry.key} maps to ${entry.value}', () {
        expect(
          Model2VecException.fromNative(entry.key, 'msg').kind,
          entry.value,
        );
      });
    }

    test('an unrecognized code maps to unknown', () {
      expect(
        Model2VecException.fromNative(99, 'msg').kind,
        Model2VecErrorKind.unknown,
      );
    });

    test('carries the native message and code', () {
      final e = Model2VecException.fromNative(2, 'boom');
      expect(e.message, 'boom');
      expect(e.code, 2);
      expect(e.toString(), contains('boom'));
    });
  });

  group('native error path', () {
    test('initEmbedderFromBytes with garbage throws a typed exception', () {
      final garbage = Uint8List.fromList([0, 1, 2, 3]);
      expect(
        () => Model2Vec.initEmbedderFromBytes(
          tokenizerBytes: garbage,
          modelBytes: garbage,
          configBytes: garbage,
        ),
        throwsA(
          isA<Model2VecException>()
              .having(
                (e) => e.kind,
                'kind',
                Model2VecErrorKind.initFromBytesFailed,
              )
              .having((e) => e.message, 'message', isNotEmpty),
        ),
      );
    });
  });
}
