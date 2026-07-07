import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('Model2VecException.fromNative', () {
    // Mirrors the CODE_* constants in native/src/lib.rs.
    const cases = <int, Model2VecErrorKind>{
      1: .notInitialized,
      2: .modelLoadFailed,
      3: .initFromBytesFailed,
      4: .lockPoisoned,
      5: .nullArgument,
      6: .tokenizationFailed,
      7: .emptyResult,
      8: .panic,
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
}
