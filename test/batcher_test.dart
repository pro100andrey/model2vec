import 'package:model2vec/src/batcher.dart';
import 'package:test/test.dart';

void main() {
  group('batched', () {
    test('groups into full batches with a smaller final remainder', () async {
      final result = await batched(Stream.fromIterable([1, 2, 3, 4, 5]), 2)
          .toList();
      expect(result, [
        [1, 2],
        [3, 4],
        [5],
      ]);
    });

    test('an exact multiple leaves no remainder', () async {
      final result = await batched(Stream.fromIterable([1, 2, 3, 4]), 2)
          .toList();
      expect(result, [
        [1, 2],
        [3, 4],
      ]);
    });

    test('size larger than the input yields a single batch', () async {
      final result = await batched(Stream.fromIterable([1, 2]), 10).toList();
      expect(result, [
        [1, 2],
      ]);
    });

    test('size 1 yields singletons', () async {
      final result = await batched(Stream.fromIterable([1, 2, 3]), 1).toList();
      expect(result, [
        [1],
        [2],
        [3],
      ]);
    });

    test('an empty stream yields nothing', () async {
      final result = await batched(const Stream<int>.empty(), 3).toList();
      expect(result, isEmpty);
    });

    test('rejects a size below 1', () {
      expect(
        batched(Stream.fromIterable([1]), 0).toList(),
        throwsArgumentError,
      );
    });
  });
}
