/// Groups a [source] stream into fixed-size batches.
///
/// Each batch holds [size] items and is emitted as soon as it fills; the final
/// batch may be smaller. This is a pure stream transformer with no isolate or
/// model dependency, so it is trivially testable in isolation.
///
/// Throws [ArgumentError] if [size] is less than 1.
Stream<List<T>> batched<T>(Stream<T> source, int size) async* {
  if (size < 1) {
    throw ArgumentError.value(size, 'size', 'must be >= 1');
  }

  var buffer = <T>[];
  await for (final item in source) {
    buffer.add(item);
    if (buffer.length >= size) {
      yield buffer;
      buffer = <T>[];
    }
  }

  if (buffer.isNotEmpty) {
    yield buffer;
  }
}
