/// Exception thrown when a Model2Vec operation fails.
class Model2VecException implements Exception {
  const Model2VecException(this.message, [this.code]);

  /// Creates a [Model2VecException] from a native error code.
  factory Model2VecException.fromCode(int code, String context) {
    final msg = switch (code) {
      -1 => 'Model not initialized or invalid path.',
      -2 => 'Failed to load model from the provided path.',
      -3 => 'Failed to initialize model from memory bytes.',
      -4 => 'Native library internal error (Lock poisoned).',
      _ => 'Unknown native error.',
    };
    return Model2VecException('$context: $msg', code);
  }

  /// Error message describing the failure.
  final String message;

  /// Optional error code returned by the native library.
  final int? code;

  @override
  String toString() {
    if (code != null) {
      return 'Model2VecException (Code $code): $message';
    }
    return 'Model2VecException: $message';
  }
}
