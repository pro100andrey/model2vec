/// Exception thrown when a Model2Vec operation fails.
class Model2VecException implements Exception {
  const Model2VecException(this.message, [this.code]);

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
