/// Classification of a failure that crossed the native (FFI) boundary.
///
/// Values mirror the stable error codes the Rust layer returns, so callers can
/// branch exhaustively with a `switch` instead of matching magic numbers.
enum Model2VecErrorKind {
  /// No model has been initialized yet (call `loadModel` first).
  notInitialized,

  /// A model could not be loaded from the given repo id or path.
  modelLoadFailed,

  /// A model could not be initialized from the provided bytes.
  initFromBytesFailed,

  /// The native model lock was poisoned by a previous panic.
  lockPoisoned,

  /// A required argument was null or otherwise invalid.
  nullArgument,

  /// Tokenization of the input failed.
  tokenizationFailed,

  /// The native layer produced no result where one was expected.
  emptyResult,

  /// A panic was caught at the native boundary.
  panic,

  /// An error code the current Dart version does not recognize.
  unknown,
}

/// Exception thrown when a Model2Vec operation fails.
class Model2VecException implements Exception {
  /// Creates an exception with an explicit [kind] and [message].
  const Model2VecException(this.kind, this.message, [this.code]);

  /// Builds an exception from a native status [code] and the native [message].
  factory Model2VecException.fromNative(int code, String message) =>
      Model2VecException(_kindFromCode(code), message, code);

  /// The category of failure, suitable for exhaustive handling.
  final Model2VecErrorKind kind;

  /// Human-readable description. When the failure originated in the native
  /// layer, this is the message that layer produced.
  final String message;

  /// The raw native status code, when the failure came from the native layer.
  final int? code;

  @override
  String toString() => code != null
      ? 'Model2VecException(${kind.name}, code $code): $message'
      : 'Model2VecException(${kind.name}): $message';
}

/// Maps a stable native status code to its [Model2VecErrorKind].
///
/// Keep in sync with the `CODE_*` constants in `native/src/lib.rs`.
Model2VecErrorKind _kindFromCode(int code) => switch (code) {
  1 => Model2VecErrorKind.notInitialized,
  2 => Model2VecErrorKind.modelLoadFailed,
  3 => Model2VecErrorKind.initFromBytesFailed,
  4 => Model2VecErrorKind.lockPoisoned,
  5 => Model2VecErrorKind.nullArgument,
  6 => Model2VecErrorKind.tokenizationFailed,
  7 => Model2VecErrorKind.emptyResult,
  8 => Model2VecErrorKind.panic,
  _ => Model2VecErrorKind.unknown,
};
