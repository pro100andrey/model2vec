/// The stage a model load is currently in.
///
/// A load moves forward through these stages: [resolving] → (optionally)
/// [downloading] → [parsing] → [done]. The [downloading] stage is skipped when
/// the model is already cached or loaded from a local path.
enum LoadPhase {
  /// Locating the model files — resolving the repo layout and checking the
  /// local cache. No bytes have been transferred yet.
  resolving,

  /// Downloading the model weights from Hugging Face. This is the only stage
  /// with meaningful byte counts.
  downloading,

  /// Parsing the files and building the in-memory model. Fast; no byte counts.
  parsing,

  /// The model is loaded and ready. Always the final event of the stream.
  done,
}

/// A snapshot of a model load in progress, emitted by
/// `Model2Vec.loadModelWithProgress`.
///
/// Byte counts are only populated during [LoadPhase.downloading]; they stay `0`
/// when the size is unknown, on a cache hit, or for a local path. Use
/// [fraction] for a progress bar and fall back to an indeterminate spinner when
/// it is `null`.
final class LoadProgress {
  /// Creates a load-progress snapshot.
  const LoadProgress({
    required this.phase,
    this.bytesDownloaded = 0,
    this.totalBytes = 0,
  });

  /// The stage the load is currently in.
  final LoadPhase phase;

  /// Bytes of the model weights downloaded so far, or `0` when nothing has been
  /// downloaded yet (or the model was already cached).
  final int bytesDownloaded;

  /// Total size in bytes of the weights being downloaded, or `0` when it is not
  /// yet known (before the download starts, on a cache hit, or a local path).
  final int totalBytes;

  /// Download completion in `[0.0, 1.0]`, or `null` when the total size is not
  /// known — before the download starts, on a cache hit, or for a local path.
  double? get fraction => totalBytes > 0 ? bytesDownloaded / totalBytes : null;

  @override
  String toString() =>
      'LoadProgress(phase: ${phase.name}, bytesDownloaded: $bytesDownloaded, '
      'totalBytes: $totalBytes)';
}
