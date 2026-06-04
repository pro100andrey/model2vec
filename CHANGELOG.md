<!-- markdownlint-disable-file MD025 -->

# 1.1.0

**New Features:**

- `getRecommendedModels()` no longer calls FFI — now returns a hardcoded list of 7 models
- Removed `get_model_list` from FFI bindings (Rust, Dart, `.h`)
- `generateEmbedding()` now accepts `maxLength` parameter — signature changed
- `generateBatchEmbeddings()` now accepts `maxLength` and `batchSize` parameters — signature changed

- **Streaming API** — `generateEmbeddingStream()` for processing large datasets with batching and optional worker isolate
- **Async API** — `generateEmbeddingAsync()` and `generateBatchEmbeddingsAsync()` with `maxLength` / `batchSize` support
- **Advanced init** — `initEmbedderAdvanced()` with `hfToken`, `cacheDirectory`, `normalize`, `subfolder`
- **In-memory init** — `initEmbedderFromBytes()` for loading models from raw bytes
- **`boot()`** — manual initialization with a custom `DynamicLibrary`
- **`isNormalized`** — getter for L2-normalization check
- **`medianTokenLength`** — getter for median token length
- **`maxLength`** — token truncation parameter for `generateEmbedding()`
- **`batchSize`** — internal batching control for `generateBatchEmbeddings()`
- **`Model2VecUtils`** — vector math: `cosineSimilarity`, `dotProduct`, `euclideanDistance`, `similaritySearch`, `similaritySearchWithThreshold`, `cosineDistance`, `normalize`, `meanPooling`, `quantizeToInt8`, `toBase64`, `fromBase64`, `pairwiseSimilarity`

**Improvements:**

- **Streaming API Performance:** `generateEmbeddingStream()` now utilizes a single long-lived worker isolate instead of spawning one per batch, dramatically reducing IPC and memory overhead for large datasets.
- **Inter-Isolate Communication:** Switched from `Map<String, dynamic>` to Dart 3 Records for significantly faster and strictly typed isolate communication.
- **FFI Optimization:** `generateEmbedding()` in Rust rewritten to avoid array pointer allocations and correctly respect `max_length`.
- Refactored `quantizeToInt8()` to use Dart's native `.clamp()`.
- Added clear documentation for zero-vector handling in `cosineSimilarity` and `normalize`.
- Added documentation warning about IPC overhead in `generateEmbeddingStream` for CLI/Server applications.
- Better error messages when loading the native library fails, explaining possible missing Rust builds.
- Cleaned up FFI bindings: removed dead `get_model_list` symbol from `.h` and bindings.
- `generate_embedding` in Rust now returns `-5` on empty results instead of silently corrupting data.
- `generate_batch_embeddings_advanced` validates result count matches input count.
- Benchmark updated to run all 5 models.
- README fully rewritten with API reference and accurate model dimensions.

# 1.0.0

- Initial version.
