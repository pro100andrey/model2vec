<!-- markdownlint-disable-file MD025 -->
# 2.0.0

Major release reworking the FFI boundary and public surface for testability and
correctness. **This release is breaking** — see migration below.

**Breaking changes:**

- **Static API.** `Model2Vec` is now a stateless namespace of static methods.
  `Model2Vec.instance`, the `Model2Vec(DynamicLibrary)` constructor and
  `Model2Vec.boot(...)` were removed — the native library is resolved
  automatically through Native Assets (`@Native` code assets). Replace
  `Model2Vec.instance.foo(...)` with `Model2Vec.foo(...)`.
- **Recommended models.** `getRecommendedModels()` (returning
  `List<Map<String, dynamic>>`) is replaced by the typed constant
  `Model2Vec.recommendedModels` (`List<RecommendedModel>`).
- **Typed errors.** `Model2VecException` now carries a `Model2VecErrorKind kind`;
  its constructor is `(kind, message, [code])` and the `fromCode` factory is
  replaced by `fromNative(code, message)`. Native failures surface the message
  produced by the Rust layer, each with an exhaustively-switchable `kind`.

**Improvements:**

- **Native memory safety.** The `generate_*` FFI functions now allocate their
  output inside the native call (returned as a pointer the caller frees),
  removing a dimension/model-switch race that could overflow the output buffer.
  Every native entry point is wrapped in `catch_unwind`, so a panic (including
  from a malformed model) surfaces as a typed error instead of undefined
  behaviour.
- **Windows ABI fix.** FFI length parameters use `size_t` (was `unsigned long`,
  32-bit on 64-bit Windows and mismatched against Rust's `usize`).
- **Streaming rework.** `generateEmbeddingStream` is rebuilt on small, tested
  modules — a batching transformer, a transport-agnostic worker protocol, and a
  worker isolate. Worker errors cross the isolate boundary as typed
  `Model2VecException`s (kind + code preserved) rather than stringified errors.

**Migration:**

| 1.x | 2.0.0 |
| --- | --- |
| `Model2Vec.instance.generateEmbedding(t)` | `Model2Vec.generateEmbedding(t)` |
| `Model2Vec.boot(lib)` / `Model2Vec(lib)` | removed — resolution is automatic |
| `Model2Vec.instance.getRecommendedModels()` | `Model2Vec.recommendedModels` (typed) |
| `catch (e) { e.code }` | still works; add `e.kind` for exhaustive handling |

# 1.2.0

- Lowered minimum Dart SDK requirement to `3.10.0` to support a wider range of environments.

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
