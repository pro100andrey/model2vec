<!-- markdownlint-disable-file MD025 -->
# 2.0.3

Fixes the other half of 2.0.2: the hook stopped declaring the build tree as an
input, but still BUILT into it — and the build tree is in the pub cache, which
every project on the machine shares.

- **Cargo now builds into the directory the invoker hands out**
  (`out_dir_shared`), passed as `--target-dir`, instead of defaulting to
  `<crate>/native/target`. That directory is unique per hook per project and the
  runner serialises concurrent invocations into it, which is what it is
  documented for: "shared output and intermediate artifacts".

  The old default put one cargo build tree under
  `~/.pub-cache/hosted/pub.dev/model2vec-*/native/target` for every consumer at
  once. The hook runner tracks the artifact it was handed, and macOS stamps a
  fresh `LC_UUID` into every link, so a relink is a content change even when no
  source moved. Two consumers building under different configs — `dart test`
  runs with `linking_enabled: false` and `dart build` with `true`, and a second
  checkout or a Flutter workspace counts too — therefore invalidated each
  other's hook cache merely by building.

  What that looked like downstream: the runner deletes `output.json` before
  re-running a hook, so whatever a parallel `dart test` had already scheduled
  failed with `No asset with id 'package:model2vec/model2vec.so'`. The
  diagnosis is hard because the failure names neither this package nor the
  build that caused it, and re-running after a warm-up makes it disappear — it
  had been living in a dependent repository's contributor guide as "run this
  command first when the suite goes red", which is a workaround with a
  documentation entry rather than a fix.

  Writing into the pub cache at all is the underlying mistake; a package
  directory there is shared, and on some setups read-only.

- `test/build_hook_test.dart` pins it: the registered asset must live under the
  invoker's `out_dir_shared`, and no path component may be `native`. Verified
  discriminating — with the previous line restored it fails naming the pub-cache
  path it built into.

Also a performance pass over the native core. Embedding output is bit-identical
up to float rounding (verified ≤ 1.5e-7 against the previous code on a real
model); the pooling semantics are now pinned by 12 Rust unit tests.

- **Model loading got measurably faster and lighter.** The unk-token id is read
  straight off the tokenizer's model instead of serializing the entire
  tokenizer — vocabulary included — to a JSON tree to look up one field, and
  the weights file is memory-mapped instead of read into RAM, so parsing no
  longer holds both the raw file and the decoded f32 table at once (for
  `potion-multilingual-128M` that second copy alone was ~250 MB). Warm-cache
  load of `potion-base-8M`: 31 ms → 21 ms; `potion-base-32M`: 83 ms → 52 ms.

- **Batch pooling now runs in parallel** (rayon, already in the dependency tree
  via `tokenizers`): each sentence pools into its own disjoint chunk of the one
  flat output buffer. Single-sentence calls keep the sequential path and skip
  the thread pool. On short-text batches the gain is invisible — tokenization
  dominates — it shows on long texts and grows with embedding dimension.

- Assorted per-call overhead removed from the encode path: sentences are no
  longer copied into fresh `String`s before tokenization (`encode_batch_fast`
  takes `&str`), token ids are pooled straight off the encoding instead of
  being copied to a `Vec` for unk-filtering and truncation, and the final
  mean-and-normalize is one multiply pass instead of two division passes.

- Rust dependencies raised to current: `safetensors` 0.7 → 0.8 (no API change
  for our usage), `ureq` 3.4, `rayon` 1.12, plus `anyhow`/`serde_json` patch
  bumps. `hf-hub` deliberately stays on 0.5, the last `ureq`-based line: 1.0
  is a reqwest/tokio redesign with a mandatory `hf-xet` dependency that grows
  the dependency tree from 200 to ~350 crates, and its `rustls-tls` flavour
  pulls `aws-lc-sys`, which needs `cmake` on every machine that builds the
  crate — this package builds on the consumer's machine via the native-assets
  hook, so that would become an install requirement for every user of the
  package (the `native-tls` flavour trades that for OpenSSL headers on Linux).

# 2.0.2

Fixes a build hook that could never be cached, so every build of a dependent
package paid to rebuild this crate.

- **The hook no longer declares its own build output as its input.** It looked
  for cargo's dep-info beside the artifact as `'$binaryPath.d'` —
  `libm2v_ffi.dylib.d`. Cargo writes one dep-info per *target*, named after the
  target: a crate built as `["staticlib", "cdylib"]` produces `libm2v_ffi.a`,
  `libm2v_ffi.dylib` and a single `libm2v_ffi.d`. The requested name is written
  on no platform, so the parse returned nothing **every time** and the fallback
  — `output.dependencies.add(.directory(nativeDir))` — was not a fallback but
  the only path ever taken. `native/` contains cargo's own `target/`, so the
  hook declared as its input a tree it rewrites on every run. Anything that
  writes there — a second consumer building the same crate — marks the hook
  dirty, and the runner reports `File modified during build. Build must be
  rerun.` and invokes it again. Measured on this repository against
  `hooks 2.1.0`, a file written into `target/` without touching `src/`:
  **1 re-run and 1.55 s** before, **0 and 0.85 s** after. The re-run is cheap
  only while cargo itself has
  nothing to do; when the re-run coincides with a real rebuild the cost is the
  rebuild, and in a dependent workspace that measured **68 s** against **3.5 s**
  settled, with the spawning tests of a parallel `dart test` timing out. See
  [dart-lang/native#1998](https://github.com/dart-lang/native/issues/1998) for
  the same shape reported against `native_toolchain_c`, and
  [dart.dev/tools/hooks](https://dart.dev/tools/hooks): dependencies are the
  inputs a hook reads, never the outputs it produces.
- **The dep-info parser no longer breaks on Windows paths.** It split the file
  on the first `':'`, which on Windows is the drive letter — so
  `C:\out\m2v_ffi.dll: C:\src\lib.rs` yielded `\out\m2v_ffi.dll` as a
  "dependency": a path that exists nowhere, with the hook still reporting
  success. That bug was dormant only because the filename above meant the
  parser was never reached, so fixing the filename alone would have shipped it.
  The separator is now the first `": "`. While there, the parser reads **every**
  artifact line rather than only the first, and honours cargo's `\` escape so
  a path containing a space stays one path.
- **When the dep-info really is missing**, the fallback now names the crate's
  actual inputs — `native/src/`, `Cargo.lock`, `rust-toolchain.toml` and the
  manifest — instead of the directory that holds the build tree.

`test/build_hook_test.dart` covers both: an integration test reads the
dependencies the hook actually writes for the host, and unit tests pin the
Windows, multi-line and escaped-space cases that a run on one platform cannot
produce. No API change; this is a build-time fix only.

# 2.0.1

Fixes a build hook that broke every Flutter app depending on this package.

- **`flutter run` no longer fails with "Building native assets failed".** The
  hook read `input.config.code` without first asking
  `input.config.buildCodeAssets`. That is fine for `dart build`, which only ever
  invokes hooks with code assets requested — but Flutter also runs them from its
  asset-bundling pass (`buildCodeAssets: null`), and with data assets behind a
  feature flag that pass arrives with `build_asset_types` **empty**. Reading
  `.code` there throws `StateError`, the hook exits 255, and Flutter reports the
  whole app's native-asset build as failed, including the pass that would have
  built the library. The hook now returns early when no code assets are asked
  for. `test/build_hook_test.dart` pins the empty case, which no local `dart`
  command reaches on its own.

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
- **Lifecycle naming.** The `initEmbedder*` methods are renamed to `loadModel*`,
  pairing `loadModel` ⇄ `unloadModel` over the model. `initEmbedder`,
  `initEmbedderAdvanced`, `initEmbedderFromBytes` and their async forms are
  removed. `Model2VecUtils.similaritySearch` /`similaritySearchWithThreshold`
  are removed in favour of `similaritySearchWithScores` (read `.index`).
- **Batch signature.** `generateBatchEmbeddings` no longer takes `batchSize`
  (its signature is now `(List<String> texts, {int maxLength})`). The native
  layer batches internally; `batchSize` remains only on
  `generateEmbeddingStream`, which still controls its per-batch size.

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

**New capabilities:**

- **Local vector index.** `EmbeddingIndex` — store embeddings by id, then
  `search` the nearest by cosine similarity. Optional int8-quantized storage
  (~4x less memory) and binary `toBytes`/`fromBytes` persistence. Turns the
  package into a local retrieval engine for RAG.
- **RAG pipeline helpers.** `chunkText` (overlapping character chunker),
  `Model2VecUtils.similaritySearchWithScores` (index + score), and
  `Model2VecUtils.maximalMarginalRelevance` (MMR reranking for diverse results).
- **Lifecycle & DX.** `Model2Vec.isInitialized` (non-throwing check),
  `Model2Vec.unloadModel()` (free the native model), `Model2Vec.modelInfo`
  (all metadata in one `ModelInfo`), and `Model2VecUtils.dequantizeInt8`
  (the inverse of `quantizeToInt8`).
- **Load progress.** `Model2Vec.loadModelWithProgress()` loads on a background
  isolate and returns a `Stream<LoadProgress>` reporting the HF weights download
  (`bytesDownloaded` / `totalBytes` / `fraction`) plus a coarse `LoadPhase`
  (resolving → downloading → parsing → done). A cached model or local path
  streams straight to `done`.
- **Parallel worker pool.** `EmbeddingPool` fans batches across N worker
  isolates to embed concurrently across CPU cores.

**Migration:**

| 1.x | 2.0.0 |
| --- | --- |
| `Model2Vec.instance.generateEmbedding(t)` | `Model2Vec.generateEmbedding(t)` |
| `Model2Vec.boot(lib)` / `Model2Vec(lib)` | removed — resolution is automatic |
| `Model2Vec.instance.getRecommendedModels()` | `Model2Vec.recommendedModels` (typed) |
| `Model2Vec.instance.initEmbedder(path)` | `Model2Vec.loadModel(path)` |
| `Model2VecUtils.similaritySearch(q, c)` | `similaritySearchWithScores(q, c).map((r) => r.index)` |
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
