# model2vec

[![pub package](https://img.shields.io/pub/v/model2vec.svg)](https://pub.dev/packages/model2vec)
[![CI](https://github.com/pro100andrey/model2vec/actions/workflows/ci.yml/badge.svg)](https://github.com/pro100andrey/model2vec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Fast, on-device text embeddings for Dart & Flutter.** A wrapper around
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) via Rust FFI and Native
Assets. Model2Vec turns text into vectors with a static vocabulary lookup — not a
transformer — so embeddings are generated in **microseconds**, with no server, no
Python, and no network after the model is cached.

## Features

- **Fast & local** — embeddings in microseconds; fully offline once a model is cached.
- **Hugging Face models** — load any Potion model by id (auto-download + cache), a local directory, or raw bytes.
- **Scales** — batch embedding (Rust SIMD), background isolates, a streaming API for millions of rows, and a worker pool across CPU cores.
- **Built-in retrieval** — an on-device `EmbeddingIndex` (cosine search, int8 quantization, disk persistence) plus vector math (cosine, MMR, pooling, quantization).
- **Native Assets** — the Rust library builds automatically on `pub get`; nothing to link, bundle, or ship.

## Installation

```bash
dart pub add model2vec
```

Requirements:

- **Dart SDK** 3.10+.
- **Rust toolchain** ([rustup](https://rustup.rs)). The native library is compiled
  automatically via Native Assets; the exact version is pinned in
  [`native/rust-toolchain.toml`](native/rust-toolchain.toml) and installed for you.

## Quick start

```dart
import 'package:model2vec/model2vec.dart';

void main() {
  // One active model per process; the native library loads automatically.
  Model2Vec.loadModel('minishlab/potion-base-2M'); // downloads on first run

  final a = Model2Vec.generateEmbedding('I love programming in Dart');
  final b = Model2Vec.generateEmbedding('Dart is a great language');

  print(Model2VecUtils.cosineSimilarity(a, b)); // ~0.8 — semantically close
}
```

> In a Flutter app, use `loadModelAsync` instead so the first download never
> blocks the UI (see [Loading off the main thread](#loading-off-the-main-thread)).

**Migrating from 1.x?** The instance API is gone: replace
`Model2Vec.instance.foo(...)` with `Model2Vec.foo(...)`, drop `Model2Vec.boot(...)`
/ `Model2Vec(lib)` (resolution is automatic), and read
`Model2Vec.recommendedModels` instead of `getRecommendedModels()`. Full list in the
[CHANGELOG](CHANGELOG.md).

## Models

Load any of these by id, or browse the typed catalog at `Model2Vec.recommendedModels`.

| Model | Params | Dim | Language | Best for |
| ----- | ------ | --- | -------- | -------- |
| [`potion-base-2M`](https://huggingface.co/minishlab/potion-base-2M) | 1.8M | 64 | English | Smallest, very fast |
| [`potion-base-4M`](https://huggingface.co/minishlab/potion-base-4M) | 3.7M | 128 | English | Small and efficient |
| [`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | 7.5M | 256 | English | Balanced default |
| [`potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) | 32.3M | 512 | English | Large and accurate |
| [`potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M) | 32.3M | 512 | English | RAG / retrieval |
| [`potion-code-16M`](https://huggingface.co/minishlab/potion-code-16M) | 16M | 384 | Code | Code search |
| [`potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) | 128M | 768 | 101 languages | Multilingual tasks |

## Guide

Runnable versions of everything below live in [`example/`](example/).

### Generating embeddings

```dart
// Single vector
final v = Model2Vec.generateEmbedding('Hello world');

// Batch — one native call, SIMD across the batch. maxLength truncates long input.
final batch = Model2Vec.generateBatchEmbeddings(
  ['Dart', 'Rust', 'Flutter'],
  maxLength: 256,
);
```

For datasets too large to hold in memory, stream them — the input is processed in
batches and the output is a `Stream<Float32List>`:

```dart
final vectors = Model2Vec.generateEmbeddingStream(
  lines,               // a Stream<String> from a file, DB, socket…
  batchSize: 500,
  useIsolate: true,    // run the work off the main isolate
);

await for (final v in vectors) {
  save(v);             // bounded memory, whatever the input size
}
```

### Loading off the main thread

The first load of a model downloads tens to hundreds of MB. `loadModelAsync` runs
it on a background isolate; because the native model is a single process-global, it
becomes visible to every isolate once loaded.

```dart
await Model2Vec.loadModelAsync('minishlab/potion-base-2M');
```

To show a progress bar for that download, use `loadModelWithProgress` — it streams
`LoadProgress` snapshots and always ends on `LoadPhase.done` (a cached model or
local path jumps straight there; a failed load surfaces as a stream error):

```dart
await for (final p in Model2Vec.loadModelWithProgress('minishlab/potion-base-8M')) {
  switch (p.phase) {
    case LoadPhase.downloading:
      print('${((p.fraction ?? 0) * 100).round()}%'); // fraction is null until size is known
    case LoadPhase.resolving || LoadPhase.parsing:
      print('Preparing…');
    case LoadPhase.done:
      print('Ready');
  }
}
```

### Similarity & vector math

`Model2VecUtils` is a set of static helpers tuned for embeddings.

```dart
final query = Model2Vec.generateEmbedding('cat');
final docs = [
  Model2Vec.generateEmbedding('kitten'),
  Model2Vec.generateEmbedding('rocket'),
];

// Cosine similarity of two vectors (-1.0 … 1.0)
Model2VecUtils.cosineSimilarity(query, docs[0]);

// Rank an in-memory list: (index, score) pairs, top-K with an optional threshold
Model2VecUtils.similaritySearchWithScores(query, docs, topK: 5, threshold: 0.5);

// Compress Float32 → Int8 (¼ the memory) and serialize for storage
final int8 = Model2VecUtils.quantizeToInt8(query);
final str = Model2VecUtils.toBase64(query);
```

### Local retrieval (RAG)

Build a searchable, persistable index entirely on-device — chunk, embed, store,
query. No server.

```dart
// Split documents into overlapping passages, then embed and index them. Storing
// each passage as the entry's payload means a hit carries its text directly.
final index = EmbeddingIndex(quantized: true); // int8 storage, ~¼ the memory
for (final passage in chunkText(document, maxChars: 800, overlap: 100)) {
  index.add(passage, Model2Vec.generateEmbedding(passage), payload: passage);
}

// Query — SearchResult(id, score, payload), most similar first.
final query = Model2Vec.generateEmbedding('How do I reset my password?');
for (final hit in index.search(query, topK: 5)) {
  print('${hit.score.toStringAsFixed(3)}  ${hit.payload}');
}

// Persist and reload — no model needed to load, only to embed new queries.
final reloaded = EmbeddingIndex.fromBytes(index.toBytes());
```

Use `Model2VecUtils.maximalMarginalRelevance` to rerank for diverse results.

### Parallel embedding & lifecycle

```dart
// Fan batches across CPU cores with a pool of worker isolates.
final pool = await EmbeddingPool.start(); // defaults to the core count
final results = await pool.embedBatches(listOfBatches);
await pool.close();

// State & teardown.
if (Model2Vec.isInitialized) {
  final info = Model2Vec.modelInfo; // dimension, vocabulary, normalized, median
  print('dimension ${info.dimension}');
}
Model2Vec.unloadModel(); // frees the native model
```

## API reference

Full docs are generated on [pub.dev](https://pub.dev/documentation/model2vec/latest/).
The essentials:

**`Model2Vec`** — `loadModel` · `loadModelAdvanced` · `loadModelFromBytes` ·
`loadModelAsync` · `loadModelAdvancedAsync` · `loadModelWithProgress` ·
`generateEmbedding` · `generateBatchEmbeddings` · `generateEmbeddingAsync` ·
`generateBatchEmbeddingsAsync` · `generateEmbeddingStream` · `tokenize` ·
`isInitialized` · `modelInfo` · `unloadModel` · `recommendedModels` ·
`embeddingDimension` · `vocabularySize` · `isNormalized` · `medianTokenLength`.

**`Model2VecUtils`** — `cosineSimilarity` · `cosineDistance` · `euclideanDistance` ·
`dotProduct` · `similaritySearchWithScores` · `maximalMarginalRelevance` ·
`normalize` · `meanPooling` · `quantizeToInt8` · `dequantizeInt8` ·
`pairwiseSimilarity` · `toBase64` · `fromBase64`.

**Types** — `EmbeddingIndex` (on-device vector store), `EmbeddingPool` (parallel
embedding), `chunkText` (overlapping chunker), `ModelInfo`, `LoadProgress` /
`LoadPhase`, `RecommendedModel`, `Model2VecException` / `Model2VecErrorKind`.

## Performance

Single-vector math runs natively in Dart with zero FFI overhead; batch generation
uses the Rust engine's SIMD. Measured on an Apple Silicon laptop from an AOT
`dart build` bundle, best of 3 runs (absolute numbers vary by machine):

| Model | Load (cached) | Single embedding | Batch of 32 |
| ----- | ------------- | ---------------- | ----------- |
| `potion-base-2M` | 21 ms | 240 μs | 3.8 ms |
| `potion-base-4M` | 22 ms | 248 μs | 3.8 ms |
| `potion-base-8M` | 24 ms | 251 μs | 4.0 ms |
| `potion-base-32M` | 66 ms | 254 μs | 4.3 ms |
| `potion-multilingual-128M` | 841 ms | 312 μs | 4.1 ms |

A single embedding is a few hundred microseconds; `similaritySearchWithScores`
over 100,000 vectors takes **< 100 ms** in pure Dart.

## Deployment

The native library ships with your app automatically — nothing to link by hand:

- **Flutter** (`flutter build …`) and **`dart run`** bundle and resolve it for you.
- **Standalone CLI**: build with `dart build cli` (not `dart compile exe`, which
  does not bundle native assets yet). The library is copied into `bundle/lib/`
  next to the executable, so the bundle is self-contained.

## Development

The Rust library builds automatically through Native Assets (`cargo build` runs when
you run Dart code). Regenerate the FFI bindings after changing the C API in
[`native/src/lib.rs`](native/src/lib.rs):

```bash
dart run ffigen
```

Testing — model-dependent tests are tagged `integration` (they download a small
model on first run and cache it):

```bash
dart test                 # everything
dart test -x integration  # fast lane: unit tests only, no model download
dart test -t integration  # only the model-dependent tests
```

Before pushing, mirror the CI checks
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
dart format .
dart analyze --fatal-infos
dart test
```

## License

MIT — see [LICENSE](LICENSE).
