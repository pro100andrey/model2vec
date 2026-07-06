# model2vec

[![pub package](https://img.shields.io/pub/v/model2vec.svg)](https://pub.dev/packages/model2vec)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**High-performance, local text embeddings for Dart and Flutter.** A Dart wrapper around [model2vec-rs](https://github.com/MinishLab/model2vec-rs) using Rust FFI and Native Assets. Model2Vec creates small, fast, and effective text embeddings by distilling knowledge from large language models into a simple vocabulary-based look-up table.

## Table of Contents

- [model2vec](#model2vec)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Recommended Models](#recommended-models)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Recipes \& Patterns](#recipes--patterns)
    - [1. Advanced Batch Processing](#1-advanced-batch-processing)
    - [2. Massive Data Streaming](#2-massive-data-streaming)
    - [3. Asynchronous Isolate Execution](#3-asynchronous-isolate-execution)
    - [4. Vector Math \& Quantization](#4-vector-math--quantization)
  - [API Reference](#api-reference)
    - [Core Methods (`Model2Vec` class)](#core-methods-model2vec-class)
    - [Math Utilities (`Model2VecUtils` class)](#math-utilities-model2vecutils-class)
  - [Performance](#performance)
  - [Development \& Contributing](#development--contributing)
  - [License](#license)

## Key Features

- **Extreme Performance:** Built on top of a highly optimized Rust engine. Up to **~1.7x faster** than the official Python implementation, generating embeddings in microseconds.
- **Compact & Quantized:** Models are typically 25MB - 100MB. Perfect for edge computing.
- **Massive Streaming:** Built-in `generateEmbeddingStream` for processing millions of rows without blocking the Event Loop or overflowing RAM.
- **Hugging Face Integration:** Automatically downloads and caches models directly from the Hugging Face Hub.
- **Zero-Stutter Async:** Transparently runs heavy tokenization and math in background Dart Isolates using `Async` methods.
- **Vector Utilities:** Ships with high-performance mathematical tools (`cosineSimilarity`, `quantizeToInt8`, `similaritySearch`, etc.).

## Recommended Models

Model2Vec provides a variety of pre-trained models optimized for different use cases. These can be loaded directly via their Hugging Face model ID.

| Model ID | Language | Distilled From | Params | Dimension | Size |
| -------- | -------- | -------------- | ------ | --------- | ---- |
| [`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) | English | bge-base-en-v1.5 | 32.3M | 512 | ~150MB |
| [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) | Multi | bge-m3 | 128M | 768 | ~500MB |
| [`minishlab/potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M) | English | bge-base-en-v1.5 | 32.3M | 512 | ~150MB |
| [`minishlab/potion-code-16M`](https://huggingface.co/minishlab/potion-code-16M) | Code | CodeRankEmbed | 16M | 384 | ~80MB |
| [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | English | bge-base-en-v1.5 | 7.5M | 256 | ~50MB |
| [`minishlab/potion-base-4M`](https://huggingface.co/minishlab/potion-base-4M) | English | bge-base-en-v1.5 | 3.7M | 128 | ~30MB |
| [`minishlab/potion-base-2M`](https://huggingface.co/minishlab/potion-base-2M) | English | bge-base-en-v1.5 | 1.8M | 64 | ~25MB |

## Installation

Add `model2vec` to your `pubspec.yaml`:

```yaml
dependencies:
  model2vec: any
```

Or add it using the command line:

```bash
dart pub add model2vec
```

*Requires **Dart SDK**: 3.10.0+ and **Rust toolchain**: 1.86.0+ (to build the native library via Native Assets).*

## Quick Start

```dart
import 'package:model2vec/model2vec.dart';

void main() {
  // Model2Vec is a stateless namespace of static methods — there is one
  // active model per process, loaded automatically via Native Assets.

  // Initialize with a model from Hugging Face
  Model2Vec.initEmbedder('minishlab/potion-base-2M');

  // Generate an embedding
  final embedding = Model2Vec.generateEmbedding('Dart FFI is blazingly fast 🚀');

  print('Vector dimension: ${Model2Vec.embeddingDimension}');
  print('Vocabulary size: ${Model2Vec.vocabularySize}');
}
```

> **Migrating from 1.x?** The instance API is gone. Replace
> `Model2Vec.instance.foo(...)` with `Model2Vec.foo(...)`, drop any
> `Model2Vec.boot(...)` / `Model2Vec(lib)` calls (library resolution is now
> automatic), and read `Model2Vec.recommendedModels` instead of calling
> `getRecommendedModels()`. See the [CHANGELOG](CHANGELOG.md) for the full list.

## Recipes & Patterns

### 1. Advanced Batch Processing

Process multiple strings at once for maximum hardware utilization. You can control sequence truncation and batch sizes.

```dart
final texts = ['Dart', 'Rust', 'Flutter'];

final embeddings = Model2Vec.generateBatchEmbeddings(
  texts,
  maxLength: 256,   // Truncate strings longer than 256 tokens
  batchSize: 1024,  // Internal chunks sent to the FFI layer
);
```

### 2. Massive Data Streaming

When reading gigabytes of text from files or databases, loading everything into memory will crash the app. Use the **Streaming API** to handle data in chunks automatically.

```dart
import 'dart:convert';
import 'dart:io';

Future<void> processHugeFile() async {
  final fileStream = File('massive_dataset.txt')
      .openRead()
      .transform(utf8.decoder)
      .transform(const LineSplitter());

  // Converts a Stream<String> into a Stream<Float32List>
  final embeddingStream = Model2Vec.generateEmbeddingStream(
    fileStream,
    batchSize: 500, // Process 500 strings at a time
    useIsolate: true, // Run math in background threads
  );

  await for (final embedding in embeddingStream) {
    saveToDb(embedding); // Memory safe!
  }
}
```

### 3. Asynchronous Isolate Execution

Never block the main thread. If you are building a Flutter app, always use the `Async` variants to perform generation in a background `Isolate`.

```dart
final embedding = await Model2Vec.generateEmbeddingAsync('A very long text...');
final batch = await Model2Vec.generateBatchEmbeddingsAsync(['A', 'B', 'C']);
```

### 4. Vector Math & Quantization

The library ships with `Model2VecUtils` — a powerful suite of math operations tuned for embeddings.

```dart
final query = Model2Vec.generateEmbedding('cat');
final candidates = [
  Model2Vec.generateEmbedding('dog'),
  Model2Vec.generateEmbedding('space'),
];

// 1. Semantic Similarity (Cosine)
final sim = Model2VecUtils.cosineSimilarity(query, candidates[0]);

// 2. Threshold Searching (Find all matches > 80%)
final matches = Model2VecUtils.similaritySearchWithThreshold(
  query, candidates, threshold: 0.8,
);

// 3. Scalar Quantization (Compress Float32 to Int8 to save 4x RAM)
final compressed = Model2VecUtils.quantizeToInt8(query);

// 4. Mean Pooling (Average multiple vectors into one)
final sentenceVector = Model2VecUtils.meanPooling(candidates);

// 5. DB Serialization
final base64String = Model2VecUtils.toBase64(query);
```

### 5. Local Retrieval (RAG) with `EmbeddingIndex`

Build a searchable, persistable index of your documents entirely on-device — chunk, embed, store, and query without a server.

```dart
// 1. Split long documents into overlapping passages
final passages = chunkText(document, maxChars: 800, overlap: 100);

// 2. Embed and index them (int8 storage cuts memory ~4x)
final index = EmbeddingIndex(quantized: true);
for (var i = 0; i < passages.length; i++) {
  index.add('passage-$i', Model2Vec.generateEmbedding(passages[i]));
}

// 3. Query — returns SearchResult(id, score), most similar first
final query = Model2Vec.generateEmbedding('How do I reset my password?');
final hits = index.search(query, topK: 5);
for (final hit in hits) {
  print('${hit.id}: ${hit.score.toStringAsFixed(3)}');
}

// 4. Persist to disk and reload later
final bytes = index.toBytes();
final reloaded = EmbeddingIndex.fromBytes(bytes);

// (Optional) Rerank a candidate vector list for diverse results (MMR):
final diverse = Model2VecUtils.maximalMarginalRelevance(
  query, candidateVectors, topK: 5, lambda: 0.5,
);
```

### 6. Parallel Embedding & Lifecycle

```dart
// Embed across all CPU cores with a pool of worker isolates
final pool = await EmbeddingPool.start();          // defaults to core count
final results = await pool.embedBatches(listOfBatches);
await pool.close();

// Non-throwing state check + free native memory when done
if (Model2Vec.isInitialized) {
  final info = Model2Vec.modelInfo;                // dim, vocab, normalized, median
  print('Loaded model with dimension ${info.dimension}');
}
Model2Vec.unloadModel();                           // releases the native model
```

## API Reference

### Core Methods (`Model2Vec` class)

| Method / Property | Description |
| ----------------- | ----------- |
| `initEmbedder(path)` | Initializes the model from a Hugging Face repo ID or local path. |
| `initEmbedderAdvanced(...)` | Advanced initialization with custom `cacheDirectory`, `hfToken`, or `normalize` overrides. |
| `initEmbedderFromBytes(...)` | Initializes the model directly from raw `Uint8List` bytes (`model.safetensors`, `tokenizer.json`, etc). |
| `recommendedModels` | A typed `List<RecommendedModel>` catalog of officially recommended Potion models (offline). |
| `tokenize(text)` | Runs the internal BPE tokenizer and returns a `List<String>`. |
| `generateEmbedding(text)` | Synchronously generates a `Float32List` embedding vector. |
| `generateBatchEmbeddings(texts)` | Synchronously generates embeddings for a `List<String>` using Rust SIMD. |
| `generateEmbeddingAsync(text)` | Asynchronously generates an embedding in a background `Isolate`. |
| `generateEmbeddingStream(stream)` | Processes a huge `Stream<String>` into a `Stream<Float32List>` in batches. |
| `isInitialized` | Non-throwing check for whether a model is currently loaded. |
| `modelInfo` | All model metadata in one `ModelInfo` (dimension, vocabulary, normalized, median). |
| `unloadModel()` | Unloads the active model and frees its native memory. |
| `embeddingDimension` | Property returning the vector size (e.g., 256, 384, 512). |
| `vocabularySize` | Property returning the number of tokens in the model's vocabulary. |

### Math Utilities (`Model2VecUtils` class)

| Method | Description |
| ------ | ----------- |
| `cosineSimilarity(a, b)` | Calculates cosine similarity (-1.0 to 1.0) between two vectors. |
| `cosineDistance(a, b)` | Calculates cosine distance (0.0 to 2.0). |
| `euclideanDistance(a, b)` | Calculates Euclidean (L2) distance. |
| `similaritySearch(query, docs)` | Returns the indices of the Top-K most similar vectors in a database. |
| `similaritySearchWithThreshold` | Returns all indices with similarity above a given threshold. |
| `quantizeToInt8(vector)` | Compresses a `Float32List` into an `Int8List` (4x memory savings). |
| `normalize(vector)` | Applies L2 normalization to a vector. |
| `meanPooling(vectors)` | Averages multiple vectors into a single vector. |
| `toBase64` / `fromBase64` | Serializes/Deserializes a vector to/from a Base64 string for DB storage. |

## Performance

`model2vec` uses highly optimized FFI bindings. For mathematical operations on embeddings, Dart handles single-vector math natively with zero-overhead, while batch generation leverages Rust's SIMD (auto-vectorization) capabilities.

Here is a performance benchmark run on a typical machine (AOT compiled):

| Model | Load Time (Cache) | Single Embedding | Batch (32) |
| --- | --- | --- | --- |
| `minishlab/potion-base-2M` | ~40 ms | 372.9 μs | 3.85 ms |
| `minishlab/potion-base-4M` | ~40 ms | 363.7 μs | 4.19 ms |
| `minishlab/potion-base-8M` | ~40 ms | 382.1 μs | 5.60 ms |
| `minishlab/potion-base-32M` | ~120 ms | 452.6 μs | 6.79 ms |
| `minishlab/potion-multilingual-128M` | ~1050 ms | 416.1 μs | 5.38 ms |

> *Note: Initial load times may vary slightly based on the disk speed. Generating an embedding takes just **a few microseconds** per string.*

- `similaritySearch` over 100,000 vectors takes **<100ms** in pure Dart.

## Development & Contributing

The library uses Dart Native Assets, meaning `cargo build` is invoked automatically when running Dart code.

To manually re-build bindings if you modify the Rust C-API (`native/src/lib.rs`):

```bash
dart run ffigen
```

To run the test suite:

```bash
dart test
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
