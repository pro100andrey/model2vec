# model2vec

A Dart wrapper for [model2vec-rs](https://github.com/MinishLab/model2vec-rs) using Rust FFI and Dart Native Assets.

Model2Vec is a technique to create small, fast, and effective text embeddings by distilling knowledge from large language models into a simple vocabulary-based look-up table.

## Features

- **Blazing Fast**: Generate embeddings in microseconds.
- **Compact**: Models are typically 25MB - 100MB.
- **Versatile**: Supports loading from Hugging Face, local paths, or memory.
- **Asynchronous API**: Built-in support for running heavy tasks in background Isolates.
- **Native Assets**: Automatic building and bundling of the Rust library.

## Requirements

- **Rust toolchain**: 1.86.0+ (to build the native library)
- **Dart SDK**: 3.12.0+

## Installation

Add `model2vec` to your `pubspec.yaml`:

```yaml
dependencies:
  model2vec: ^0.1.0
```

## Usage

### Basic Usage

The simplest way is to use the singleton `Model2Vec.instance`.

```dart
import 'package:model2vec/model2vec.dart';

void main() {
  final m2v = Model2Vec.instance;
  
  // Initialize with a model from Hugging Face
  m2v.initEmbedder('minishlab/potion-base-2M');
  
  // Generate an embedding
  final embedding = m2v.generateEmbedding('Hello world');
  print('Vector dimension: ${embedding.length}');
  print('First 5 elements: ${embedding.sublist(0, 5)}');
}
```

### Batch Processing

For multiple strings, use the batch methods for better performance.

```dart
final texts = ['Hello world', 'Model2Vec is fast', 'Dart + Rust = ❤️'];
final embeddings = m2v.generateBatchEmbeddings(texts);
```

### Asynchronous Execution

To avoid blocking the main UI thread (useful in Flutter), use the `Async` variants:

```dart
final embedding = await m2v.generateEmbeddingAsync('Some long text...');
```

### Vector Utilities

The package includes high-performance utilities for vector operations.

```dart
import 'package:model2vec/model2vec.dart';

void main() {
  final query = m2v.generateEmbedding('cat');
  final candidates = [
    m2v.generateEmbedding('dog'),
    m2v.generateEmbedding('space'),
    m2v.generateEmbedding('kitten'),
  ];
  
  // Cosine Similarity
  final similarity = Model2VecUtils.cosineSimilarity(query, candidates[0]);
  
  // Similarity Search (Top-K)
  final topIndices = Model2VecUtils.similaritySearch(query, candidates, topK: 2);
  print('Top match: ${candidates[topIndices[0]]}');
}
```

## Development

### Building the Native Library

The library uses Dart Native Assets, so it builds automatically during `dart run` or `flutter run`. To build it manually:

```bash
cd native
cargo build --release
```

### Regenerating FFI Bindings

If you modify the Rust C-API (`native/src/lib.rs`), you need to regenerate the Dart bindings:

```bash
dart run ffigen
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
