# model2vec

A Dart wrapper for [model2vec-rs](https://github.com/MinishLab/model2vec-rs) using Rust FFI and Dart Native Assets.

Model2Vec is a technique to create small, fast, and effective text embeddings by distilling knowledge from large language models into a simple vocabulary-based look-up table.

## Features

- **Blazing Fast**: The underlying Rust engine is **~1.7x faster** than the official Python implementation, generating embeddings in microseconds.
- **Compact & Quantized**: Models are typically 25MB - 100MB. Supports `f32`, `f16`, and `i8` quantized models for minimal RAM footprint on mobile devices.
- **Hugging Face Hub Integration**: Automatically downloads and caches models directly from Hugging Face via `initEmbedder`.
- **Asynchronous API**: Built-in support for running heavy tasks in background Isolates, preventing UI freezes in Flutter.
- **Native Assets**: Automatic building and bundling of the Rust library using Dart Native Assets.
- **Efficient Tokenization**: Uses a highly optimized BPE tokenizer under the hood.

## Recommended Models

Model2Vec provides a variety of pre-trained models optimized for different use cases. These can be loaded by their Hugging Face model ID.

| Model ID | Language | Distilled From | Params | Dimension | Size | Task |
|----------|----------|----------------|--------|-----------|------|------|
| [`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) | English | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 32.3M | 512 | ~150MB | General |
| [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) | Multi | [bge-m3](https://huggingface.co/BAAI/bge-m3) | 128M | 768 | ~500MB | General |
| [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | English | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 7.5M | 384 | ~50MB | General |
| [`minishlab/potion-base-4M`](https://huggingface.co/minishlab/potion-base-4M) | English | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 3.7M | 256 | ~30MB | General |
| [`minishlab/potion-base-2M`](https://huggingface.co/minishlab/potion-base-2M) | English | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 1.8M | 256 | ~25MB | General |
| [`minishlab/potion-code-16M`](https://huggingface.co/minishlab/potion-code-16M) | Code | [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) | 16M | 384 | ~80MB | Code |

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
