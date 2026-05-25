# model2vec

A Dart wrapper for [model2vec-rs](https://github.com/MinishLab/model2vec-rs) using Rust FFI and Dart Native Assets.

## Features

- Fast text embeddings using model2vec.
- Supports loading models from Hugging Face Hub or local path.
- Supports initializing from memory (bytes).
- Automatic native library building via `dart run` (Native Assets).

## Requirements

- Rust toolchain (1.86.0+)
- Dart SDK 3.12.0+

## Usage

```dart
import 'dart:ffi';
import 'package:model2vec/model2vec.dart';

void main() {
  // Load the library (in a real app, native assets handles this)
  final lib = DynamicLibrary.open('libm2v_ffi.so');
  final m2v = Model2Vec(lib);
  
  // Initialize
  m2v.initEmbedder('minishlab/potion-base-2M');
  
  // Embed
  final embedding = m2v.generateEmbedding('Hello world');
  print(embedding);
}
```

## Development

To rebuild the native library manually:
```bash
cd src
cargo build --release
```

To regenerate bindings:
```bash
dart run ffigen
```
