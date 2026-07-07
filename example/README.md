# Model2Vec examples

Runnable examples for the `model2vec` package. Each is a standalone script that
imports the package with `package:model2vec/model2vec.dart`, just like your own
code would.

Each example tells one story. Start with `main.dart`, then reach for the other
two when you need retrieval or scale.

## Running

```sh
dart run example/main.dart            # quickstart
dart run example/scaling_example.dart # batch, parallel, streaming
dart run example/rag_example.dart     # local retrieval (RAG)
```

The native Rust library is compiled automatically on first run via the package's
build hook, so there is nothing to set up. The first run also downloads the
embedding model from Hugging Face (a few MB), so it needs network access; later
runs use the local cache.

## What's here

- **[`main.dart`](main.dart)** — quickstart. Load a model, embed a few
  sentences, and compare them with cosine similarity. The shortest path from
  zero to a result.

- **[`scaling_example.dart`](scaling_example.dart)** — production, at scale:
  loading with a live download progress bar (`loadModelWithProgress`), batch
  embedding, parallel embedding across CPU cores with an `EmbeddingPool`, and
  the streaming API for datasets too large to hold in memory.

- **[`rag_example.dart`](rag_example.dart)** — a local retrieval (RAG) pipeline:
  `chunkText` splits documents, `EmbeddingIndex` stores each passage with its
  text as payload, then it answers questions with nearest-neighbour search,
  threshold filtering, and MMR reranking for diverse results — and persists the
  index to disk with `toBytes` / `fromBytes`.
