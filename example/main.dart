import 'dart:io';

import 'package:model2vec/model2vec.dart';

/// Quickstart: load a model, embed a few sentences, and compare them by cosine
/// similarity. Run with `dart run example/main.dart`.
///
/// See `scaling_example.dart` for batch/parallel/streaming embedding, and
/// `rag_example.dart` for local retrieval (RAG).
Future<void> main() async {
  // Loads the model on a background isolate, downloading it on first run.
  await Model2Vec.loadModelAsync('minishlab/potion-base-2M');

  final kitten = Model2Vec.generateEmbedding('A cute little kitten');
  final puppy = Model2Vec.generateEmbedding('A small puppy');
  final rocket = Model2Vec.generateEmbedding('Interstellar space travel');

  // Higher cosine similarity means closer in meaning.
  final puppyPct = (Model2VecUtils.cosineSimilarity(kitten, puppy) * 100)
      .toStringAsFixed(1);
  final rocketPct = (Model2VecUtils.cosineSimilarity(kitten, rocket) * 100)
      .toStringAsFixed(1);

  stdout
    ..writeln('Embedding dimension: ${kitten.length}')
    ..writeln('kitten vs puppy:  $puppyPct%')
    ..writeln('kitten vs rocket: $rocketPct%');
}
