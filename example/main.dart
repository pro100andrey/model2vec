import 'dart:io';

import 'package:model2vec/model2vec.dart';

/// Example demonstrating production-ready usage of the Model2Vec package.
Future<void> main() async {
  try {
    stdout.writeln('=== Model2Vec Example ===');

    // 1. Explore Recommended Models
    const models = Model2Vec.recommendedModels;
    stdout.writeln('\n📦 Available models:');
    for (final m in models) {
      stdout.writeln('  - ${m.name.padRight(25)} ID: ${m.id}');
    }

    // 2. Initialize a Model
    const modelId = 'minishlab/potion-base-2M';
    stdout.writeln('\n🚀 Initializing $modelId...');
    final sw = Stopwatch()..start();
    Model2Vec.initEmbedder(modelId);
    stdout
      ..writeln('✨ Initialized in ${sw.elapsedMilliseconds}ms')
      // 3. Inspect Model Metadata
      ..writeln('\n📊 Model Metadata:')
      ..writeln('  - Dimension:           ${Model2Vec.embeddingDimension}')
      ..writeln('  - Vocabulary Size:     ${Model2Vec.vocabularySize}')
      ..writeln('  - Is Normalized:       ${Model2Vec.isNormalized}')
      ..writeln('  - Median Token Length: ${Model2Vec.medianTokenLength}');

    // 4. Tokenization Demo
    const text = 'Model2Vec is incredibly fast!';
    final tokens = Model2Vec.tokenize(text);
    stdout
      ..writeln('\n🔍 Tokenization: "$text"')
      ..writeln('  - Tokens: $tokens')
      // 5. Single Embedding
      ..writeln('\n🧠 Generating single embedding...');
    final embedding = Model2Vec.generateEmbedding(text);
    stdout
      ..writeln('  - Vector (first 3): ${embedding.take(3).toList()}')
      ..writeln('  - Total Length:    ${embedding.length}');

    // 6. Batch Embedding (Production Optimization)
    final texts = [
      'The first sentence.',
      'A second, slightly longer sentence for the batch.',
      'Third one.',
    ];
    stdout.writeln(
      '\n⚡ Generating batch embeddings for ${texts.length} sentences...',
    );
    final batchStartTime = DateTime.now();
    final batch = Model2Vec.generateBatchEmbeddings(texts);
    final batchDuration = DateTime.now().difference(batchStartTime);

    stdout.writeln('  - Processed in ${batchDuration.inMicroseconds}μs');
    for (var i = 0; i < batch.length; i++) {
      stdout.writeln('  - Result $i length: ${batch[i].length}');
    }

    // 7. Vector Math & Semantic Search
    stdout.writeln('\n🧠 Vector Math & Semantic Search:');
    final query = Model2Vec.generateEmbedding('A cute little kitten');
    final db = [
      Model2Vec.generateEmbedding('A small cat'),
      Model2Vec.generateEmbedding('A big dog'),
      Model2Vec.generateEmbedding('Space exploration'),
    ];

    final simCat = Model2VecUtils.cosineSimilarity(query, db[0]);
    final simSpace = Model2VecUtils.cosineSimilarity(query, db[2]);
    stdout
      ..writeln(
        '  - Sim(kitten, cat):   ${(simCat * 100).toStringAsFixed(1)}%',
      )
      ..writeln(
        '  - Sim(kitten, space): ${(simSpace * 100).toStringAsFixed(1)}%',
      );

    final topMatch = Model2VecUtils.similaritySearch(query, db, topK: 1);
    stdout
      ..writeln('  - Best match index:   ${topMatch.first}')
      // 8. Streaming API for Huge Datasets
      ..writeln('\n🌊 Streaming API (1000 items):');
    final stream = Stream.fromIterable(List.generate(1000, (i) => 'Item $i'));
    final resultStream = Model2Vec.generateEmbeddingStream(
      stream,
      batchSize: 200,
    );

    var count = 0;
    await for (final _ in resultStream) {
      count++;
    }
    stdout
      ..writeln(
        '  - Successfully streamed and processed $count embeddings.',
      )
      ..writeln('\n🎉 All operations completed successfully.');
  } on Model2VecException catch (e) {
    stdout.writeln('\n❌ Model2Vec Error: ${e.message}');
    if (e.code != null) {
      stdout.writeln('   Error Code: ${e.code}');
    }
  } on Object catch (e) {
    stdout.writeln('\n💥 Unexpected Error: $e');
  }
}
