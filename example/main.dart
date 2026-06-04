import 'dart:io';

import 'package:model2vec/model2vec.dart';

/// Example demonstrating production-ready usage of the Model2Vec package.
Future<void> main() async {
  try {
    // 1. Initialize the API via shared instance
    final m2v = Model2Vec.instance;
    stdout.writeln('✅ Model2Vec library loaded successfully.');

    // 3. Explore Recommended Models
    final models = m2v.getRecommendedModels();
    stdout.writeln('\n📦 Available models:');
    for (final m in models) {
      final name = m['name']! as String;

      stdout.writeln('  - ${name.padRight(25)} ID: ${m['id']}');
    }

    // 4. Initialize a Model
    const modelId = 'minishlab/potion-base-2M';
    stdout.writeln('\n🚀 Initializing $modelId...');
    final sw = Stopwatch()..start();
    m2v.initEmbedder(modelId);
    stdout
      ..writeln('✨ Initialized in ${sw.elapsedMilliseconds}ms')
      // 5. Inspect Model Metadata
      ..writeln('\n📊 Model Metadata:')
      ..writeln('  - Dimension:           ${m2v.embeddingDimension}')
      ..writeln('  - Vocabulary Size:     ${m2v.vocabularySize}')
      ..writeln('  - Is Normalized:       ${m2v.isNormalized}')
      ..writeln('  - Median Token Length: ${m2v.medianTokenLength}');

    // 6. Tokenization Demo
    const text = 'Model2Vec is incredibly fast!';
    final tokens = m2v.tokenize(text);
    stdout
      ..writeln('\n🔍 Tokenization: "$text"')
      ..writeln('  - Tokens: $tokens')
      // 7. Single Embedding
      ..writeln('\n🧠 Generating single embedding...');
    final embedding = m2v.generateEmbedding(text);
    stdout
      ..writeln('  - Vector (first 3): ${embedding.take(3).toList()}')
      ..writeln('  - Total Length:    ${embedding.length}');

    // 8. Batch Embedding (Production Optimization)
    final texts = [
      'The first sentence.',
      'A second, slightly longer sentence for the batch.',
      'Third one.',
    ];
    stdout.writeln(
      '\n⚡ Generating batch embeddings for ${texts.length} sentences...',
    );
    final batchStartTime = DateTime.now();
    final batch = m2v.generateBatchEmbeddings(texts);
    final batchDuration = DateTime.now().difference(batchStartTime);

    stdout.writeln('  - Processed in ${batchDuration.inMicroseconds}μs');
    for (var i = 0; i < batch.length; i++) {
      stdout.writeln('  - Result $i length: ${batch[i].length}');
    }

    // 9. Vector Math & Semantic Search
    stdout.writeln('\n🧠 Vector Math & Semantic Search:');
    final query = m2v.generateEmbedding('A cute little kitten');
    final db = [
      m2v.generateEmbedding('A small cat'),
      m2v.generateEmbedding('A big dog'),
      m2v.generateEmbedding('Space exploration'),
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
      // 10. Streaming API for Huge Datasets
      ..writeln('\n🌊 Streaming API (1000 items):');
    final stream = Stream.fromIterable(List.generate(1000, (i) => 'Item $i'));
    final resultStream = m2v.generateEmbeddingStream(
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
