import 'dart:io';

import 'package:model2vec/model2vec.dart';

/// A tiny knowledge base: article id -> body text.
const _knowledgeBase = <String, String>{
  'password':
      'To reset your password, open Settings, tap Security, and choose Reset '
      'Password. We email you a secure link that expires in one hour. If the '
      'email does not arrive, check your spam folder and confirm the address '
      'on your account is correct.',
  'twoFactor':
      'Two-factor authentication adds a second step at sign-in. After your '
      'password you enter a short code from an authenticator app or an SMS '
      'message, so a stolen password alone is not enough to log in.',
  'refunds':
      'Our refund policy allows returns within 30 days of purchase for a full '
      'refund. Items must be unused and in the original packaging. Refunds are '
      'issued to the original payment method within five business days.',
  'shipping':
      'Shipping is free on orders over fifty dollars. Standard delivery '
      'usually arrives in three to five business days; express delivery '
      'arrives the next day for an extra fee.',
  'billing':
      'To cancel your subscription, open Billing and choose Cancel Plan before '
      'the renewal date. You keep access until the end of the current billing '
      'period and are not charged again.',
};

Future<void> main() async {
  try {
    stdout.writeln('=== Model2Vec RAG example ===\n');

    // 1. Load an embedding model. For real retrieval, potion-retrieval-32M is
    //    purpose-built; potion-base-8M is a lighter, still-capable choice.
    Model2Vec.initEmbedder('minishlab/potion-base-8M');

    // 2. Chunk each article and index the passages. The index stores id ->
    //    vector; we keep our own id -> text map to show the passage behind a
    //    hit (the index deliberately does not store the text).
    final index = EmbeddingIndex();
    final passages = <String, String>{};

    for (final article in _knowledgeBase.entries) {
      final chunks = chunkText(article.value, maxChars: 160, overlap: 32);
      for (var i = 0; i < chunks.length; i++) {
        final id = '${article.key}#$i';
        passages[id] = chunks[i];
        index.add(id, Model2Vec.generateEmbedding(chunks[i]));
      }
    }
    stdout.writeln(
      'Indexed ${index.length} passages from '
      '${_knowledgeBase.length} articles.\n',
    );

    // 3. Answer questions by retrieving the closest passages.
    const questions = [
      'How do I change my login password?',
      'Can I get my money back?',
      'When will my package arrive?',
    ];

    for (final question in questions) {
      final query = Model2Vec.generateEmbedding(question);
      stdout.writeln('Q: $question');
      for (final hit in index.search(query, topK: 2)) {
        final score = hit.score.toStringAsFixed(3);
        stdout.writeln('  [$score] ${hit.id}: ${passages[hit.id]}');
      }
      stdout.writeln();
    }

    // 4. Threshold search — keep only confident matches.
    final twoFa = Model2Vec.generateEmbedding('two step verification code');
    final strong = index.searchWithThreshold(twoFa, threshold: 0.4);
    stdout.writeln(
      'Confident matches for "two step verification code": '
      '${strong.map((h) => h.id).toList()}\n',
    );

    // 5. Diverse reranking (MMR): pull a wider candidate set, then rerank to
    //    avoid near-duplicate passages in the top results.
    final broad = index.search(twoFa, topK: 6);
    final candidateVectors = [
      for (final hit in broad) Model2Vec.generateEmbedding(passages[hit.id]!),
    ];
    final order = Model2VecUtils.maximalMarginalRelevance(
      twoFa,
      candidateVectors,
      topK: 3,
      lambda: 0.6,
    );
    final reranked = [for (final i in order) broad[i].id];
    stdout.writeln('MMR-reranked top 3: $reranked\n');

    // 6. Persist the index to disk and reload it. Loading needs no model —
    //    only generating new query vectors does.
    final file = File('${Directory.systemTemp.path}/m2v_rag_index.bin');
    await file.writeAsBytes(index.toBytes());
    final reloaded = EmbeddingIndex.fromBytes(await file.readAsBytes());
    stdout.writeln('Saved and reloaded index: ${reloaded.length} passages.');
    await file.delete();

    stdout.writeln('\n🎉 Done.');
  } on Model2VecException catch (e) {
    stdout.writeln('❌ Model2Vec error (${e.kind.name}): ${e.message}');
  }
}
