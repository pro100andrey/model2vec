import 'dart:typed_data';

import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

Float32List _v(List<double> xs) => Float32List.fromList(xs);

void main() {
  group('EmbeddingIndex', () {
    test('add / length / contains / remove / clear', () {
      final index = EmbeddingIndex()
        ..add('a', _v([1, 0, 0]))
        ..add('b', _v([0, 1, 0]));

      expect(index.length, 2);
      expect(index.contains('a'), isTrue);
      expect(index.contains('z'), isFalse);
      expect(index.dimension, 3);

      expect(index.remove('a'), isTrue);
      expect(index.remove('a'), isFalse);
      expect(index.length, 1);

      index.clear();
      expect(index.isEmpty, isTrue);
      expect(index.dimension, isNull);
    });

    test('search ranks by cosine similarity, most similar first', () {
      final index = EmbeddingIndex()
        ..add('a', _v([1, 0, 0]))
        ..add('b', _v([0, 1, 0]))
        ..add('near-a', _v([0.9, 0.1, 0]));

      final hits = index.search(_v([1, 0, 0]), topK: 2);
      expect(hits.map((h) => h.id).toList(), ['a', 'near-a']);
      expect(hits.first.score, closeTo(1.0, 1e-6));
      expect(hits[1].score, greaterThan(0.9));
    });

    test('topK larger than size returns all', () {
      final index = EmbeddingIndex()..add('a', _v([1, 0]));
      expect(index.search(_v([1, 0]), topK: 10).length, 1);
    });

    test('searchWithThreshold filters and sorts', () {
      final index = EmbeddingIndex()
        ..add('a', _v([1, 0, 0]))
        ..add('b', _v([0, 1, 0]))
        ..add('near-a', _v([0.9, 0.1, 0]));

      final hits = index.searchWithThreshold(_v([1, 0, 0]), threshold: 0.5);
      expect(hits.map((h) => h.id).toList(), ['a', 'near-a']);
    });

    test('empty index search returns empty', () {
      expect(EmbeddingIndex().search(_v([1, 0])), isEmpty);
    });

    test('addAll and overwrite by id replaces the stored vector', () {
      final index = EmbeddingIndex()
        ..addAll({'a': _v([1, 0]), 'b': _v([0, 1])});
      expect(index.length, 2);

      index.add('a', _v([0, 1])); // overwrite a's vector [1,0] -> [0,1]
      expect(index.length, 2); // no new entry, still 2

      double scoreOf(String id, Float32List query) =>
          index.search(query, topK: 2).firstWhere((h) => h.id == id).score;

      // The overwrite took effect: 'a' now matches its new vector and no
      // longer matches the old one.
      expect(scoreOf('a', _v([0, 1])), closeTo(1.0, 1e-6));
      expect(scoreOf('a', _v([1, 0])), closeTo(0.0, 1e-6));
    });

    test('dimension mismatch throws on add and on query', () {
      final index = EmbeddingIndex()..add('a', _v([1, 0, 0]));
      expect(() => index.add('b', _v([1, 0])), throwsArgumentError);
      expect(() => index.search(_v([1, 0])), throwsArgumentError);
    });

    test('toBytes/fromBytes round-trips exactly (float)', () {
      final index = EmbeddingIndex()
        ..add('doc-1', _v([0.1, 0.2, 0.3, 0.4]))
        ..add('doc-2', _v([-0.5, 0.5, 0.0, 1.0]));

      final restored = EmbeddingIndex.fromBytes(index.toBytes());
      expect(restored.length, 2);
      expect(restored.isQuantized, isFalse);
      expect(restored.dimension, 4);

      final q = _v([0.1, 0.2, 0.3, 0.4]);
      final before = index.search(q);
      final after = restored.search(q);
      expect(after.map((h) => h.id).toList(), before.map((h) => h.id).toList());
      for (var i = 0; i < before.length; i++) {
        expect(after[i].score, closeTo(before[i].score, 1e-6));
      }
    });

    test('quantized storage round-trips and searches', () {
      final index = EmbeddingIndex(quantized: true)
        ..add('a', _v([1, 0, 0]))
        ..add('b', _v([0, 1, 0]))
        ..add('near-a', _v([0.9, 0.1, 0]));

      expect(index.isQuantized, isTrue);
      final hits = index.search(_v([1, 0, 0]), topK: 2);
      expect(hits.map((h) => h.id).toList(), ['a', 'near-a']);

      final restored = EmbeddingIndex.fromBytes(index.toBytes());
      expect(restored.isQuantized, isTrue);
      final after = restored.search(_v([1, 0, 0]), topK: 2);
      expect(after.map((h) => h.id).toList(), ['a', 'near-a']);
      expect(after.first.score, closeTo(hits.first.score, 1e-6));
    });

    test('fromBytes rejects a non-index blob', () {
      expect(
        () => EmbeddingIndex.fromBytes(Uint8List.fromList([1, 2, 3])),
        throwsArgumentError,
      );
    });

    test('empty index round-trips', () {
      final restored = EmbeddingIndex.fromBytes(EmbeddingIndex().toBytes());
      expect(restored.isEmpty, isTrue);
      expect(restored.dimension, isNull);
    });

    test('a very long id (> 65535 bytes) round-trips', () {
      final longId = 'x' * 70000;
      final index = EmbeddingIndex()..add(longId, _v([1, 0, 0]));
      final restored = EmbeddingIndex.fromBytes(index.toBytes());
      expect(restored.length, 1);
      expect(restored.contains(longId), isTrue);
    });

    test('fromBytes rejects a blob with a bad magic byte', () {
      expect(
        () => EmbeddingIndex.fromBytes(Uint8List(14)..[0] = 0xFF),
        throwsArgumentError,
      );
    });

    test('fromBytes rejects a truncated blob', () {
      final full = (EmbeddingIndex()..add('a', _v([1, 2, 3, 4]))).toBytes();
      expect(
        () => EmbeddingIndex.fromBytes(full.sublist(0, full.length - 2)),
        throwsArgumentError,
      );
    });

    test('fromBytes rejects an unsupported version byte', () {
      // Magic-valid header, but version 3 is beyond what the decoder reads.
      final blob = Uint8List.fromList([
        0x4D, 0x32, 0x56, 0x49, // magic 'M2VI'
        0x03, // version 3 (unsupported)
        0x00, // flags: not quantized
        0x00, 0x00, 0x00, 0x00, // dim = 0
        0x00, 0x00, 0x00, 0x00, // count = 0
      ]);
      expect(
        () => EmbeddingIndex.fromBytes(blob),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('unsupported EmbeddingIndex version'),
          ),
        ),
      );
    });

    test('removing the last entry resets the dimension to null', () {
      final index = EmbeddingIndex()..add('a', _v([1, 2, 3]));
      expect(index.dimension, 3);

      expect(index.remove('a'), isTrue);
      expect(index.isEmpty, isTrue);
      expect(index.dimension, isNull);

      // A cleared dimension lets a different-dim vector be added afresh.
      index.add('b', _v([1, 0]));
      expect(index.dimension, 2);
    });

    test('negative topK returns empty', () {
      final index = EmbeddingIndex()..add('a', _v([1, 0]));
      expect(index.search(_v([1, 0]), topK: -1), isEmpty);
    });

    group('payload', () {
      test('search returns the payload stored with each vector', () {
        final index = EmbeddingIndex()
          ..add('a', _v([1, 0, 0]), payload: 'alpha')
          ..add('b', _v([0, 1, 0]), payload: 'beta');

        final hits = index.search(_v([1, 0, 0]), topK: 1);
        expect(hits.single.id, 'a');
        expect(hits.single.payload, 'alpha');
      });

      test('an entry added without a payload has a null payload', () {
        final index = EmbeddingIndex()..add('a', _v([1, 0, 0]));
        expect(index.search(_v([1, 0, 0])).single.payload, isNull);
      });

      test('payloads survive toBytes/fromBytes round-trip', () {
        final index = EmbeddingIndex()
          ..add('a', _v([1, 0, 0]), payload: 'alpha')
          ..add('b', _v([0, 1, 0])); // no payload

        final restored = EmbeddingIndex.fromBytes(index.toBytes());
        final byId = {
          for (final r in restored.search(_v([1, 0, 0]), topK: 2))
            r.id: r.payload,
        };
        expect(byId['a'], 'alpha');
        expect(byId['b'], isNull);
      });

      test('an empty-string payload round-trips distinct from null', () {
        final index = EmbeddingIndex()
          ..add('empty', _v([1, 0]), payload: '')
          ..add('none', _v([0, 1]));

        final restored = EmbeddingIndex.fromBytes(index.toBytes());
        final byId = {
          for (final r in restored.search(_v([1, 0]), topK: 2)) r.id: r.payload,
        };
        expect(byId['empty'], '');
        expect(byId['none'], isNull);
      });

      test('a multibyte UTF-8 id and payload round-trip exactly', () {
        // Guards the byte-vs-char offset math: both id and payload contain
        // multibyte code points (CJK + emoji).
        const id = '文書-😀';
        const payload = '説明: café ☕';
        final index = EmbeddingIndex()..add(id, _v([1, 0]), payload: payload);

        final restored = EmbeddingIndex.fromBytes(index.toBytes());
        expect(restored.contains(id), isTrue);
        final hit = restored.search(_v([1, 0])).single;
        expect(hit.id, id);
        expect(hit.payload, payload);
      });

      test('quantized index carries payloads too', () {
        final index = EmbeddingIndex(quantized: true)
          ..add('a', _v([1, 0, 0]), payload: 'alpha');
        final restored = EmbeddingIndex.fromBytes(index.toBytes());
        expect(restored.search(_v([1, 0, 0])).single.payload, 'alpha');
      });

      test('a v1 blob (no payload section) still decodes', () {
        // Golden v1 blob: one entry id='a', dim=2, float [1,0], not quantized.
        final v1 = Uint8List.fromList([
          0x4D, 0x32, 0x56, 0x49, // magic 'M2VI'
          0x01, // version 1
          0x00, // flags: not quantized
          0x02, 0x00, 0x00, 0x00, // dim = 2
          0x01, 0x00, 0x00, 0x00, // count = 1
          0x01, 0x00, 0x00, 0x00, // idLen = 1
          0x61, // 'a'
          0x00, 0x00, 0x80, 0x3F, // 1.0f little-endian
          0x00, 0x00, 0x00, 0x00, // 0.0f
        ]);
        final index = EmbeddingIndex.fromBytes(v1);
        expect(index.length, 1);
        expect(index.contains('a'), isTrue);
        expect(index.search(_v([1, 0])).single.payload, isNull);
      });
    });
  });
}
