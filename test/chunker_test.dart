import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

void main() {
  group('chunkText', () {
    test('short text is a single chunk', () {
      expect(chunkText('hello world'), ['hello world']);
    });

    test('blank input yields no chunks', () {
      expect(chunkText('   \n\t  '), isEmpty);
    });

    test('splits long text and covers every word exactly once (no overlap)',
        () {
      final words = List.generate(40, (i) => 'token$i');
      final chunks = chunkText(words.join(' '), maxChars: 40, overlap: 0);

      expect(chunks.length, greaterThan(1));
      final emitted = chunks.expand((c) => c.split(' ')).toList();
      expect(emitted, equals(words)); // order preserved, no duplication
    });

    test('overlap re-includes boundary words', () {
      final words = List.generate(40, (i) => 'token$i').join(' ');
      final withOverlap = chunkText(words, maxChars: 40, overlap: 14);
      final without = chunkText(words, maxChars: 40, overlap: 0);

      expect(withOverlap.length, greaterThan(1));
      final totalWith = withOverlap.expand((c) => c.split(' ')).length;
      final totalWithout = without.expand((c) => c.split(' ')).length;
      expect(totalWith, greaterThan(totalWithout));
    });

    test('chunks stay within maxChars except a single oversized word', () {
      const text = 'aaaa bbbbbbbbbb cccccccccccc dddddddddd eeeeeeee ffffff';
      final chunks = chunkText(text, maxChars: 30, overlap: 25);
      for (final chunk in chunks) {
        if (chunk.split(' ').length > 1) {
          expect(chunk.length, lessThanOrEqualTo(30));
        }
      }
    });

    test('validates arguments', () {
      expect(() => chunkText('a', maxChars: 0), throwsArgumentError);
      expect(
        () => chunkText('a', maxChars: 10, overlap: 10),
        throwsArgumentError,
      );
      expect(
        () => chunkText('a', maxChars: 10, overlap: -1),
        throwsArgumentError,
      );
    });
  });
}
