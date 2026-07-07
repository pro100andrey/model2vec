import 'package:model2vec/model2vec.dart';
import 'package:test/test.dart';

/// A well-formed catalog entry to reuse across the value-type tests.
const _entry = RecommendedModel(
  id: 'minishlab/potion-base-8M',
  name: 'Potion Base 8M',
  lang: 'English',
  params: '7.5M',
  description: 'Balanced English model.',
);

void main() {
  group('RecommendedModel value type', () {
    test('two instances with the same fields are equal', () {
      const other = RecommendedModel(
        id: 'minishlab/potion-base-8M',
        name: 'Potion Base 8M',
        lang: 'English',
        params: '7.5M',
        description: 'Balanced English model.',
      );

      expect(_entry, equals(other));
      expect(_entry.hashCode, equals(other.hashCode));
    });

    test('a differing field breaks equality', () {
      const differ = RecommendedModel(
        id: 'minishlab/potion-base-8M',
        name: 'Potion Base 8M',
        lang: 'English',
        params: '7.5M',
        description: 'A different description.',
      );

      expect(_entry, isNot(equals(differ)));
    });

    test('toString contains the id', () {
      expect(_entry.toString(), contains(_entry.id));
    });
  });

  group('recommendedModels catalog', () {
    const catalog = Model2Vec.recommendedModels;

    test('is non-empty', () {
      expect(catalog, isNotEmpty);
    });

    test('every id is unique', () {
      final ids = catalog.map((m) => m.id).toList();
      expect(ids.toSet(), hasLength(ids.length));
    });

    test('every field is non-empty', () {
      for (final m in catalog) {
        expect(m.id, isNotEmpty, reason: 'id of $m');
        expect(m.name, isNotEmpty, reason: 'name of $m');
        expect(m.lang, isNotEmpty, reason: 'lang of $m');
        expect(m.params, isNotEmpty, reason: 'params of $m');
        expect(m.description, isNotEmpty, reason: 'description of $m');
      }
    });

    test('every id is a well-formed "owner/model" repo id', () {
      final repoId = RegExp(r'^[\w.-]+/[\w.-]+$');
      for (final m in catalog) {
        expect(
          m.id,
          matches(repoId),
          reason: '${m.id} should look like a Hugging Face repo id',
        );
      }
    });
  });
}
