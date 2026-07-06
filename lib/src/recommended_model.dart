/// A curated Potion model recommendation from the Model2Vec catalog.
///
/// This is editorial data (a friendly name, language and "best for" blurb) that
/// only makes sense hand-curated — see `docs/adr/0003` for why the catalog is a
/// hardcoded constant rather than fetched from Hugging Face.
final class RecommendedModel {
  /// Creates a catalog entry.
  const RecommendedModel({
    required this.id,
    required this.name,
    required this.lang,
    required this.params,
    required this.description,
  });

  /// Hugging Face repo id, e.g. `minishlab/potion-base-8M`.
  final String id;

  /// Human-friendly display name.
  final String name;

  /// Language coverage, e.g. `English` or `Multilingual (101)`.
  final String lang;

  /// Approximate parameter count, e.g. `7.5M`.
  final String params;

  /// One-line description of what the model is best for.
  final String description;

  @override
  bool operator ==(Object other) =>
      other is RecommendedModel &&
      other.id == id &&
      other.name == name &&
      other.lang == lang &&
      other.params == params &&
      other.description == description;

  @override
  int get hashCode => Object.hash(id, name, lang, params, description);

  @override
  String toString() => 'RecommendedModel($id)';
}
