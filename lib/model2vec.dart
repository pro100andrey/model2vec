/// A Dart package that provides a simple interface to the Model2Vec Rust
/// library for generating text embeddings.
library;

export 'src/exception.dart' show Model2VecErrorKind, Model2VecException;
export 'src/model2vec_base.dart' show Model2Vec;
export 'src/recommended_model.dart' show RecommendedModel;
export 'src/utils.dart' show Model2VecUtils;
