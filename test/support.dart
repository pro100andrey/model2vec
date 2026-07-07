// Shared fixtures for the core Model2Vec integration tests.
//
// The native model is a single process-global, so every core test file loads
// one of these known models. Centralizing the ids and their dimensions here
// removes the magic numbers that used to be duplicated across the suite.

/// Small, fast English model used as the default fixture across the suite.
const testModelId = 'minishlab/potion-base-2M';

/// Embedding dimension of [testModelId].
const testModelDim = 64;

/// Conservative lower bound on the vocabulary size of [testModelId].
const testModelVocabMin = 20000;

/// A second, larger model used to exercise model switching.
const largeModelId = 'minishlab/potion-base-8M';

/// Embedding dimension of [largeModelId].
const largeModelDim = 256;
