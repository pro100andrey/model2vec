#include <stddef.h>

/*
 * Error model
 * -----------
 * Every fallible function returns an int status code:
 *   0            success
 *   > 0          failure; the value is a stable Model2VecErrorKind (see below)
 * On failure the function writes an owned, human-readable message into
 * `*out_error` (when `out_error` is non-null). The caller must release that
 * message with `free_string`. On success `*out_error` is set to NULL.
 *
 * Output buffers
 * --------------
 * The native layer owns every output allocation. `generate_*` write a pointer
 * to a freshly allocated float buffer into `*out_data`; the caller copies it
 * and then releases it with `free_floats(ptr, len)`. String outputs are
 * released with `free_string`.
 *
 * Stable error codes (Model2VecErrorKind):
 *   1 not initialized       2 model load failed     3 init from bytes failed
 *   4 lock poisoned         5 null argument         6 tokenization failed
 *   7 empty result          8 panic
 */

int init_embedder_advanced(
    const char* model_path,
    const char* hf_token,
    const char* cache_dir,
    int normalize,
    const char* subfolder,
    char** out_error
);

int init_embedder_from_bytes(
    const unsigned char* tokenizer_ptr,
    size_t tokenizer_len,
    const unsigned char* model_ptr,
    size_t model_len,
    const unsigned char* config_ptr,
    size_t config_len,
    char** out_error
);

int get_embedding_dimension(int* out_value, char** out_error);

int get_vocabulary_size(int* out_value, char** out_error);

int is_normalized(int* out_value, char** out_error);

int get_median_token_length(int* out_value, char** out_error);

int tokenize(const char* text, char** out_json, char** out_error);

int generate_embedding(
    const char* text,
    size_t max_length,
    float** out_data,
    size_t* out_dim,
    char** out_error
);

int generate_batch_embeddings_advanced(
    const char** texts_ptr,
    size_t count,
    size_t max_length,
    size_t batch_size,
    float** out_data,
    size_t* out_dim,
    size_t* out_count,
    char** out_error
);

void free_string(char* s);

void free_floats(float* ptr, size_t len);
