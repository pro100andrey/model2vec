int init_embedder(const char* model_path);

int init_embedder_advanced(
    const char* model_path,
    const char* hf_token,
    const char* cache_dir,
    int normalize,
    const char* subfolder
);

int init_embedder_from_bytes(
    const unsigned char* tokenizer_ptr,
    unsigned long tokenizer_len,
    const unsigned char* model_ptr,
    unsigned long model_len,
    const unsigned char* config_ptr,
    unsigned long config_len
);

int get_embedding_dimension();

int get_vocabulary_size();

int is_normalized();

int get_median_token_length();

char* tokenize(const char* text);

int generate_embedding(
    const char* text,
    float* out_vector,
    unsigned long max_len
);

int generate_batch_embeddings(
    const char** texts_ptr,
    unsigned long count,
    float* out_vectors
);

int generate_batch_embeddings_advanced(
    const char** texts_ptr,
    unsigned long count,
    float* out_vectors,
    unsigned long max_length,
    unsigned long batch_size
);

void free_string(char* s);
