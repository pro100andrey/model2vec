use anyhow::{anyhow, Context, Result};
use half::f16;
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::api::Progress;
use hf_hub::Cache;
use rayon::prelude::*;
use safetensors::{tensor::Dtype, SafeTensors};
use serde_json::Value;
use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tokenizers::{ModelWrapper, Tokenizer};

// --- Load progress -------------------------------------------------------
// Only one model is ever loading at a time (the model is a process global), so
// progress lives in process-global atomics that any isolate can poll while the
// load itself runs on another. The phase codes below are mirrored on the Dart
// side (see model2vec.h and load_progress.dart).

/// No load in progress.
pub const PHASE_IDLE: u8 = 0;
/// Locating the model files (resolving repo layout / checking the cache).
pub const PHASE_RESOLVING: u8 = 1;
/// Downloading the model weights from Hugging Face.
pub const PHASE_DOWNLOADING: u8 = 2;
/// Parsing the files and building the in-memory model.
pub const PHASE_PARSING: u8 = 3;
/// The model is loaded and ready.
pub const PHASE_DONE: u8 = 4;

static LOAD_PHASE: AtomicU8 = AtomicU8::new(PHASE_IDLE);
static LOAD_DOWNLOADED: AtomicUsize = AtomicUsize::new(0);
static LOAD_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Arms progress for a fresh load: zeroes the byte counters and enters the
/// resolving phase. Called at the start of a load and by the caller before it,
/// so a previous load's terminal state is never observed as the new one's.
pub fn begin_load() {
    LOAD_DOWNLOADED.store(0, Ordering::Relaxed);
    LOAD_TOTAL.store(0, Ordering::Relaxed);
    LOAD_PHASE.store(PHASE_RESOLVING, Ordering::Relaxed);
}

fn set_phase(phase: u8) {
    LOAD_PHASE.store(phase, Ordering::Relaxed);
}

/// Snapshot of the current load as `(phase, downloaded_bytes, total_bytes)`.
pub fn load_progress() -> (u8, usize, usize) {
    (
        LOAD_PHASE.load(Ordering::Relaxed),
        LOAD_DOWNLOADED.load(Ordering::Relaxed),
        LOAD_TOTAL.load(Ordering::Relaxed),
    )
}

/// hf-hub download sink that writes progress into the global load state.
struct AtomicProgress;

impl Progress for AtomicProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        LOAD_TOTAL.store(size, Ordering::Relaxed);
        LOAD_DOWNLOADED.store(0, Ordering::Relaxed);
        LOAD_PHASE.store(PHASE_DOWNLOADING, Ordering::Relaxed);
    }

    fn update(&mut self, size: usize) {
        LOAD_DOWNLOADED.fetch_add(size, Ordering::Relaxed);
    }

    fn finish(&mut self) {
        let total = LOAD_TOTAL.load(Ordering::Relaxed);
        LOAD_DOWNLOADED.store(total, Ordering::Relaxed);
    }
}

/// Static embedding model for Model2Vec.
///
/// `embeddings` is a flat, row-major `rows * cols` buffer: row `r` (a token's
/// vector) is `embeddings[r * cols .. (r + 1) * cols]`. Keeping it a plain slice
/// (rather than an ndarray) lets the pooling loop iterate contiguous `&[f32]`
/// rows, which the compiler auto-vectorizes.
#[derive(Debug, Clone)]
pub struct StaticModel {
    pub tokenizer: Tokenizer,
    embeddings: Cow<'static, [f32]>,
    rows: usize,
    cols: usize,
    weights: Option<Cow<'static, [f32]>>,
    token_mapping: Option<Cow<'static, [usize]>>,
    normalize: bool,
    median_token_length: usize,
    unk_token_id: Option<usize>,
}

#[derive(Debug, Clone)]
struct ModelFiles {
    tokenizer: PathBuf,
    model: PathBuf,
    config: PathBuf,
}

fn match_local_layout(config_base: &Path, model_base: &Path, config_file: &str) -> Option<ModelFiles> {
    let config = config_base.join(config_file);
    let tokenizer = model_base.join("tokenizer.json");
    let model = model_base.join("model.safetensors");
    (config.exists() && tokenizer.exists() && model.exists()).then_some(ModelFiles {
        tokenizer,
        model,
        config,
    })
}

fn decode_token_mapping(dtype: Dtype, raw: &[u8]) -> Result<Vec<usize>> {
    let mapping = match dtype {
        Dtype::I64 => raw
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect(),
        Dtype::I32 => raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect(),
        other => return Err(anyhow!("unsupported mapping dtype: {:?}", other)),
    };
    Ok(mapping)
}

fn is_not_found(e: &hf_hub::api::sync::ApiError) -> bool {
    use hf_hub::api::sync::ApiError;
    if let ApiError::RequestError(e) = e {
        if let ureq::Error::StatusCode(404) = **e {
            return true;
        }
    }
    false
}

fn match_hub_layout(
    repo: &ApiRepo,
    cache: &Cache,
    repo_id: &str,
    config_prefix: &str,
    model_prefix: &str,
    config_file: &str,
) -> Result<Option<ModelFiles>> {
    let fetch = |path: String| -> Result<Option<PathBuf>> {
        match repo.get(&path) {
            Ok(p) => Ok(Some(p)),
            Err(e) if is_not_found(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    };
    let Some(config) = fetch(format!("{config_prefix}{config_file}"))? else { return Ok(None); };
    let Some(tokenizer) = fetch(format!("{model_prefix}tokenizer.json"))? else { return Ok(None); };
    let model_file = format!("{model_prefix}model.safetensors");
    let Some(model) = fetch_weights_with_progress(repo, cache, repo_id, model_file)? else { return Ok(None); };
    Ok(Some(ModelFiles { tokenizer, model, config }))
}

/// Fetches the (large) weights file, reporting download progress on a cache
/// miss. A cache hit returns immediately with no download and no progress —
/// mirroring `ApiRepo::get`, which we can't reuse because it downloads without
/// a progress hook.
fn fetch_weights_with_progress(
    repo: &ApiRepo,
    cache: &Cache,
    repo_id: &str,
    filename: String,
) -> Result<Option<PathBuf>> {
    if let Some(path) = cache.model(repo_id.to_owned()).get(&filename) {
        return Ok(Some(path));
    }
    match repo.download_with_progress(&filename, AtomicProgress) {
        Ok(path) => Ok(Some(path)),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn resolve_local_model_files(folder: &Path) -> Option<ModelFiles> {
    match_local_layout(folder, folder, "config.json")
        .or_else(|| match_local_layout(folder, folder, "config_sentence_transformers.json"))
        .or_else(|| match_local_layout(folder, &folder.join("0_StaticEmbedding"), "config_sentence_transformers.json"))
        .or_else(|| folder.parent().and_then(|p| match_local_layout(p, folder, "config_sentence_transformers.json")))
}

fn resolve_hub_model_files(repo: &ApiRepo, cache: &Cache, repo_id: &str, prefix: &str) -> Result<ModelFiles> {
    let sub_prefix = format!("{prefix}0_StaticEmbedding/");
    let trimmed = prefix.trim_end_matches('/');
    let parent = match Path::new(trimmed).parent() {
        Some(path) if !path.as_os_str().is_empty() => format!("{}/", path.display()),
        _ => String::new(),
    };

    if let Some(f) = match_hub_layout(repo, cache, repo_id, prefix, prefix, "config.json")? { return Ok(f); }
    if let Some(f) = match_hub_layout(repo, cache, repo_id, prefix, prefix, "config_sentence_transformers.json")? { return Ok(f); }
    if let Some(f) = match_hub_layout(repo, cache, repo_id, prefix, &sub_prefix, "config_sentence_transformers.json")? { return Ok(f); }
    match_hub_layout(repo, cache, repo_id, &parent, prefix, "config_sentence_transformers.json")?
        .ok_or_else(|| anyhow!("no valid model layout found in '{prefix}'"))
}

impl StaticModel {
    pub fn from_bytes<T, M, C>(
        tokenizer_bytes: T,
        model_bytes: M,
        config_bytes: C,
        normalize: Option<bool>,
    ) -> Result<Self>
    where
        T: AsRef<[u8]>,
        M: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
        let cfg: Value = serde_json::from_slice(config_bytes.as_ref()).context("failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

        let safet = SafeTensors::deserialize(model_bytes.as_ref()).context("failed to parse safetensors")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .or_else(|_| safet.tensor("embedding.weight"))
            .context("embeddings tensor not found")?;

        let [rows, cols]: [usize; 2] = tensor.shape().try_into().context("embedding tensor is not 2-D")?;
        let raw = tensor.data();
        let floats: Vec<f32> = match tensor.dtype() {
            Dtype::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            Dtype::F16 => raw.chunks_exact(2).map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
            Dtype::I8 => raw.iter().map(|&b| f32::from(b as i8)).collect(),
            other => return Err(anyhow!("unsupported tensor dtype: {other:?}")),
        };

        let weights = match safet.tensor("weights") {
            Ok(t) => {
                let raw = t.data();
                let v: Vec<f32> = match t.dtype() {
                    Dtype::F64 => raw.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
                    Dtype::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                    Dtype::F16 => raw.chunks_exact(2).map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32()).collect(),
                    other => return Err(anyhow!("unsupported weights dtype: {:?}", other)),
                };
                Some(v)
            }
            Err(_) => None,
        };

        let token_mapping = match safet.tensor("mapping") {
            Ok(t) => Some(decode_token_mapping(t.dtype(), t.data())?),
            Err(_) => None,
        };

        Self::from_owned(tokenizer, floats, rows, cols, normalize, weights, token_mapping)
    }

    pub fn from_pretrained<P: AsRef<Path>>(
        repo_or_path: P,
        token: Option<&str>,
        cache_dir: Option<&Path>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        begin_load();
        let files = resolve_model_files(repo_or_path, token, cache_dir, subfolder)?;
        set_phase(PHASE_PARSING);
        let tokenizer_bytes = fs::read(&files.tokenizer).context("failed to read tokenizer.json")?;
        let config_bytes = fs::read(&files.config).context("failed to read config.json")?;
        // The weights file dwarfs the other two; map it instead of reading it
        // so parsing never holds both the raw file and the decoded f32 table
        // in memory at once.
        let model_file = fs::File::open(&files.model).context("failed to open model.safetensors")?;
        // SAFETY: read-only mapping of a file in the model cache, alive only
        // for the duration of parsing. Concurrent truncation of the mapped
        // file is the standard, accepted mmap caveat.
        let model_bytes = unsafe { memmap2::Mmap::map(&model_file) }.context("failed to mmap model.safetensors")?;
        let model = Self::from_bytes(tokenizer_bytes, &model_bytes[..], config_bytes, normalize)?;
        set_phase(PHASE_DONE);
        Ok(model)
    }

    pub fn from_owned(
        tokenizer: Tokenizer,
        embeddings: Vec<f32>,
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<Vec<f32>>,
        token_mapping: Option<Vec<usize>>,
    ) -> Result<Self> {
        if embeddings.len() != rows * cols {
            return Err(anyhow!("embeddings length {} != rows {} * cols {}", embeddings.len(), rows, cols));
        }
        let (median_token_length, unk_token_id) = Self::compute_metadata(&tokenizer)?;
        Ok(Self {
            tokenizer,
            embeddings: Cow::Owned(embeddings),
            rows,
            cols,
            weights: weights.map(Cow::Owned),
            token_mapping: token_mapping.map(Cow::Owned),
            normalize,
            median_token_length,
            unk_token_id,
        })
    }

    fn compute_metadata(tokenizer: &Tokenizer) -> Result<(usize, Option<usize>)> {
        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|tk| tk.len()).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);
        // Read the unk token straight off the model rather than serializing the
        // whole tokenizer (vocab included) to JSON to look up one field.
        // Unigram stores an unk *id*, not a token, so it stays None — exactly
        // what the old JSON lookup of "unk_token" yielded for it.
        let unk_token: Option<&str> = match tokenizer.get_model() {
            ModelWrapper::WordPiece(m) => Some(&m.unk_token),
            ModelWrapper::WordLevel(m) => Some(&m.unk_token),
            ModelWrapper::BPE(m) => m.unk_token.as_deref(),
            ModelWrapper::Unigram(_) => None,
        };
        let unk_token_id = if let Some(tok) = unk_token {
            let id = tokenizer.token_to_id(tok).ok_or_else(|| anyhow!("unk_token '{tok}' not found"))?;
            Some(id as usize)
        } else {
            None
        };
        Ok((median_token_length, unk_token_id))
    }

    fn truncate_str(s: &str, max_tokens: usize, median_len: usize) -> &str {
        s.char_indices().nth(max_tokens.saturating_mul(median_len)).map_or(s, |(byte_idx, _)| &s[..byte_idx])
    }

    /// The `dim`-length embedding row for token `r`, or an error if `r` is
    /// outside the table (a malformed model), so a bad id becomes a typed error
    /// rather than an out-of-bounds panic.
    fn row(&self, r: usize) -> Result<&[f32]> {
        let dim = self.cols;
        self.embeddings
            .get(r * dim..r * dim + dim)
            .ok_or_else(|| anyhow!("token row {r} out of range (vocab size {})", self.rows))
    }

    /// Encodes `sentences` into one flat, row-major `sentences.len() * cols`
    /// buffer — a single allocation for the whole batch, with no per-sentence
    /// `Vec`. Sentences within a batch are pooled in parallel: each one owns a
    /// disjoint `cols`-sized chunk of the output.
    pub fn encode_flat(
        &self,
        sentences: &[String],
        max_length: Option<usize>,
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        let dim = self.cols;
        let mut out = vec![0.0f32; sentences.len() * dim];
        for (batch, out_batch) in sentences
            .chunks(batch_size)
            .zip(out.chunks_mut(batch_size * dim))
        {
            let truncated: Vec<&str> = batch.iter().map(|text| {
                max_length.map(|max_tok| Self::truncate_str(text, max_tok, self.median_token_length)).unwrap_or(text.as_str())
            }).collect();
            let encodings = self.tokenizer
                .encode_batch_fast(truncated, false)
                .map_err(|e| anyhow!("tokenization failed: {e}"))?;
            if encodings.len() == 1 {
                // A lone sentence isn't worth a trip through the thread pool.
                self.pool_into(encodings[0].get_ids(), max_length, out_batch)?;
            } else {
                encodings
                    .par_iter()
                    .zip(out_batch.par_chunks_mut(dim))
                    .try_for_each(|(encoding, dst)| self.pool_into(encoding.get_ids(), max_length, dst))?;
            }
        }
        Ok(out)
    }

    /// Mean-pools `ids` — skipping unk tokens, then keeping at most
    /// `max_tokens` — into `dst`, an already-zeroed `cols`-length slice. Empty
    /// input leaves the zero vector.
    fn pool_into(&self, ids: &[u32], max_tokens: Option<usize>, dst: &mut [f32]) -> Result<()> {
        let mut count = 0usize;
        let tokens = ids
            .iter()
            .map(|&id| id as usize)
            .filter(|&tok| Some(tok) != self.unk_token_id)
            .take(max_tokens.unwrap_or(usize::MAX))
            .inspect(|_| count += 1);

        match (&self.weights, &self.token_mapping) {
            (Some(weights), Some(mapping)) => {
                for tok in tokens {
                    let row_idx = mapping.get(tok).copied().unwrap_or(tok);
                    let scale = weights.get(tok).copied().unwrap_or(1.0);
                    for (s, &v) in dst.iter_mut().zip(self.row(row_idx)?) {
                        *s += v * scale;
                    }
                }
            }
            (Some(weights), None) => {
                for tok in tokens {
                    let scale = weights.get(tok).copied().unwrap_or(1.0);
                    for (s, &v) in dst.iter_mut().zip(self.row(tok)?) {
                        *s += v * scale;
                    }
                }
            }
            (None, Some(mapping)) => {
                for tok in tokens {
                    let row_idx = mapping.get(tok).copied().unwrap_or(tok);
                    for (s, &v) in dst.iter_mut().zip(self.row(row_idx)?) {
                        *s += v;
                    }
                }
            }
            (None, None) => {
                for tok in tokens {
                    for (s, &v) in dst.iter_mut().zip(self.row(tok)?) {
                        *s += v;
                    }
                }
            }
        }

        if count == 0 {
            return Ok(());
        }
        // One scaling pass instead of two division passes. Dividing by the
        // token count and then by the mean's norm collapses to dividing by the
        // raw sum's norm; the near-zero guard is still evaluated against the
        // mean's norm so its threshold keeps its old meaning.
        let inv = 1.0 / count as f32;
        let scale = if self.normalize {
            let raw_sq: f32 = dst.iter().map(|&v| v * v).sum();
            let mean_sq = raw_sq * inv * inv;
            if mean_sq > 1e-12 { raw_sq.sqrt().recip() } else { inv }
        } else {
            inv
        };
        for x in dst.iter_mut() {
            *x *= scale;
        }
        Ok(())
    }

    pub fn dim(&self) -> usize { self.cols }
    pub fn vocabulary_size(&self) -> usize { self.rows }
    pub fn is_normalized(&self) -> bool { self.normalize }
    pub fn median_token_length(&self) -> usize { self.median_token_length }
}

fn resolve_model_files<P: AsRef<Path>>(
    repo_or_path: P,
    token: Option<&str>,
    cache_dir: Option<&Path>,
    subfolder: Option<&str>,
) -> Result<ModelFiles> {
    let base = repo_or_path.as_ref();
    if base.exists() {
        let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
        return resolve_local_model_files(&folder).ok_or_else(|| anyhow!("no valid model layout found in {folder:?}"));
    }
    download_model_files(repo_or_path.as_ref().to_string_lossy().as_ref(), token, cache_dir, subfolder)
}

fn download_model_files(repo_id: &str, token: Option<&str>, cache_dir: Option<&Path>, subfolder: Option<&str>) -> Result<ModelFiles> {
    // Own the cache so we can both drive the hf-hub API and check it directly
    // for a weights cache hit (matching ApiBuilder: default location, or the
    // caller's cache_dir).
    let cache = match cache_dir {
        Some(path) => Cache::new(path.to_path_buf()),
        None => Cache::default(),
    };
    let mut builder = ApiBuilder::from_cache(cache.clone());
    if let Some(tok) = token { builder = builder.with_token(Some(tok.to_string())); }

    let result = (|| {
        let api = builder.build().context("hf-hub API init failed")?;
        let repo = api.model(repo_id.to_owned());
        let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
        resolve_hub_model_files(&repo, &cache, repo_id, &prefix)
            .with_context(|| format!("could not load '{repo_id}' from HF"))
    })();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal WordLevel tokenizer: whitespace-split words looked up in a
    /// four-entry vocab, everything else mapping to `[UNK]` (id 0).
    const TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "foo": 3},
            "unk_token": "[UNK]"
        }
    }"#;

    // Rows: 0 = [UNK] (poisoned so a leak into pooling is visible),
    // 1 = hello, 2 = world, 3 = foo. dim = 2.
    const EMBEDDINGS: [f32; 8] = [100.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    fn model(normalize: bool, weights: Option<Vec<f32>>, mapping: Option<Vec<usize>>) -> StaticModel {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap();
        StaticModel::from_owned(tokenizer, EMBEDDINGS.to_vec(), 4, 2, normalize, weights, mapping).unwrap()
    }

    fn encode(m: &StaticModel, texts: &[&str], max_length: Option<usize>, batch_size: usize) -> Vec<f32> {
        let sentences: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        m.encode_flat(&sentences, max_length, batch_size).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "expected {expected:?}, got {actual:?}");
        }
    }

    #[test]
    fn metadata_from_wordlevel_model() {
        let m = model(false, None, None);
        assert_eq!(m.unk_token_id, Some(0));
        // Sorted token byte-lengths are [3, 5, 5, 5]; the median index is 2.
        assert_eq!(m.median_token_length, 5);
    }

    #[test]
    fn mean_pooling() {
        let m = model(false, None, None);
        assert_close(&encode(&m, &["hello world"], None, 32), &[2.0, 3.0]);
    }

    #[test]
    fn unk_tokens_are_dropped() {
        let m = model(false, None, None);
        // "zzz" hits [UNK] (row 0 is poisoned with 100s) and must not count
        // toward the mean either.
        assert_close(&encode(&m, &["hello zzz world"], None, 32), &[2.0, 3.0]);
    }

    #[test]
    fn empty_input_yields_zero_vector() {
        let m = model(false, None, None);
        assert_close(&encode(&m, &[""], None, 32), &[0.0, 0.0]);
    }

    #[test]
    fn empty_input_stays_zero_under_normalize() {
        let m = model(true, None, None);
        assert_close(&encode(&m, &[""], None, 32), &[0.0, 0.0]);
    }

    #[test]
    fn weights_scale_rows_by_token_id() {
        let m = model(false, Some(vec![1.0, 2.0, 3.0, 4.0]), None);
        // hello: [1,2]*2, world: [3,4]*3 -> mean [(2+9)/2, (4+12)/2].
        assert_close(&encode(&m, &["hello world"], None, 32), &[5.5, 8.0]);
    }

    #[test]
    fn mapping_redirects_token_to_row() {
        let m = model(false, None, Some(vec![0, 3, 1, 2]));
        // hello (tok 1) -> row 3 = [5,6]; world (tok 2) -> row 1 = [1,2].
        assert_close(&encode(&m, &["hello world"], None, 32), &[3.0, 4.0]);
    }

    #[test]
    fn weights_and_mapping_combine() {
        let m = model(false, Some(vec![1.0, 2.0, 3.0, 4.0]), Some(vec![0, 3, 1, 2]));
        // hello: row 3 * 2 = [10,12]; world: row 1 * 3 = [3,6] -> mean [6.5, 9].
        assert_close(&encode(&m, &["hello world"], None, 32), &[6.5, 9.0]);
    }

    #[test]
    fn normalized_output_has_unit_norm() {
        let m = model(true, None, None);
        let inv_norm = 1.0 / 5.0f32.sqrt();
        assert_close(&encode(&m, &["hello"], None, 32), &[1.0 * inv_norm, 2.0 * inv_norm]);
    }

    #[test]
    fn max_tokens_truncates_after_unk_removal() {
        let m = model(false, None, None);
        // [UNK] is filtered first, then the cap applies — so foo survives.
        let mut dst = vec![0.0f32; 2];
        m.pool_into(&[0, 3], Some(1), &mut dst).unwrap();
        assert_close(&dst, &[5.0, 6.0]);
    }

    #[test]
    fn batch_output_is_row_major_and_ordered() {
        let m = model(false, None, None);
        // batch_size 2 exercises the chunked path with a ragged final chunk.
        let flat = encode(&m, &["hello", "world", "foo", "hello world", ""], None, 2);
        assert_close(&flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn out_of_range_mapping_is_a_typed_error() {
        let m = model(false, None, Some(vec![9, 9, 9, 9]));
        let err = m.encode_flat(&["hello".to_string()], None, 32).unwrap_err();
        assert!(err.to_string().contains("out of range"), "unexpected error: {err}");
    }
}
