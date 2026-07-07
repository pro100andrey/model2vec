use anyhow::Result;
use std::any::Any;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::RwLock;

mod model;
use model::StaticModel;

static MODEL: RwLock<Option<StaticModel>> = RwLock::new(None);

// Stable error codes. Keep in sync with Model2VecErrorKind on the Dart side.
const CODE_NOT_INITIALIZED: i32 = 1;
const CODE_MODEL_LOAD_FAILED: i32 = 2;
const CODE_INIT_FROM_BYTES_FAILED: i32 = 3;
const CODE_LOCK_POISONED: i32 = 4;
const CODE_NULL_ARGUMENT: i32 = 5;
const CODE_TOKENIZATION_FAILED: i32 = 6;
const CODE_EMPTY_RESULT: i32 = 7;
const CODE_PANIC: i32 = 8;

/// A failure that crosses the FFI boundary: a stable code plus an owned message.
struct FfiError {
    code: i32,
    message: String,
}

impl FfiError {
    fn new(code: i32, message: impl Into<String>) -> Self {
        FfiError { code, message: message.into() }
    }
}

/// Runs `f`, converting any error or panic into a status code and writing the
/// message into `*out_error`. `*out_error` is set to null on success.
fn run_ffi<F>(out_error: *mut *mut c_char, f: F) -> i32
where
    F: FnOnce() -> Result<(), FfiError>,
{
    write_error(out_error, None);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            write_error(out_error, Some(&e.message));
            e.code
        }
        Err(panic) => {
            write_error(out_error, Some(&panic_message(&panic)));
            CODE_PANIC
        }
    }
}

fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        format!("panic in native code: {s}")
    } else if let Some(s) = panic.downcast_ref::<String>() {
        format!("panic in native code: {s}")
    } else {
        "panic in native code".to_string()
    }
}

/// Writes an owned copy of `msg` into `*out_error`, or null when `msg` is None.
fn write_error(out_error: *mut *mut c_char, msg: Option<&str>) {
    if out_error.is_null() {
        return;
    }
    let value = match msg {
        None => std::ptr::null_mut(),
        Some(m) => match CString::new(m.replace('\0', " ")) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
    };
    unsafe { *out_error = value };
}

/// Acquires a read lock and hands the active model to `f`.
fn with_model<F>(f: F) -> Result<(), FfiError>
where
    F: FnOnce(&StaticModel) -> Result<(), FfiError>,
{
    let lock = MODEL
        .read()
        .map_err(|_| FfiError::new(CODE_LOCK_POISONED, "model lock poisoned"))?;
    let model = lock
        .as_ref()
        .ok_or_else(|| FfiError::new(CODE_NOT_INITIALIZED, "no model initialized; call loadModel first"))?;
    f(model)
}

fn opt_cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

fn write_i32(out: *mut i32, value: i32) {
    if !out.is_null() {
        unsafe { *out = value };
    }
}

fn write_usize(out: *mut usize, value: usize) {
    if !out.is_null() {
        unsafe { *out = value };
    }
}

/// Moves `v` into a caller-owned allocation and writes its pointer into
/// `*out_data`. Released by `free_floats(ptr, v.len())`.
fn alloc_floats(out_data: *mut *mut f32, v: Vec<f32>) {
    if out_data.is_null() {
        return; // v is dropped
    }
    let mut boxed = v.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe { *out_data = ptr };
}

fn write_string(out: *mut *mut c_char, s: String) -> Result<(), FfiError> {
    if out.is_null() {
        return Ok(());
    }
    let c = CString::new(s)
        .map_err(|_| FfiError::new(CODE_TOKENIZATION_FAILED, "result contained an interior NUL byte"))?;
    unsafe { *out = c.into_raw() };
    Ok(())
}

#[no_mangle]
pub extern "C" fn init_embedder_advanced(
    model_path: *const c_char,
    hf_token: *const c_char,
    cache_dir: *const c_char,
    normalize: i32, // -1: default, 0: false, 1: true
    subfolder: *const c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    run_ffi(out_error, || {
        if model_path.is_null() {
            return Err(FfiError::new(CODE_NULL_ARGUMENT, "model_path is null"));
        }
        let path = unsafe { CStr::from_ptr(model_path) }.to_string_lossy().into_owned();
        let token = opt_cstr(hf_token);
        let cache = opt_cstr(cache_dir);
        let sub = opt_cstr(subfolder);
        let norm = match normalize {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };

        let model = StaticModel::from_pretrained(
            &path,
            token.as_deref(),
            cache.as_deref().map(Path::new),
            norm,
            sub.as_deref(),
        )
        .map_err(|e| FfiError::new(CODE_MODEL_LOAD_FAILED, format!("failed to load model '{path}': {e}")))?;

        let mut lock = MODEL
            .write()
            .map_err(|_| FfiError::new(CODE_LOCK_POISONED, "model lock poisoned"))?;
        *lock = Some(model);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn init_embedder_from_bytes(
    tokenizer_ptr: *const u8,
    tokenizer_len: usize,
    model_ptr: *const u8,
    model_len: usize,
    config_ptr: *const u8,
    config_len: usize,
    out_error: *mut *mut c_char,
) -> i32 {
    run_ffi(out_error, || {
        if tokenizer_ptr.is_null() || model_ptr.is_null() || config_ptr.is_null() {
            return Err(FfiError::new(CODE_NULL_ARGUMENT, "one or more byte pointers are null"));
        }
        let tokenizer_bytes = unsafe { std::slice::from_raw_parts(tokenizer_ptr, tokenizer_len) };
        let model_bytes = unsafe { std::slice::from_raw_parts(model_ptr, model_len) };
        let config_bytes = unsafe { std::slice::from_raw_parts(config_ptr, config_len) };

        let model = StaticModel::from_bytes(tokenizer_bytes, model_bytes, config_bytes, None)
            .map_err(|e| FfiError::new(CODE_INIT_FROM_BYTES_FAILED, format!("init from bytes failed: {e}")))?;

        let mut lock = MODEL
            .write()
            .map_err(|_| FfiError::new(CODE_LOCK_POISONED, "model lock poisoned"))?;
        *lock = Some(model);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn get_embedding_dimension(out_value: *mut i32, out_error: *mut *mut c_char) -> i32 {
    run_ffi(out_error, || with_model(|m| {
        write_i32(out_value, m.dim() as i32);
        Ok(())
    }))
}

#[no_mangle]
pub extern "C" fn get_vocabulary_size(out_value: *mut i32, out_error: *mut *mut c_char) -> i32 {
    run_ffi(out_error, || with_model(|m| {
        write_i32(out_value, m.vocabulary_size() as i32);
        Ok(())
    }))
}

#[no_mangle]
pub extern "C" fn is_normalized(out_value: *mut i32, out_error: *mut *mut c_char) -> i32 {
    run_ffi(out_error, || with_model(|m| {
        write_i32(out_value, if m.is_normalized() { 1 } else { 0 });
        Ok(())
    }))
}

#[no_mangle]
pub extern "C" fn get_median_token_length(out_value: *mut i32, out_error: *mut *mut c_char) -> i32 {
    run_ffi(out_error, || with_model(|m| {
        write_i32(out_value, m.median_token_length() as i32);
        Ok(())
    }))
}

#[no_mangle]
pub extern "C" fn tokenize(
    text: *const c_char,
    out_json: *mut *mut c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    run_ffi(out_error, || {
        if text.is_null() {
            return Err(FfiError::new(CODE_NULL_ARGUMENT, "text is null"));
        }
        let input = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();
        with_model(|model| {
            let encoding = model
                .tokenizer
                .encode(input, false)
                .map_err(|e| FfiError::new(CODE_TOKENIZATION_FAILED, format!("tokenization failed: {e}")))?;
            let json = serde_json::to_string(encoding.get_tokens())
                .map_err(|e| FfiError::new(CODE_TOKENIZATION_FAILED, format!("failed to serialize tokens: {e}")))?;
            write_string(out_json, json)
        })
    })
}

#[no_mangle]
pub extern "C" fn generate_embedding(
    text: *const c_char,
    max_length: usize,
    out_data: *mut *mut f32,
    out_dim: *mut usize,
    out_error: *mut *mut c_char,
) -> i32 {
    run_ffi(out_error, || {
        if text.is_null() {
            return Err(FfiError::new(CODE_NULL_ARGUMENT, "text is null"));
        }
        let text_str = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();
        with_model(|model| {
            let results = model
                .encode_with_args(&[text_str], Some(max_length), 1)
                .map_err(|e| FfiError::new(CODE_TOKENIZATION_FAILED, format!("{e}")))?;
            let embedding = results
                .into_iter()
                .next()
                .ok_or_else(|| FfiError::new(CODE_EMPTY_RESULT, "embedding result was empty"))?;
            write_usize(out_dim, embedding.len());
            alloc_floats(out_data, embedding);
            Ok(())
        })
    })
}

#[no_mangle]
pub extern "C" fn generate_batch_embeddings_advanced(
    texts_ptr: *const *const c_char,
    count: usize,
    max_length: usize,
    batch_size: usize,
    out_data: *mut *mut f32,
    out_dim: *mut usize,
    out_count: *mut usize,
    out_error: *mut *mut c_char,
) -> i32 {
    run_ffi(out_error, || {
        if texts_ptr.is_null() {
            return Err(FfiError::new(CODE_NULL_ARGUMENT, "texts pointer is null"));
        }
        let mut texts = Vec::with_capacity(count);
        for i in 0..count {
            let ptr = unsafe { *texts_ptr.add(i) };
            if ptr.is_null() {
                return Err(FfiError::new(CODE_NULL_ARGUMENT, format!("null text at index {i}")));
            }
            texts.push(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned());
        }

        with_model(|model| {
            let results = model
                .encode_with_args(&texts, Some(max_length), batch_size)
                .map_err(|e| FfiError::new(CODE_TOKENIZATION_FAILED, format!("{e}")))?;
            if results.len() != count {
                return Err(FfiError::new(
                    CODE_EMPTY_RESULT,
                    format!("expected {count} embeddings, got {}", results.len()),
                ));
            }
            // dim + data are produced under this single read lock, so the
            // dimension the caller frees against always matches the data.
            let dim = model.dim();
            let mut flat = Vec::with_capacity(count * dim);
            for embedding in &results {
                if embedding.len() != dim {
                    return Err(FfiError::new(CODE_EMPTY_RESULT, "embedding dimension mismatch"));
                }
                flat.extend_from_slice(embedding);
            }
            write_usize(out_dim, dim);
            write_usize(out_count, count);
            alloc_floats(out_data, flat);
            Ok(())
        })
    })
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

#[no_mangle]
pub extern "C" fn free_floats(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

/// Writes the current load progress into the out-params. Never fails; any null
/// out-param is skipped. `out_phase` is a load phase code (0 idle, 1 resolving,
/// 2 downloading, 3 parsing, 4 done); the byte counts are 0 when unknown (before
/// a download starts, on a cache hit, or for a local path).
#[no_mangle]
pub extern "C" fn get_load_progress(
    out_phase: *mut i32,
    out_downloaded: *mut usize,
    out_total: *mut usize,
) {
    let (phase, downloaded, total) = model::load_progress();
    write_i32(out_phase, phase as i32);
    write_usize(out_downloaded, downloaded);
    write_usize(out_total, total);
}

/// Arms progress tracking for a new load. Call this on the polling side before
/// starting the load so a previous load's terminal state isn't observed as the
/// new one's. Never fails.
#[no_mangle]
pub extern "C" fn reset_load_progress() {
    model::begin_load();
}

/// Returns 1 if a model is currently loaded, 0 otherwise. Never fails: a
/// poisoned lock is reported as "not loaded".
#[no_mangle]
pub extern "C" fn is_model_loaded() -> i32 {
    match MODEL.read() {
        Ok(lock) => i32::from(lock.is_some()),
        Err(_) => 0,
    }
}

/// Unloads the active model, freeing its memory. Idempotent (no-op when nothing
/// is loaded).
#[no_mangle]
pub extern "C" fn free_embedder(out_error: *mut *mut c_char) -> i32 {
    run_ffi(out_error, || {
        let mut lock = MODEL
            .write()
            .map_err(|_| FfiError::new(CODE_LOCK_POISONED, "model lock poisoned"))?;
        *lock = None;
        Ok(())
    })
}
