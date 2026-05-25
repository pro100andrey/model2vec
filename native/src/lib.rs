use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::RwLock;
use anyhow::Result;
use std::path::Path;

mod model;
use model::StaticModel;

static MODEL: RwLock<Option<StaticModel>> = RwLock::new(None);

#[no_mangle]
pub extern "C" fn init_embedder(model_path: *const c_char) -> i32 {
    init_embedder_advanced(model_path, ptr::null(), ptr::null(), -1, ptr::null())
}

#[no_mangle]
pub extern "C" fn init_embedder_advanced(
    model_path: *const c_char,
    hf_token: *const c_char,
    cache_dir: *const c_char,
    normalize: i32, // -1: default, 0: false, 1: true
    subfolder: *const c_char,
) -> i32 {
    let path = unsafe {
        if model_path.is_null() { return -1; }
        CStr::from_ptr(model_path).to_string_lossy().into_owned()
    };

    let token = unsafe {
        if hf_token.is_null() { None }
        else { Some(CStr::from_ptr(hf_token).to_string_lossy().into_owned()) }
    };

    let cache = unsafe {
        if cache_dir.is_null() { None }
        else { Some(CStr::from_ptr(cache_dir).to_string_lossy().into_owned()) }
    };

    let norm = match normalize {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    };

    let sub = unsafe {
        if subfolder.is_null() { None }
        else { Some(CStr::from_ptr(subfolder).to_string_lossy().into_owned()) }
    };

    match load_model_advanced(&path, token.as_deref(), cache.as_deref(), norm, sub.as_deref()) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn init_embedder_from_bytes(
    tokenizer_ptr: *const u8,
    tokenizer_len: usize,
    model_ptr: *const u8,
    model_len: usize,
    config_ptr: *const u8,
    config_len: usize,
) -> i32 {
    if tokenizer_ptr.is_null() || model_ptr.is_null() || config_ptr.is_null() {
        return -1;
    }

    let tokenizer_bytes = unsafe { std::slice::from_raw_parts(tokenizer_ptr, tokenizer_len) };
    let model_bytes = unsafe { std::slice::from_raw_parts(model_ptr, model_len) };
    let config_bytes = unsafe { std::slice::from_raw_parts(config_ptr, config_len) };

    match StaticModel::from_bytes(tokenizer_bytes, model_bytes, config_bytes, None) {
        Ok(model) => {
            let mut model_lock = match MODEL.write() {
                Ok(lock) => lock,
                Err(_) => return -4, // Poisoned lock
            };
            *model_lock = Some(model);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn get_model_list() -> *mut c_char {
    let list = serde_json::json!([
        {
            "id": "minishlab/potion-base-2M",
            "name": "Potion Base 2M",
            "lang": "English",
            "params": "1.8M",
            "description": "Smallest English model, very fast."
        },
        {
            "id": "minishlab/potion-base-8M",
            "name": "Potion Base 8M",
            "lang": "English",
            "params": "7.5M",
            "description": "Balanced English model."
        },
        {
            "id": "minishlab/potion-multilingual-128M",
            "name": "Potion Multilingual 128M",
            "lang": "Multilingual (101)",
            "params": "128M",
            "description": "Best for multi-language tasks."
        }
    ]);
    let s = list.to_string();
    CString::new(s).unwrap().into_raw()
}

fn load_model_advanced(
    path: &str, 
    token: Option<&str>, 
    cache_dir: Option<&str>,
    normalize: Option<bool>, 
    subfolder: Option<&str>
) -> Result<()> {
    let cache_path = cache_dir.map(Path::new);
    let model = StaticModel::from_pretrained(path, token, cache_path, normalize, subfolder)?;
    
    let mut model_lock = MODEL.write().map_err(|_| anyhow::anyhow!("Poisoned lock"))?;
    *model_lock = Some(model);
    Ok(())
}

#[no_mangle]
pub extern "C" fn get_embedding_dimension() -> i32 {
    MODEL.read().ok()
        .and_then(|lock| lock.as_ref().map(|m| m.dim() as i32))
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn get_vocabulary_size() -> i32 {
    MODEL.read().ok()
        .and_then(|lock| lock.as_ref().map(|m| m.vocabulary_size() as i32))
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn is_normalized() -> i32 {
    MODEL.read().ok()
        .and_then(|lock| lock.as_ref().map(|m| if m.is_normalized() { 1 } else { 0 }))
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn get_median_token_length() -> i32 {
    MODEL.read().ok()
        .and_then(|lock| lock.as_ref().map(|m| m.median_token_length() as i32))
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn tokenize(text: *const c_char) -> *mut c_char {
    let model_lock = match MODEL.read() {
        Ok(lock) => lock,
        Err(_) => return ptr::null_mut(),
    };
    let model = match model_lock.as_ref() {
        Some(m) => m,
        None => return ptr::null_mut(),
    };

    let input_text = unsafe {
        if text.is_null() { return ptr::null_mut(); }
        CStr::from_ptr(text).to_string_lossy()
    };

    let encoding = model.tokenizer.encode(input_text.to_string(), false).unwrap();
    let tokens = encoding.get_tokens();
    let json = serde_json::to_string(&tokens).unwrap_or_default();
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn generate_embedding(text: *const c_char, out_vector: *mut f32, _max_len: usize) -> i32 {
    generate_batch_embeddings_advanced(&text, 1, out_vector, 512, 1)
}

#[no_mangle]
pub extern "C" fn generate_batch_embeddings(texts_ptr: *const *const c_char, count: usize, out_vectors: *mut f32) -> i32 {
    generate_batch_embeddings_advanced(texts_ptr, count, out_vectors, 512, 1024)
}

#[no_mangle]
pub extern "C" fn generate_batch_embeddings_advanced(
    texts_ptr: *const *const c_char,
    count: usize,
    out_vectors: *mut f32,
    max_length: usize,
    batch_size: usize,
) -> i32 {
    let model_lock = match MODEL.read() {
        Ok(lock) => lock,
        Err(_) => return -4, // Poisoned lock
    };
    let model = match model_lock.as_ref() {
        Some(m) => m,
        None => return -1, // Not initialized
    };

    if texts_ptr.is_null() || out_vectors.is_null() { return -2; }

    let mut texts = Vec::with_capacity(count);
    for i in 0..count {
        unsafe {
            let ptr = *texts_ptr.add(i);
            if ptr.is_null() { return -3; }
            texts.push(CStr::from_ptr(ptr).to_string_lossy().into_owned());
        }
    }

    let results = model.encode_with_args(&texts, Some(max_length), batch_size);
    let dim = model.dim();
    for (i, embedding) in results.iter().enumerate() {
        unsafe {
            let target_ptr = out_vectors.add(i * dim);
            ptr::copy_nonoverlapping(embedding.as_ptr(), target_ptr, dim);
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        if s.is_null() { return; }
        let _ = CString::from_raw(s);
    }
}
