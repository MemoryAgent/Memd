use anyhow::{anyhow, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::info;

const DEEPSEEK_R1_1B: &str =
    "lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF~DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf";

const CONTEXT_LENGTH: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(2048) };

pub fn load_llama_model(backend: LlamaBackend, path: impl AsRef<Path>) -> LlamaModel {
    // LlamaModel::load_from_file(&backend, path, params);
    todo!()
}

fn download_from_hf(model_name: &str) -> Result<PathBuf> {
    let (repo_model, gguf_filename) = model_name
        .split_once('~')
        .ok_or(anyhow!("not a valid hf repo"))?;
    ApiBuilder::new()
        .with_progress(true)
        .build()
        .with_context(|| "unable to create hf api")?
        .model(repo_model.to_string())
        .get(gguf_filename)
        .with_context(|| "unable to download model")
}

fn create_llm_backend() -> Result<LlamaBackend> {
    LlamaBackend::init().with_context(|| "initialize llama context failed.")
}

fn create_llm_model(backend: &LlamaBackend) -> Result<LlamaModel> {
    let model_params = LlamaModelParams::default();
    let model_file = download_from_hf(DEEPSEEK_R1_1B)?;
    LlamaModel::load_from_file(backend, model_file, &model_params)
        .with_context(|| "load llama.cpp model failed.")
}

fn create_llm_ctx<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
) -> Result<LlamaContext<'a>> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(CONTEXT_LENGTH));

    model
        .new_context(backend, ctx_params)
        .with_context(|| "create llama.cpp context failed")
}

fn tokenize(model: &LlamaModel, prompt: &str) -> Result<Vec<LlamaToken>> {
    info!("tokenizing {prompt}");
    model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| "tokenize failed")
}

fn batch_decode(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tokens_list: &Vec<LlamaToken>,
) -> String {
    let mut batch = LlamaBatch::new(512, 1);

    let last_idx = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_idx;
        batch.add(*token, i, &[0], is_last).unwrap();
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama decode failed")
        .unwrap();

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1111), LlamaSampler::greedy()]);

    let mut answer = String::new();
    while n_cur <= CONTEXT_LENGTH.get() as i32 {
        {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            sampler.accept(token);

            if model.is_eog_token(token) {
                eprintln!();
                break;
            }

            let output_string = model.token_to_str(token, Special::Tokenize).unwrap();
            answer.push_str(&output_string);

            batch.clear();
            batch.add(token, n_cur, &[0], true).unwrap();
        }

        n_cur += 1;

        ctx.decode(&mut batch).unwrap();

        n_decode += 1;
    }

    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    info!(
        "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
        n_decode,
        duration.as_secs_f32(),
        n_decode as f32 / duration.as_secs_f32()
    );
    answer
}

#[test]
fn llama_test() {
    let backend = create_llm_backend().unwrap();

    let model = create_llm_model(&backend).unwrap();

    let mut ctx = create_llm_ctx(&backend, &model).unwrap();

    let prompt =
        "<｜User｜>How many r's are there in the word strawberry?<｜Assistant｜>".to_string();

    let token_list = tokenize(&model, &prompt).unwrap();

    println!("{}", batch_decode(&model, &mut ctx, &token_list));
}
