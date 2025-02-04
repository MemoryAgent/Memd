use anyhow::{anyhow, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;
use std::vec;
use tracing::info;

const DEEPSEEK_R1_1B: &str =
    "lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF~DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf";

const CONTEXT_LENGTH: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(32768) };

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
) -> Result<String> {
    let mut batch = LlamaBatch::new(512, 1);

    let last_idx = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_idx;
        batch.add(*token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama decode failed")?;

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.6),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::dist(1234),
    ]);

    let mut answer = String::new();
    while n_cur <= 2048 {
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
            batch.add(token, n_cur, &[0], true)?;
        }

        n_cur += 1;

        ctx.decode(&mut batch)?;

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
    Ok(answer)
}

#[test]
fn llama_test() {
    let backend = create_llm_backend().unwrap();

    let model = create_llm_model(&backend).unwrap();

    let mut ctx = create_llm_ctx(&backend, &model).unwrap();

    let prompt =
        "<｜User｜>How many r's are there in the word strawberry?<｜Assistant｜>".to_string();

    let token_list = tokenize(&model, &prompt).unwrap();

    println!("{}", batch_decode(&model, &mut ctx, &token_list).unwrap());
}

pub struct Llm {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl Llm {
    pub fn new() -> Result<Llm> {
        let backend = create_llm_backend()?;
        let model = create_llm_model(&backend)?;
        Ok(Llm { backend, model })
    }

    pub fn complete(&self, prompt: &str) -> Result<String> {
        let template = self.model.get_chat_template(12000)?;
        let augmented_prompt = self.model.apply_chat_template(
            Some(template),
            vec![LlamaChatMessage::new(
                "user".to_string(),
                prompt.to_string(),
            )?],
            true,
        )?;
        let tokens = tokenize(&self.model, &augmented_prompt)?;
        let mut ctx = create_llm_ctx(&self.backend, &self.model)?;
        batch_decode(&self.model, &mut ctx, &tokens)
    }
}

impl Default for Llm {
    fn default() -> Self {
        Llm::new().unwrap()
    }
}

#[test]
fn test_two_questions() {
    let llm = Llm::default();

    let question_one = llm
        .complete("what is the integration of f(x) = x^2?")
        .unwrap();
    println!("{question_one}");

    let question_two = llm.complete("what is the solution of 1 + 1?").unwrap();
    println!("{question_two}");
}

/// Deepseek R1's answers contains two parts
///
/// <think>
/// what it thinks ...
/// </think>
///
/// answer
///
/// So it is necessary to write a function to parse these two parts
fn extract_answer(answer: &str) -> (&str, &str) {
    assert!(answer.starts_with("<think>\n"));
    let subview = &answer[8..];
    subview.split_once("</think>\n\n").unwrap()
}

#[test]
fn test_extract_answer() {
    let text = "<think>
I am thinking about ...
</think>

The answer is quite straight forward.
";

    let (think, answer) = extract_answer(text);
    assert_eq!(think, "I am thinking about ...\n");
    assert_eq!(answer, "The answer is quite straight forward.\n");
}

fn build_re_prompt(q: &str) -> String {
    format!(
        "---
Extract entities means to identify important nouns from a sentence, such as:
Q: Which country does Tolstoy and Tchaikovsky live?
You should reply with a comma separated string such as:
Tolstoy,Tchaikovsky
---
Please extract all entities in this question:
{}",
        q
    )
}

#[test]
fn test_extract_relation() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let prompt = "Which country does Tolstoy and Tchaikovsky live?";
    let llm = Llm::default();
    println!("{:?}", extract_relation(prompt, &llm));
}

fn extract_relation(q: &str, llm: &Llm) -> Result<Vec<String>> {
    let re_prompt = build_re_prompt(q);
    let answer = llm.complete(&re_prompt)?;
    info!("raw output {}", answer);
    let (thinking, answer) = extract_answer(&answer);
    info!("thinking procedure is {}. answer is {}", thinking, answer);
    Ok(answer.split(',').map(|x| x.to_string()).collect())
}

fn build_parent_prompt(r: &str) -> String {
    format!(
        "---
Finding parent entity means to extract more generalized concept of the entity.
Q: Leo Tolstoy,Victor Hugo
You should reply a noun such as:
Writer
---
Please extract generalized concept from:
{}",
        r
    )
}

#[test]
fn test_parent_prompt() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let entities = vec!["Leo Tolstoy".to_string(), "Victor Hugo".to_string()];
    let llm = Llm::default();
    let pe = get_parent_entity(&entities, &llm).unwrap();
    println!("{}", pe);
}

fn vec_to_csv(v: &Vec<String>) -> String {
    v.iter().fold(String::new(), |mut acc, x| {
        acc.push_str(x);
        acc
    })
}

fn get_parent_entity(v: &Vec<String>, llm: &Llm) -> Result<String> {
    let prompt = vec_to_csv(v);
    let p_prompt = build_parent_prompt(&prompt);
    let answer = llm.complete(&p_prompt)?;
    info!("raw output {}", answer);
    let (thinking, answer) = extract_answer(&answer);
    info!("thinking procedure is {}. answer is {}", thinking, answer);
    Ok(answer.to_string())
}
