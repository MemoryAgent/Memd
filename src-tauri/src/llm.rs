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

use crate::data::Relation;

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
        LlamaSampler::greedy(),
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

//===-----------------------------------------------------------------------===
// code for embedding
//===-----------------------------------------------------------------------===

fn create_embedding_ctx<'a>(
    backend: &'a LlamaBackend,
    model: &'a LlamaModel,
) -> Result<LlamaContext<'a>> {
    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);
    model
        .new_context(backend, ctx_params)
        .with_context(|| "create embedding ctx failed.")
}

fn batch_embedding(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_ith(i)
            .with_context(|| "Failed to get embeddings")?;
        let output_embeddings = if normalise {
            normalize(embedding)
        } else {
            embedding.to_vec()
        };

        output.push(output_embeddings);
    }

    batch.clear();

    Ok(())
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}

fn llm_embedding(
    tokens_lines_list: &Vec<Vec<LlamaToken>>,
    ctx: &mut LlamaContext,
) -> Result<Vec<Vec<f32>>> {
    let mut batch = LlamaBatch::new(512, 1);

    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    let t_main_start = ggml_time_us();

    for tokens in tokens_lines_list {
        // Flush the batch if the next prompt would exceed our batch size
        if (batch.n_tokens() as usize + tokens.len()) > 512 {
            batch_embedding(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;
            max_seq_id_batch = 0;
        }

        batch.add_sequence(tokens, max_seq_id_batch, true)?;
        max_seq_id_batch += 1;
    }
    // Handle final batch
    batch_embedding(ctx, &mut batch, max_seq_id_batch, &mut output, true)?;

    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    info!(
        "Created embeddings for {} tokens in {:.2} s, speed {:.2} t/s\n",
        total_tokens,
        duration.as_secs_f32(),
        total_tokens as f32 / duration.as_secs_f32()
    );

    println!("{}", ctx.timings());
    Ok(output)
}

/// A facade for easier usage of LLM related functions.
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

    pub fn embed(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let tokens = tokenize(&self.model, text)?;
        let batched_tokens = tokens.chunks(512).map(|x| x.to_vec()).collect();
        let mut ctx = create_embedding_ctx(&self.backend, &self.model)?;
        llm_embedding(&batched_tokens, &mut ctx)
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

#[test]
fn test_embedding() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let llm = Llm::default();
    let text = "hello there this is Harry";
    println!("{:?}", llm.embed(text))
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

//===-----------------------------------------------------------------------===
// code for relationships
//===-----------------------------------------------------------------------===

fn build_ee_prompt(q: &str) -> String {
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
fn test_extract_entity() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let prompt = "Which country does Tolstoy and Tchaikovsky live?";
    let llm = Llm::default();
    println!("{:?}", extract_entity(prompt, &llm));
}

fn extract_entity(q: &str, llm: &Llm) -> Result<Vec<String>> {
    let re_prompt = build_ee_prompt(q);
    let answer = llm.complete(&re_prompt)?;
    info!("raw output {}", answer);
    let (thinking, answer) = extract_answer(&answer);
    info!("thinking procedure is {}. answer is {}", thinking, answer);
    Ok(answer.split(',').map(|x| x.to_string()).collect())
}

fn vec_to_csv(v: &Vec<String>) -> String {
    v.iter().fold(String::new(), |mut acc, x| {
        acc.push_str(x);
        acc
    })
}

fn build_re_prompt(passage: &str, entities: &Vec<String>) -> String {
    format!(
        "---
Your task is to construct relationships from the given passages and entity lists.
Respond with triples (entity, predicate, entity) representing relationships between entities.
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
- the triple should be formatted as comma-separated lists. different triples are delimited by line break.
---
Example
Passage: Tolstoy lived in Russia.
Entities: Tolstoy,Russia

You should reply:
Tolstoy,live,Russia
---
Now please do the task
Passage: {}
Entities: {}", passage, vec_to_csv(&entities)
    )
}

#[test]
fn test_re_prompt() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let entities = vec!["Tolstoy".to_string(), "Russia".to_string()];
    let llm = Llm::default();
    let pe = get_re_entity("Tolstoy lived in Russia", &entities, &llm).unwrap();
    println!("{:?}", pe);
}

fn get_re_entity(passage: &str, entities: &Vec<String>, llm: &Llm) -> Result<Vec<Relation>> {
    let prompt = build_re_prompt(passage, entities);
    let answer = llm.complete(&prompt)?;
    info!("raw output {}", answer);
    let (thinking, answer) = extract_answer(&answer);
    info!("thinking procedure is {}. answer is {}", thinking, answer);
    Ok(answer
        .to_string()
        .lines()
        .map(|l| l.to_string())
        .map(|x| Relation::parse(&x))
        .collect())
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

fn get_parent_entity(v: &Vec<String>, llm: &Llm) -> Result<String> {
    let prompt = vec_to_csv(v);
    let p_prompt = build_parent_prompt(&prompt);
    let answer = llm.complete(&p_prompt)?;
    info!("raw output {}", answer);
    let (thinking, answer) = extract_answer(&answer);
    info!("thinking procedure is {}. answer is {}", thinking, answer);
    Ok(answer.to_string())
}

fn build_complete_prompt(
    q: &str,
    topic_entity: &str,
    relations: &Vec<Relation>,
    documents: &Vec<String>,
) -> String {
    format!(
        "---
For the topic entity: {topic_entity}
We have found following relevant relationships: {relations:?}
They are from these documents: {documents:?}
Please answer the question {q}"
    )
}

fn chat_complete_prompt(
    q: &str,
    topic_entity: &str,
    relations: &Vec<Relation>,
    documents: &Vec<String>,
    llm: &Llm,
) -> Result<String> {
    let prompt = build_complete_prompt(q, topic_entity, relations, documents);
    llm.complete(&prompt)
}

#[test]
fn test_complete_prompt() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let llm = Llm::default();
    let answer = chat_complete_prompt(
        "Where did Tolstoy live",
        "Tolstoy",
        &vec![Relation::parse("Tolstoy,lived in,Russia")],
        &vec!["Tolstoy lived in Russia in the late nineteenth century.".to_string()],
        &llm,
    )
    .unwrap();
    println!("{}", answer);
}

impl Llm {
    pub fn extract_entities(&self, text: &str) -> Result<Vec<String>> {
        extract_entity(text, self)
    }

    pub fn extract_relation(&self, text: &str, entities: &Vec<String>) -> Result<Vec<Relation>> {
        get_re_entity(text, entities, self)
    }

    pub fn extract_parent(&self, v: &Vec<String>) -> Result<String> {
        get_parent_entity(v, self)
    }
}
