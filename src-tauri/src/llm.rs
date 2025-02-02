use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::Path;

const DEEPSEEK_R1_1B: &str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

pub fn load_llama_model(backend: LlamaBackend, path: impl AsRef<Path>) -> LlamaModel {
    // LlamaModel::load_from_file(&backend, path, params);
    todo!()
}


#[test]
fn llama_test() {
    let backend = LlamaBackend::init().unwrap();

    let model_params = LlamaModelParams::default();

    let prompt = "how many rs are there in the word strawberry?".to_string();

    let mut model_params = std::pin::pin!(model_params);

    let model = LlamaModel::load_from_file(&backend, DEEPSEEK_R1_1B, &model_params).unwrap();

    let mut ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));

    let mut ctx = model.new_context(&backend, ctx_params);

    let tokens_list = model.str_to_token(&prompt, AddBos::Always).unwrap();

    println!("prompt is {}", prompt);

    for token in &tokens_list {
        println!("{}", model.token_to_str(*token, Special::Tokenize).unwrap())
    }

    std::io::stdout().flush().unwrap();
}
