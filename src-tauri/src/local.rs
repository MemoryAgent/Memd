use anyhow::Result;
use candle_transformers::models::bert::BertModel;
use tokenizers::Tokenizer;

use crate::bert::encode_prompt;

pub fn chat_local(
    question: &str,
    tokenizer: &mut Tokenizer,
    bert_model: &BertModel,
) -> Result<String> {
    let encoded = encode_prompt(question, tokenizer, bert_model)?;
    Ok(format!("you said {}", encoded))
}
