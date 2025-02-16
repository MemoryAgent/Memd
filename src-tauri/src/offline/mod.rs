use anyhow::Result;
use candle_transformers::models::bert::BertModel;
use std::fmt::Debug;
use tokenizers::Tokenizer;

mod bert;
mod cache;
mod data;
mod llm;
mod sqlite;

use crate::offline::{
    bert::{build_model_and_tokenizer, encode_prompt, encode_sentence},
    cache::{InMemCache, VecStore},
    data::{Chunk, Store},
    llm::Llm,
};

pub struct LocalComponent {
    tokenizer: Tokenizer,
    bert: BertModel,
    llm: Llm,
    cache: Box<dyn VecStore + Sync + Send>,
    store: Store,
}

impl Debug for LocalComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LOCAL").finish()
    }
}

impl Default for LocalComponent {
    fn default() -> Self {
        let (bert, tokenizer) =
            build_model_and_tokenizer(None, None).expect("init embedding model failed.");
        let cache = Box::new(InMemCache::new());
        let llm = Llm::default();
        let store = Store::default();
        Self {
            tokenizer,
            bert,
            llm,
            cache,
            store,
        }
    }
}

pub async fn add_local(text: Vec<String>, local_comps: &mut LocalComponent) -> Result<()> {
    let encoded = encode_sentence(&text, &mut local_comps.tokenizer, &local_comps.bert)?;
    encoded.iter().zip(&text).for_each(|(t, txt)| {
        local_comps.cache.add(t, &txt);
    });
    Ok(())
}

pub async fn query_local(prompt: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let encoded = encode_prompt(prompt, &mut local_comps.tokenizer, &local_comps.bert)?;
    let memory = local_comps.cache.query(&encoded)?;
    Ok(memory.text)
}

pub async fn chat_local(question: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let answer = query_local(question, local_comps).await?;
    add_local(vec![question.to_string()], local_comps).await?;
    let llm_answer = local_comps.llm.complete(question)?;
    Ok(format!("based on {}, answer is {}", answer, llm_answer))
}

pub async fn upload_local() {
    todo!("later then")
}
