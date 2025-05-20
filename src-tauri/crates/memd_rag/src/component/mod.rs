//! component provides common used items for building a RAG system.

use crate::index::MemdIndex;

pub mod bert;
pub mod cache;
pub mod database;
pub mod operation;
pub mod sqlite;

// TODO: consider put deepseek module below LLM module
// TODO: add phi-4, GLM-edge, Qwen2.5-3B-Instruct, MiniCPM3-4B
// TODO: as well as Deepseek R1 for baseline comparison
pub mod deepseek;

pub mod cloud_model;

// TODO: compile llama.cpp in android
#[cfg(not(target_os = "android"))]
pub mod llm;

// TODO: Remove this hack after compile llama.cpp
#[cfg(target_os = "android")]
pub mod llm {
    use super::*;
    use crate::operation::Relation;
    use anyhow::Result;

    pub struct Llm;

    impl Default for LocalLlm {
        fn default() -> Self {
            Self
        }
    }

    impl LocalLlm {
        pub fn extract_entities(&self, _question: &str) -> Result<Vec<String>> {
            Ok(vec![])
        }

        pub fn complete(&self, _prompt: &str) -> Result<String> {
            Ok("".to_string())
        }

        pub fn extract_relation(
            &self,
            _question: &str,
            _entities: &Vec<String>,
        ) -> Result<Vec<Relation>> {
            Ok(vec![])
        }
    }

    pub fn build_complete_prompt(
        _question: &str,
        _entity: &str,
        _relations: &Vec<Relation>,
        _texts: &Vec<String>,
    ) -> String {
        "".to_string()
    }
}

/// [`LocalComponent`] is a set of common used components, made for convenience.
pub struct LocalComponent {
    pub tokenizer: tokenizers::Tokenizer,
    pub bert: candle_transformers::models::bert::BertModel,
    pub llm: Box<dyn llm::LLM + Sync + Send>,
    pub cache: Box<dyn cache::VecStore + Sync + Send>,
    pub store: database::Store,
    pub index: Box<dyn MemdIndex + Sync + Send>,
}

impl std::fmt::Debug for LocalComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LOCAL").finish()
    }
}

impl Default for LocalComponent {
    fn default() -> Self {
        let (bert, tokenizer) =
            bert::build_model_and_tokenizer(None, None).expect("init embedding model failed.");
        let cache = Box::new(cache::InMemCache::new());
        let llm = Box::new(llm::LocalLlm::default());
        let store = database::Store::default();
        let index = Box::new(
            usearch::new_index(&usearch::IndexOptions {
                dimensions: 384,
                ..Default::default()
            })
            .unwrap(),
        );

        Self {
            tokenizer,
            bert,
            llm,
            cache,
            store,
            index,
        }
    }
}

impl LocalComponent {
    pub fn reset(&mut self) {
        self.store.reset();
    }
}
