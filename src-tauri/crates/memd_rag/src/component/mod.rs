//! component provides common used items for building a RAG system.

pub mod bert;
pub mod cache;
pub mod database;
pub mod operation;
pub mod sqlite;

// TODO: compile llama.cpp in android
#[cfg(not(target_os = "android"))]
pub mod llm;

#[cfg(target_os = "android")]
pub mod llm {
    use super::*;
    use crate::operation::Relation;
    use anyhow::Result;

    pub struct Llm;

    impl Default for Llm {
        fn default() -> Self {
            Self
        }
    }

    impl Llm {
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
    pub llm: llm::Llm,
    pub cache: Box<dyn cache::VecStore + Sync + Send>,
    pub store: database::Store,
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
        let llm = llm::Llm::default();
        let store = database::Store::default();
        Self {
            tokenizer,
            bert,
            llm,
            cache,
            store,
        }
    }
}
