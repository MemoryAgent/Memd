use anyhow::Result;
use candle_transformers::models::bert::BertModel;
use database::Entity;
use llm::build_complete_prompt;
use operation::graph_search;
use std::fmt::Debug;
use tokenizers::Tokenizer;
use tracing::info;

mod bert;
mod cache;
mod database;
// TODO: compile llama.cpp in android
#[cfg(not(target_os = "android"))]
mod llm;
#[cfg(target_os = "android")]
mod llm {
    use super::*;
    use crate::operation::Relation;

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

mod operation;
mod sqlite;

use crate::{
    bert::{build_model_and_tokenizer, encode_sentence, encode_single_sentence},
    cache::{InMemCache, VecStore},
    database::{Chunk, Store},
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

pub async fn insert(doc: &operation::Document, local_comps: &mut LocalComponent) -> Result<()> {
    let stored_doc = local_comps.store.add_document(&doc.name)?;
    let chunks = operation::chunk_document(
        doc,
        stored_doc.id,
        512,
        128,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )
    .await?;
    for chunk in &chunks {
        let stored_chunk = local_comps.store.add_chunk(chunk)?;
        info!("inserted chunk {:?}", stored_chunk);

        let entities = operation::chunk_extract_entity(
            chunk,
            &local_comps.llm,
            &mut local_comps.tokenizer,
            &local_comps.bert,
        )
        .await?;

        if entities.is_empty() {
            continue;
        }

        let stored_entites = entities
            .iter()
            .map(|entity| local_comps.store.add_entity(entity, &stored_chunk))
            .collect::<Result<Vec<Entity>>>()?;
        info!("inserted entities {:?}", stored_entites);

        let relations =
            operation::chunk_extract_relation(chunk, &entities, &local_comps.llm).await?;

        if relations.is_empty() {
            continue;
        }

        let entity_name_to_id = stored_entites
            .iter()
            .map(|entity| (entity.name.clone(), entity.id))
            .collect::<std::collections::HashMap<_, _>>();
        let relations = relations
            .iter()
            .map(|relation| local_comps.store.add_relation(relation, &entity_name_to_id))
            .collect::<Result<Vec<_>>>()?;
        info!("inserted relation {:?}", relations);
    }
    Ok(())
}

#[tokio::test]
async fn test_insert() {
    tracing_subscriber::fmt::init();
    let doc = operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
}

pub async fn query(question: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let entities = local_comps.llm.extract_entities(question)?;
    let matching_entities = local_comps.store.find_entities_by_names(&entities)?;
    let relations = local_comps
        .store
        .find_relation_by_entities(&matching_entities)?;
    let all_entity_ids = graph_search(&matching_entities, &relations).await?;
    let all_entities = local_comps.store.find_entities_by_ids(&all_entity_ids)?;
    let chunks = local_comps
        .store
        .find_chunks_by_entity_ids(&all_entity_ids)?;
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
    let prompt = build_complete_prompt(
        question,
        if all_entities.len() > 0 {
            &all_entities[0].name
        } else {
            ""
        },
        &vec![],
        &texts,
    );
    local_comps.llm.complete(&prompt)
}

#[tokio::test]
async fn test_query() {
    tracing_subscriber::fmt::init();
    let doc = operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
    let question = "What is the capital of China?";
    let answer = query(question, &mut local_comps).await.unwrap();
    println!("answer: {}", answer);
}

pub async fn query_local(prompt: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let encoded = encode_single_sentence(prompt, &mut local_comps.tokenizer, &local_comps.bert)?;
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
