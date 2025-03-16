pub use crate::component;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use super::QueryResults;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemdAgentOption {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub retrieve_top_k: usize,
    pub ranking_top_k: usize,
}

impl Default for MemdAgentOption {
    fn default() -> Self {
        Self {
            chunk_size: 20,
            chunk_overlap: 2,
            retrieve_top_k: 1,
            ranking_top_k: 1,
        }
    }
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
) -> Result<()> {
    let stored_doc = local_comps.store.add_document(&doc)?;
    let chunks = component::operation::chunk_document(
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

        let entities = component::operation::chunk_extract_entity(
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
            .collect::<Result<Vec<component::database::Entity>>>()?;
        info!("inserted entities {:?}", stored_entites);

        let relations =
            component::operation::chunk_extract_relation(chunk, &entities, &local_comps.llm)
                .await?;

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
    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = component::LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
}

pub async fn query(
    query: &str,
    local_comps: &mut component::LocalComponent,
    opt: &MemdAgentOption,
) -> Result<QueryResults> {
    Ok(QueryResults(vec![]))
}

pub async fn chat(
    question: &str,
    local_comps: &mut component::LocalComponent,
    opt: &MemdAgentOption,
) -> Result<String> {
    let entities = local_comps.llm.extract_entities(question)?;
    let entities_embedding: Vec<Vec<f32>> = entities
        .iter()
        .flat_map(|x| {
            component::bert::encode_single_sentence(
                x,
                &mut local_comps.tokenizer,
                &local_comps.bert,
            )
            .and_then(|x| x.to_vec1::<f32>().with_context(|| "encoding failed"))
        })
        .collect();
    let matching_entities = local_comps
        .store
        .find_entitiy_ids_by_embeddings(&entities_embedding, opt.retrieve_top_k)?;
    let matching_entity_ids = matching_entities
        .iter()
        .map(|x| x.0.try_into().unwrap())
        .collect::<Vec<_>>();
    let enriched_entities = local_comps
        .store
        .find_entities_by_ids(&matching_entity_ids)?;

    let relations = local_comps
        .store
        .find_relation_by_entities(&enriched_entities)?;

    let all_entity_ids =
        component::operation::graph_search(&enriched_entities, &relations, opt.ranking_top_k)
            .await?;
    let all_entities = local_comps.store.find_entities_by_ids(&all_entity_ids)?;
    let chunks = local_comps
        .store
        .find_chunks_by_entity_ids(&all_entity_ids)?;
    let texts: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
    let enriched_relations = local_comps.store.enrich_relations(&relations)?;
    let prompt = component::llm::build_complete_prompt(
        question,
        if all_entities.len() > 0 {
            &all_entities[0].name
        } else {
            ""
        },
        &enriched_relations,
        &texts,
    );
    local_comps.llm.complete(&prompt)
}

#[tokio::test]
async fn test_query() {
    tracing_subscriber::fmt::init();
    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Beijing is the capital of China.".to_string(),
    };
    let mut local_comps = component::LocalComponent::default();
    insert(&doc, &mut local_comps).await.unwrap();
    let question = "What is the capital of China?";
    let answer = chat(question, &mut local_comps, &MemdAgentOption::default())
        .await
        .unwrap();
    println!("answer: {}", answer);
}

pub async fn query_local(
    prompt: &str,
    local_comps: &mut component::LocalComponent,
) -> Result<String> {
    let encoded = component::bert::encode_single_sentence(
        prompt,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )?;
    let memory = local_comps.cache.query(&encoded)?;
    Ok(memory.text)
}

pub async fn chat_local(
    question: &str,
    local_comps: &mut component::LocalComponent,
) -> Result<String> {
    let answer = query_local(question, local_comps).await?;
    insert(
        &component::operation::Document {
            name: "".to_string(),
            content: question.to_string(),
        },
        local_comps,
    )
    .await?;
    let llm_answer = local_comps.llm.complete(question)?;
    Ok(format!("based on {}, answer is {}", answer, llm_answer))
}

pub async fn upload_local() {
    todo!("later then")
}
