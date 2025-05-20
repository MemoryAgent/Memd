use crate::component::deepseek;

use super::{
    component::{
        self,
        database::Chunk,
        operation::{self, Document},
    },
    QueryResult, QueryResults,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveRAGOption {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
}

impl Default for NaiveRAGOption {
    fn default() -> Self {
        Self {
            chunk_size: Default::default(),
            chunk_overlap: Default::default(),
            top_k: Default::default(),
        }
    }
}

pub async fn chunking(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<Vec<Chunk>> {
    let stored_doc = local_comps.store.add_document(&doc)?;
    let chunks = operation::chunk_document(
        doc,
        stored_doc.id,
        opt.chunk_size,
        opt.chunk_overlap,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )
    .await?;
    let inserted_chunks = chunks
        .iter()
        .map(|c| local_comps.store.add_chunk(c))
        .collect::<Result<Vec<Chunk>>>()?;
    Ok(inserted_chunks)
}

pub async fn bulk_build_index(
    chunks: &[Chunk],
    local_comps: &mut component::LocalComponent,
) -> Result<()> {
    local_comps.index.bulk_build_chunk(chunks)
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<()> {
    let stored_doc = local_comps.store.add_document(&doc)?;
    let chunks = operation::chunk_document(
        doc,
        stored_doc.id,
        opt.chunk_size,
        opt.chunk_overlap,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )
    .await?;
    for chunk in &chunks {
        local_comps.store.add_chunk(&chunk)?;
    }
    Ok(())
}

pub fn retrieve(
    question: &str,
    local_comps: &mut component::LocalComponent,
    top_k: usize,
) -> Result<Vec<(Chunk, f64)>> {
    let question_embedding = component::bert::encode_single_sentence(
        question,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )?;

    let search_results = local_comps
        .index
        .query_chunk(&question_embedding.to_vec1()?, top_k)?;

    search_results
        .iter()
        .map(|(chunk_id, conf)| {
            Ok((
                local_comps
                    .store
                    .find_chunk_by_id((*chunk_id).try_into()?)?,
                (*conf).try_into()?,
            ))
        })
        .collect::<Result<Vec<(Chunk, f64)>>>()
}

#[tokio::test]
async fn test_retrieve() {
    let question = "What is the capital of France?";

    let mut local_comps = component::LocalComponent::default();

    let opt = NaiveRAGOption {
        chunk_size: 10,
        chunk_overlap: 2,
        top_k: 5,
    };

    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Paris is the capital of France. Berlin is the capital of Germany".to_string(),
    };

    insert(&doc, &mut local_comps, &opt).await.unwrap();

    let chunks = retrieve(question, &mut local_comps, opt.top_k).unwrap();

    println!("{:?}", chunks);
}

pub fn query(
    query: &str,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<QueryResults> {
    let chunks = retrieve(query, local_comps, opt.top_k)?;
    let mut doc_ids = chunks
        .iter()
        .map(|(chunk, conf)| (chunk.full_doc_id, *conf))
        .collect::<Vec<(i64, f64)>>();
    doc_ids.dedup_by_key(|x| x.0);
    doc_ids
        .iter()
        .map(|(doc_id, conf)| {
            let doc = local_comps.store.find_document_by_id(*doc_id)?;
            Ok(QueryResult {
                document: Document {
                    name: doc.doc_name.clone(),
                    content: doc.content.clone(),
                },
                conf_score: *conf,
            })
        })
        .collect::<Result<Vec<QueryResult>>>()
        .map(|x| QueryResults(x))
}

fn rerank(chunks: Vec<(Chunk, f64)>) -> Vec<Chunk> {
    let mut chunks = chunks;
    chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    chunks.iter().map(|(chunk, _)| chunk.clone()).collect()
}

fn build_rag_prompt(chunks: Vec<Chunk>, question: &str) -> String {
    format!(
        "You have following context information:
        {}
        Please answer the following question:
        {}",
        chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<String>>()
            .join("\n"),
        question
    )
}

pub async fn chat(
    question: &str,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<String> {
    let chunks = retrieve(question, local_comps, opt.top_k)?;
    let reranked_chunks = rerank(chunks);
    let prompt = build_rag_prompt(reranked_chunks, question);
    info!("Prompt: {}", prompt);
    let answer = local_comps.llm.llm_complete(&prompt)?;
    info!("Answer: {}", answer);
    let (_, answer) = deepseek::extract_answer(&answer);
    Ok(answer.to_string())
}

#[tokio::test]
async fn test_query() {
    let question = "What is the capital of France?";

    let mut local_comps = component::LocalComponent::default();

    let opt = NaiveRAGOption {
        chunk_size: 10,
        chunk_overlap: 2,
        top_k: 5,
    };

    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Paris is the capital of France. Berlin is the capital of Germany".to_string(),
    };

    insert(&doc, &mut local_comps, &opt).await.unwrap();

    let doc = query(question, &mut local_comps, &opt);

    println!("{:?}", doc);
}

#[tokio::test]
async fn test_chat() {
    let question = "What is the capital of France?";

    let mut local_comps = component::LocalComponent::default();

    let opt = NaiveRAGOption {
        chunk_size: 10,
        chunk_overlap: 2,
        top_k: 5,
    };

    let doc = component::operation::Document {
        name: "test".to_string(),
        content: "Paris is the capital of France. Berlin is the capital of Germany".to_string(),
    };

    insert(&doc, &mut local_comps, &opt).await.unwrap();

    let doc = chat(question, &mut local_comps, &opt).await;

    println!("{:?}", doc);
}
