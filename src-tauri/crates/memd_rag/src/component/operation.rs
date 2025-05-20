//! This module contains main data structures and functions for RAG.
//!
//! They are intermediate data structures emitted during RAG process. It
//! is the purpose of these data structures to better manage the data flow.
//!

use candle_core::Tensor;
use candle_transformers::models::bert::BertModel;
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use anyhow::Result;

use super::{
    bert::{encode_single_sentence, normalize_l2},
    llm::LocalLlm,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub full_doc_id: i64,
    pub chunk_index: i64,
    pub tokens: usize,
    pub content: String,
    pub embedding: Tensor,
}

pub async fn chunk_document(
    Document { content, .. }: &Document,
    full_doc_id: i64,
    max_tokens: usize,
    overlap: usize,
    tokenizer: &mut Tokenizer,
    embedder: &BertModel,
) -> Result<Vec<Chunk>> {
    let tokenizer_impl = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokenizer_impl
        .encode(content.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let mut index = 0;
    let mut start = 0;
    let mut chunks = Vec::new();
    while start < tokens.len() {
        let end = std::cmp::min(start + max_tokens, tokens.len());
        let chunk_tokens = tokens[start..end].to_vec();
        let chunk_content = tokenizer_impl
            .decode(&chunk_tokens, true)
            .map_err(anyhow::Error::msg)?;
        let token_ids = Tensor::new(&chunk_tokens[..], &embedder.device)?.unsqueeze(0)?;
        let embedding = embedder
            .forward(&token_ids, &token_ids.zeros_like()?, None)
            .map_err(anyhow::Error::msg)?;
        let chunk_embedding = normalize_l2(&(embedding.sum((0, 1))? / (tokens.len() as f64))?)?;
        chunks.push(Chunk {
            full_doc_id,
            chunk_index: index,
            tokens: chunk_tokens.len(),
            content: chunk_content,
            embedding: chunk_embedding,
        });

        start += max_tokens - overlap;
        index += 1;
    }
    Ok(chunks)
}

#[tokio::test]
async fn test_chunk_document() {
    use super::bert::build_model_and_tokenizer;

    let (embedder, tokenizer) = build_model_and_tokenizer(None, None).unwrap();
    let doc = Document {
        name: "test".to_string(),
        content: "This is a test document".to_string(),
    };

    let chunks = chunk_document(&doc, 1, 5, 2, &mut tokenizer.clone(), &embedder)
        .await
        .unwrap();

    println!("{:?}", chunks);
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub embedding: Tensor,
}

pub(crate) async fn chunk_extract_entity(
    Chunk { content, .. }: &Chunk,
    llm: &LocalLlm,
    tokenizer: &mut Tokenizer,
    embedder: &BertModel,
) -> Result<Vec<Entity>> {
    let entity_names = llm.extract_entities(&content)?;
    entity_names
        .iter()
        .map(|x| {
            let name = x.trim().to_lowercase();
            let embedding = encode_single_sentence(&name, tokenizer, embedder)?;
            Ok(Entity { name, embedding })
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub source_name: String,
    pub target_name: String,
    pub relationship: String,
}

impl Relation {
    pub fn parse(s: &str) -> Self {
        let xs: Vec<String> = s.split(',').map(|x| x.to_string()).collect();
        Relation {
            source_name: xs[0].trim().to_string(),
            target_name: xs[2].trim().to_string(),
            relationship: xs[1].trim().to_string(),
        }
    }
}

pub async fn chunk_extract_relation(
    Chunk { content, .. }: &Chunk,
    entities: &Vec<Entity>,
    llm: &LocalLlm,
) -> Result<Vec<Relation>> {
    let entity_names = entities.iter().map(|x| x.name.clone()).collect();
    llm.extract_relation(&content, &entity_names)
}

pub async fn graph_search(
    entities: &Vec<super::database::Entity>,
    relations: &Vec<super::database::Relation>,
    ranking_top_k: usize,
) -> Result<Vec<i64>> {
    let edges = relations
        .iter()
        .map(|x| (x.source_id as u32, x.target_id as u32))
        .collect::<Vec<_>>();
    let g = UnGraph::<u32, ()>::from_edges(&edges);
    println!("{:?}", g);
    let output_ranks = petgraph::algo::page_rank(&g, 0.7, 10);
    println!("{:?}", output_ranks);
    // get the argmax index of the output ranks
    let mut output_ranks = output_ranks
        .iter()
        .enumerate()
        .map(|(i, x)| (i as i64, *x)) // TODO: this should be replaced... bad api of petgraph lib
        .collect::<Vec<_>>();

    output_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ranked_entities: Vec<i64> = output_ranks
        .iter()
        .filter(|x| x.0 != 0) // TODO: remove this hack
        .map(|x| x.0)
        .collect();
    let ranked_entities = ranked_entities
        .iter()
        .take(ranking_top_k)
        .map(|x| *x)
        .collect();
    Ok(ranked_entities)
}
