use super::component::{self, database::Chunk, operation};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct NaiveRAGOption {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<()> {
    let stored_doc = local_comps.store.add_document(&doc.name)?;
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
) -> Result<Vec<Chunk>> {
    let question_embedding = component::bert::encode_single_sentence(
        question,
        &mut local_comps.tokenizer,
        &local_comps.bert,
    )?;

    let search_results = local_comps
        .store
        .vector_search(&question_embedding.to_vec1()?, top_k)?;

    search_results
        .iter()
        .map(|(chunk_id, _)| {
            local_comps
                .store
                .find_chunk_by_id((*chunk_id).try_into().unwrap())
        })
        .collect::<Result<Vec<Chunk>>>()
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

pub async fn query(
    question: &str,
    local_comps: &mut component::LocalComponent,
    opt: &NaiveRAGOption,
) -> Result<String> {
    let chunks = retrieve(question, local_comps, opt.top_k)?;
    let prompt = build_rag_prompt(chunks, question);
    let answer = local_comps.llm.complete(&prompt)?;
    Ok(answer)
}
