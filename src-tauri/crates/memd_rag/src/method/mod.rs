pub use crate::component;
use anyhow::Result;

mod hippo_rag;
mod memd_agent;
mod naive_rag;
mod no_rag;
mod raptor;
mod read_agent;

pub enum RAGMethods {
    HippoRAG,
    MemdAgent,
    Raptor,
    ReadAgent,
    NoRAG,
    NaiveRAG(naive_rag::NaiveRAGOption),
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
    method: RAGMethods,
) -> Result<()> {
    match method {
        RAGMethods::HippoRAG => todo!(),
        RAGMethods::MemdAgent => memd_agent::insert(doc, local_comps).await,
        RAGMethods::Raptor => todo!(),
        RAGMethods::ReadAgent => todo!(),
        RAGMethods::NoRAG => no_rag::insert(doc, local_comps),
        RAGMethods::NaiveRAG(opt) => todo!(),
    }
}

pub async fn query(
    question: &str,
    local_comps: &mut component::LocalComponent,
    method: RAGMethods,
) -> Result<String> {
    match method {
        RAGMethods::HippoRAG => todo!(),
        RAGMethods::MemdAgent => memd_agent::query(question, local_comps).await,
        RAGMethods::Raptor => todo!(),
        RAGMethods::ReadAgent => todo!(),
        RAGMethods::NoRAG => no_rag::query(question, local_comps),
        RAGMethods::NaiveRAG(opt) => todo!(),
    }
}

pub async fn upload_local() {
    todo!("later then")
}
