pub use crate::component;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod cluster;
mod hippo_rag;
mod memd_agent;
pub mod naive_rag;
mod no_rag;
mod raptor;
mod read_agent;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RAGMethods {
    HippoRAG,
    MemdAgent(memd_agent::MemdAgentOption),
    Raptor,
    ReadAgent,
    NoRAG,
    NaiveRAG(naive_rag::NaiveRAGOption),
}

#[test]
fn test_rag_method_serialization() {
    let method = RAGMethods::MemdAgent(memd_agent::MemdAgentOption::default());
    let serialized = serde_json::to_string(&method).unwrap();
    println!("{}", serialized);
    let method = RAGMethods::NoRAG;
    let serialized = serde_json::to_string(&method).unwrap();
    println!("{}", serialized);
}

impl Default for RAGMethods {
    fn default() -> Self {
        RAGMethods::NoRAG
    }
}

pub async fn insert(
    doc: &component::operation::Document,
    local_comps: &mut component::LocalComponent,
    method: RAGMethods,
) -> Result<()> {
    match method {
        RAGMethods::HippoRAG => todo!(),
        RAGMethods::MemdAgent(_) => memd_agent::insert(doc, local_comps).await,
        RAGMethods::Raptor => todo!(),
        RAGMethods::ReadAgent => todo!(),
        RAGMethods::NoRAG => no_rag::insert(doc, local_comps),
        RAGMethods::NaiveRAG(opt) => naive_rag::insert(doc, local_comps, &opt).await,
    }
}

#[derive(Serialize, Debug, Clone, Deserialize)]
pub struct QueryResult {
    pub document: component::operation::Document,
    pub conf_score: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct QueryResults(pub Vec<QueryResult>);

#[test]
fn test_query_results_serialization() {
    let results = QueryResults(vec![
        QueryResult {
            document: component::operation::Document {
                name: "name".to_string(),
                content: "content".to_string(),
            },
            conf_score: 0.5,
        },
        QueryResult {
            document: component::operation::Document {
                name: "name".to_string(),
                content: "content".to_string(),
            },
            conf_score: 0.5,
        },
    ]);

    let serialized = serde_json::to_string(&results).unwrap();
    println!("{}", serialized);
}

pub async fn query(
    question: &str,
    local_comps: &mut component::LocalComponent,
    method: RAGMethods,
) -> Result<QueryResults> {
    match method {
        RAGMethods::HippoRAG => todo!(),
        RAGMethods::MemdAgent(opt) => memd_agent::query(question, local_comps, &opt).await,
        RAGMethods::Raptor => todo!(),
        RAGMethods::ReadAgent => todo!(),
        RAGMethods::NoRAG => no_rag::query(question, local_comps),
        RAGMethods::NaiveRAG(opt) => naive_rag::query(question, local_comps, &opt),
    }
}

pub async fn chat(
    question: &str,
    local_comps: &mut component::LocalComponent,
    method: RAGMethods,
) -> Result<String> {
    match method {
        RAGMethods::HippoRAG => todo!(),
        RAGMethods::MemdAgent(memd_agent_option) => {
            memd_agent::chat(question, local_comps, &memd_agent_option).await
        }
        RAGMethods::Raptor => todo!(),
        RAGMethods::ReadAgent => todo!(),
        RAGMethods::NoRAG => no_rag::chat(question, local_comps),
        RAGMethods::NaiveRAG(naive_ragoption) => {
            naive_rag::chat(question, local_comps, &naive_ragoption).await
        }
    }
}

pub async fn upload_local() {
    todo!("later then")
}
