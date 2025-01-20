use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use candle_transformers::models::bert::BertModel;
use serde::Deserialize;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::task::JoinHandle;
use tracing::info;

use crate::{
    bert::{build_model_and_tokenizer, encode_prompt, encode_sentence},
    db::{InMemDB, VecStore},
};

pub struct LocalComponent {
    tokenizer: Tokenizer,
    bert: BertModel,
    db: Box<dyn VecStore + Sync + Send>,
}

impl Default for LocalComponent {
    fn default() -> Self {
        let (bert, tokenizer) =
            build_model_and_tokenizer(None, None).expect("init embedding model failed.");
        let db = Box::new(InMemDB::new());
        Self {
            tokenizer,
            bert,
            db,
        }
    }
}

pub struct LocalState {
    comps: Arc<RwLock<LocalComponent>>,
}

impl Default for LocalState {
    fn default() -> Self {
        LocalState {
            comps: Arc::new(RwLock::new(LocalComponent::default())),
        }
    }
}

impl LocalState {
    pub fn handle(&self) -> Arc<RwLock<LocalComponent>> {
        self.comps.clone()
    }
}

#[derive(Deserialize)]
struct StorePayload(Vec<String>);

fn add_comps(text: Vec<String>, local_comps: &mut LocalComponent) -> Result<()> {
    let encoded = encode_sentence(&text, &mut local_comps.tokenizer, &local_comps.bert)?;
    encoded
        .iter()
        .zip(text.iter())
        .for_each(|(embedding, text)| local_comps.db.add(embedding, text));
    Ok(())
}

fn query_comps(prompt: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let encoded = encode_prompt(prompt, &mut local_comps.tokenizer, &local_comps.bert)?;
    let memory = local_comps.db.query(&encoded)?;
    Ok(memory.text)
}

fn chat_comps(question: &str, local_comps: &mut LocalComponent) -> Result<String> {
    let answer = query_comps(question, local_comps)?;
    add_comps(vec![question.to_string()], local_comps)?;
    Ok(format!("you said {}", answer))
}

pub fn chat_local(question: &str, local_state: &LocalState) -> Result<String> {
    let mut local_comps = local_state.comps.write().unwrap();
    chat_comps(question, &mut local_comps)
}

#[allow(unused)]
async fn bench_open() -> &'static str {
    "happy for challenge."
}

#[allow(unused)]
async fn bench_store(
    State(local_component): State<Arc<RwLock<LocalComponent>>>,
    text: Json<StorePayload>,
) -> &'static str {
    add_comps(text.0 .0, &mut local_component.write().unwrap()).unwrap();
    "added"
}

#[allow(unused)]
async fn bench_query(
    State(local_component): State<Arc<RwLock<LocalComponent>>>,
    query: String,
) -> String {
    query_comps(&query, &mut local_component.write().unwrap()).unwrap_or("not found".to_string())
}

async fn bench_close() -> &'static str {
    "Ceterum censeo Carthaginem esse delendam"
}

pub fn open_for_benchmark(local_state: Arc<RwLock<LocalComponent>>) -> Result<JoinHandle<()>> {
    info!("opening server for testing");

    let app = Router::new()
        .route("/open", post(bench_open))
        .route("/store", post(bench_store))
        .route("/query", post(bench_query))
        .route("/close", post(bench_close))
        .with_state(local_state);

    let handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("localhost:3000")
            .await
            .unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    Ok(handle)
}
