use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use candle_transformers::models::bert::BertModel;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};
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
    encoded.iter().zip(&text).for_each(|(t, txt)| {
        local_comps.db.add(t, &txt);
    });
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

/// [`Timer`] is used to record the processing time of this program.
#[derive(Default)]
struct Timer {
    total: Duration,
    session_started: Option<SystemTime>,
}

impl Timer {
    fn reset(&mut self) {
        self.total = Duration::from_secs(0);
        self.session_started = None;
    }

    fn start(&mut self) {
        self.session_started = Some(match self.session_started {
            Some(_) => panic!("started timer when it is started."),
            None => SystemTime::now(),
        });
    }

    fn pause(&mut self) {
        self.total += SystemTime::elapsed(&self.session_started.unwrap()).unwrap();
        self.session_started = None
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct MetricData {
    embedding_cost: Duration,
    query_cost: Duration,
}

#[derive(Default)]
struct Metrics {
    embedding_timer: Timer,
    query_timer: Timer,
}

impl Metrics {
    fn reset(&mut self) {
        self.embedding_timer.reset();
        self.query_timer.reset();
    }

    fn start_embedding(&mut self) {
        self.embedding_timer.start();
    }

    fn end_embedding(&mut self) {
        self.embedding_timer.pause();
    }

    fn start_query(&mut self) {
        self.query_timer.start();
    }

    fn end_query(&mut self) {
        self.query_timer.pause();
    }

    fn report(&self) -> MetricData {
        MetricData {
            embedding_cost: self.embedding_timer.total,
            query_cost: self.query_timer.total,
        }
    }
}

struct BenchServerState {
    local_comps: Arc<RwLock<LocalComponent>>,
    metrics: Metrics,
}

impl BenchServerState {
    fn new(local_comps: Arc<RwLock<LocalComponent>>) -> Self {
        Self {
            local_comps,
            metrics: Metrics::default(),
        }
    }
}

#[allow(unused)]
async fn bench_open(State(bs_state): State<Arc<RwLock<BenchServerState>>>) -> &'static str {
    let mut bs_state = bs_state.write().unwrap();
    bs_state.metrics.reset();
    "happy for challenge."
}

#[allow(unused)]
async fn bench_store(
    State(bs_state): State<Arc<RwLock<BenchServerState>>>,
    text: Json<StorePayload>,
) -> &'static str {
    let mut bs_state = bs_state.write().unwrap();
    bs_state.metrics.start_embedding();
    add_comps(text.0 .0, &mut bs_state.local_comps.write().unwrap()).unwrap();
    bs_state.metrics.end_embedding();
    "added"
}

#[allow(unused)]
async fn bench_query(
    State(bs_state): State<Arc<RwLock<BenchServerState>>>,
    query: String,
) -> String {
    let mut bs_state = bs_state.write().unwrap();
    bs_state.metrics.start_query();
    let answer = query_comps(&query, &mut bs_state.local_comps.write().unwrap())
        .unwrap_or("not found".to_string());
    bs_state.metrics.end_query();
    answer
}

async fn bench_close(State(bs_state): State<Arc<RwLock<BenchServerState>>>) -> Json<MetricData> {
    let bs_state = bs_state.read().unwrap();
    Json(bs_state.metrics.report())
}

pub fn open_for_benchmark(local_state: Arc<RwLock<LocalComponent>>) -> Result<JoinHandle<()>> {
    info!("opening server for benchmark");

    let b_state = Arc::new(RwLock::new(BenchServerState::new(local_state)));

    let app = Router::new()
        .route("/open", post(bench_open))
        .route("/store", post(bench_store))
        .route("/query", post(bench_query))
        .route("/close", post(bench_close))
        .with_state(b_state);

    let handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("localhost:3000")
            .await
            .unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    Ok(handle)
}
