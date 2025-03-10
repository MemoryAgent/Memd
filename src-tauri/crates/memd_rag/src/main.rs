use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, SystemTime},
};

use axum::{
    debug_handler, extract::State, http::StatusCode, response::IntoResponse, routing::post, Json,
    Router,
};
use memd_rag::{
    component::{operation::Document, LocalComponent},
    method::QueryResults,
};
use serde::{Deserialize, Serialize};
use tokio::signal::{unix::signal, unix::SignalKind};
use tracing::info;

/// Wrapper for error handling
/// Taken from https://github.com/tokio-rs/axum/blob/main/examples/anyhow-error-response/src/main.rs
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

type Result<T> = std::result::Result<T, AppError>;

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

struct App {
    local_comps: LocalComponent,
    metrics: Metrics,
}

impl App {
    fn new(local_comps: LocalComponent) -> Self {
        Self {
            local_comps,
            metrics: Metrics::default(),
        }
    }
}

#[derive(Clone)]
struct AppState(Arc<App>);

impl Deref for AppState {
    type Target = App;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AppState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::get_mut(&mut self.0).unwrap()
    }
}

async fn open_benchmark_api(State(mut bs_state): State<AppState>) -> &'static str {
    bs_state.metrics.reset();
    "happy for challenge."
}

#[derive(Deserialize)]
struct StorePayload {
    title: Option<String>,
    content: String,
}

async fn store_api(
    State(mut bs_state): State<AppState>,
    Json(text): Json<StorePayload>,
) -> Result<&'static str> {
    bs_state.metrics.start_embedding();
    memd_rag::method::insert(
        &Document {
            name: match text.title {
                Some(title) => title,
                None => "".to_string(),
            },
            content: text.content,
        },
        &mut bs_state.local_comps,
        memd_rag::method::RAGMethods::NoRAG,
    )
    .await?;
    bs_state.metrics.end_embedding();
    Ok("added")
}

#[debug_handler]
/// query is a intermediate step of RAG. It gives the relating document with confidence score.
async fn query_api(
    State(mut bs_state): State<AppState>,
    query: String,
) -> Result<Json<QueryResults>> {
    bs_state.metrics.start_query();
    let answer = memd_rag::method::query(
        &query,
        &mut bs_state.local_comps,
        memd_rag::method::RAGMethods::NoRAG,
    )
    .await?;
    bs_state.metrics.end_query();
    Ok(Json(answer))
}

async fn chat_api(State(mut bs_state): State<AppState>, question: String) -> Result<String> {
    let answer = memd_rag::method::chat(
        &question,
        &mut bs_state.local_comps,
        memd_rag::method::RAGMethods::NoRAG,
    )
    .await?;

    Ok(answer.to_string())
}

async fn close_benchmark_api(State(bs_state): State<AppState>) -> Json<MetricData> {
    Json(bs_state.metrics.report())
}

async fn shutdown_signal() {
    let interrupt = async {
        signal(SignalKind::interrupt())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    let terminate = async {
        signal(SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = interrupt => {},
        _ = terminate => {},
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let app = App::new(LocalComponent::default());

    let app_state = AppState(Arc::new(app));

    let router = Router::new()
        .route("/open", post(open_benchmark_api))
        .route("/store", post(store_api))
        .route("/query", post(query_api))
        .route("/chat", post(chat_api))
        .route("/close", post(close_benchmark_api))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("localhost:3000")
        .await
        .unwrap();

    info!("listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}
