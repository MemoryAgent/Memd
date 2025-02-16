use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, SystemTime},
};

use axum::{extract::State, routing::post, Json, Router};
use memd_lib::offline::LocalComponent;
use serde::{Deserialize, Serialize};
use tokio::signal::{unix::signal, unix::SignalKind};
use tracing::info;

#[derive(Deserialize)]
struct StorePayload(Vec<String>);

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

async fn bench_open(State(mut bs_state): State<AppState>) -> &'static str {
    bs_state.metrics.reset();
    "happy for challenge."
}

async fn bench_store(
    State(mut bs_state): State<AppState>,
    text: Json<StorePayload>,
) -> &'static str {
    bs_state.metrics.start_embedding();
    memd_lib::offline::add_local(text.0 .0, &mut bs_state.local_comps)
        .await
        .unwrap();
    bs_state.metrics.end_embedding();
    "added"
}

async fn bench_query(State(mut bs_state): State<AppState>, query: String) -> String {
    bs_state.metrics.start_query();
    let answer = memd_lib::offline::query_local(&query, &mut bs_state.local_comps)
        .await
        .unwrap_or("not found".to_string());
    bs_state.metrics.end_query();
    answer
}

async fn bench_close(State(bs_state): State<AppState>) -> Json<MetricData> {
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
    let app = App::new(LocalComponent::default());

    let app_state = AppState(Arc::new(app));

    let router = Router::new()
        .route("/open", post(bench_open))
        .route("/store", post(bench_store))
        .route("/query", post(bench_query))
        .route("/close", post(bench_close))
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
