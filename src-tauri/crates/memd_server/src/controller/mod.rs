use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, SystemTime},
};

use memd_rag::component::LocalComponent;
use axum::{http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use tokio::signal::{unix::signal, unix::SignalKind};

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

pub struct App {
    local_comps: LocalComponent,
    metrics: Metrics,
}

impl App {
    pub fn new(local_comps: LocalComponent) -> Self {
        Self {
            local_comps,
            metrics: Metrics::default(),
        }
    }
}

#[derive(Clone)]
pub struct AppState(pub Arc<App>);

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

pub async fn shutdown_signal() {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StorePayload {
    pub title: Option<String>,
    pub content: String,
}

#[test]
fn test_store_payload() {
    let sth = StorePayload {
        title: Some("title".to_string()),
        content: "content".to_string(),
    };

    let serialized_sth = serde_json::to_string(&sth).unwrap();

    println!("{}", serialized_sth);
}

pub mod mock_router;

pub mod router;
