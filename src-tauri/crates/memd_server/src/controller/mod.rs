use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use axum::{http::StatusCode, response::IntoResponse};
use memd_rag::{
    component::{database::Chunk, LocalComponent},
    method::naive_rag::NaiveRAGOption,
};
use serde::{Deserialize, Serialize};
use tokio::{
    signal::unix::{signal, SignalKind},
    sync::Mutex,
};

use crate::metric::Metrics;

pub struct App {
    pub local_comps: LocalComponent,
    metrics: Metrics,
    rag_options: NaiveRAGOption,
    bulk_insertion_buffer: Vec<Chunk>,
}

impl App {
    pub fn new(local_comps: LocalComponent) -> Self {
        Self {
            local_comps,
            metrics: Metrics::new(),
            rag_options: NaiveRAGOption::default(),
            bulk_insertion_buffer: vec![],
        }
    }

    pub fn reset(&mut self) {
        self.local_comps.reset();
    }

    pub fn add_to_buffer(&mut self, chunk: &[Chunk]) {
        self.bulk_insertion_buffer.extend_from_slice(chunk);
    }
}

#[derive(Clone)]
pub struct AppState(pub Arc<Mutex<App>>);

impl Deref for AppState {
    type Target = Arc<Mutex<App>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AppState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
pub struct ServerMetadata {
    opt: NaiveRAGOption,
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
