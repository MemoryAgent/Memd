use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use axum::{http::StatusCode, response::IntoResponse};
use memd_rag::{component::LocalComponent, method::RAGMethods};
use serde::{Deserialize, Serialize};
use tokio::{
    signal::unix::{signal, SignalKind},
    sync::Mutex,
};

use crate::metric::Metrics;

pub struct App {
    local_comps: LocalComponent,
    metrics: Metrics,
    rag_options: RAGMethods,
}

impl App {
    pub fn new(local_comps: LocalComponent) -> Self {
        Self {
            local_comps,
            metrics: Metrics::new(),
            rag_options: RAGMethods::default(),
        }
    }

    pub fn reset(&mut self) {
        self.local_comps.reset();
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
    opt: memd_rag::method::RAGMethods,
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
