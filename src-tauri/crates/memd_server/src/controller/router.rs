use crate::metric::{self, MetricData, Timer};

use super::{AppState, Result, ServerMetadata, StorePayload};
use axum::{debug_handler, extract::State, Json};
use memd_rag::{component::operation::Document, method::QueryResults};

async fn open_benchmark_api(
    State(bs_state): State<AppState>,
    metadata: Json<ServerMetadata>,
) -> Result<&'static str> {
    bs_state.lock().await.metrics.reset();
    bs_state.lock().await.rag_options = metadata.opt.clone();
    Ok("happy for challenge.")
}

#[debug_handler]
async fn store_api(
    State(bs_state): State<AppState>,
    Json(text): Json<StorePayload>,
) -> Result<&'static str> {
    let method = bs_state.lock().await.rag_options.clone();
    let start_memory = metric::get_current_memory().unwrap().physical_mem;
    let timer = Timer::new();
    memd_rag::method::insert(
        &Document {
            name: match text.title {
                Some(title) => title,
                None => "".to_string(),
            },
            content: text.content,
        },
        &mut bs_state.lock().await.local_comps,
        method,
    )
    .await?;
    let elapsed = timer.read()?;
    let store_memory = bs_state.lock().await.local_comps.store.get_memory_usage();
    let end_memory = metric::get_current_memory().unwrap().physical_mem;
    bs_state
        .lock()
        .await
        .metrics
        .add_store_metric(start_memory, end_memory, store_memory, elapsed);
    Ok("added")
}

/// query is a intermediate step of RAG. It gives the relating document with confidence score.
async fn query_api(State(bs_state): State<AppState>, query: String) -> Result<Json<QueryResults>> {
    let method = bs_state.lock().await.rag_options.clone();
    let start_memory = metric::get_current_memory().unwrap().physical_mem;
    let timer = Timer::new();
    let answer =
        memd_rag::method::query(&query, &mut bs_state.lock().await.local_comps, method).await?;
    let elapsed = timer.read()?;
    let store_memory = bs_state.lock().await.local_comps.store.get_memory_usage();
    let end_memory = metric::get_current_memory().unwrap().physical_mem;
    bs_state
        .lock()
        .await
        .metrics
        .add_query_metric(start_memory, end_memory, store_memory, elapsed);
    Ok(Json(answer))
}

async fn chat_api(State(bs_state): State<AppState>, question: String) -> Result<String> {
    let method = bs_state.lock().await.rag_options.clone();
    let start_memory = metric::get_current_memory().unwrap().physical_mem;
    let timer = Timer::new();
    let answer =
        memd_rag::method::chat(&question, &mut bs_state.lock().await.local_comps, method).await?;
    let elapsed = timer.read()?;
    let store_memory = bs_state.lock().await.local_comps.store.get_memory_usage();
    let end_memory = metric::get_current_memory().unwrap().physical_mem;
    bs_state
        .lock()
        .await
        .metrics
        .add_chat_metric(start_memory, end_memory, store_memory, elapsed);
    Ok(answer.to_string())
}

async fn close_benchmark_api(State(bs_state): State<AppState>) -> Json<MetricData> {
    let report = bs_state.lock().await.metrics.get_metrics().clone();
    bs_state.lock().await.metrics.reset();
    bs_state.lock().await.reset();
    Json(report)
}

pub fn make_router(app_state: AppState) -> axum::Router {
    axum::Router::new()
        .route("/open", axum::routing::post(open_benchmark_api))
        .route("/store", axum::routing::post(store_api))
        .route("/query", axum::routing::post(query_api))
        .route("/chat", axum::routing::post(chat_api))
        .route("/close", axum::routing::post(close_benchmark_api))
        .with_state(app_state)
}
