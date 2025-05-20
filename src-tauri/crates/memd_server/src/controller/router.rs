use crate::metric::{self, MetricData, Timer};

use super::{AppState, Result, ServerMetadata, StorePayload};
use axum::{debug_handler, extract::State, Json};
use memd_rag::{
    component::operation::Document,
    method::{naive_rag::chunking, QueryResults},
};
use tracing::info;

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
    memd_rag::method::naive_rag::insert(
        &Document {
            name: match text.title {
                Some(title) => title,
                None => "".to_string(),
            },
            content: text.content,
        },
        &mut bs_state.lock().await.local_comps,
        &method,
    )
    .await?;
    let elapsed = timer.read()?;
    let store_memory = bs_state
        .lock()
        .await
        .local_comps
        .index
        .evaluate_memory_usage()?;
    let end_memory = metric::get_current_memory().unwrap().physical_mem;
    bs_state
        .lock()
        .await
        .metrics
        .add_store_metric(start_memory, end_memory, store_memory, elapsed);
    Ok("added")
}

async fn prepare_api(
    State(bs_state): State<AppState>,
    Json(text): Json<StorePayload>,
) -> Result<&'static str> {
    let method = bs_state.lock().await.rag_options.clone();

    let document = Document {
        name: match text.title {
            Some(title) => title,
            None => "".to_string(),
        },
        content: text.content,
    };

    let timer = Timer::new();
    let chunk = chunking(&document, &mut bs_state.lock().await.local_comps, &method).await?;
    bs_state.lock().await.add_to_buffer(&chunk);
    let elapsed = timer.read()?;

    bs_state
        .lock()
        .await
        .metrics
        .add_store_metric(0, 0, 0, elapsed);
    Ok("queued")
}

async fn bulk_build_api(State(bs_state): State<AppState>) -> Result<&'static str> {
    let buffer = bs_state.lock().await.bulk_insertion_buffer.clone();

    let timer = Timer::new();
    memd_rag::method::naive_rag::bulk_build_index(&buffer, &mut bs_state.lock().await.local_comps)
        .await?;
    let elapsed = timer.read()?;

    let store_memory = bs_state
        .lock()
        .await
        .local_comps
        .index
        .evaluate_memory_usage()?;

    info!("built with {}", store_memory);

    bs_state
        .lock()
        .await
        .metrics
        .add_store_metric(0, 0, store_memory, elapsed);

    Ok("built")
}

/// query is a intermediate step of RAG. It gives the relating document with confidence score.
async fn query_api(State(bs_state): State<AppState>, query: String) -> Result<Json<QueryResults>> {
    let method = bs_state.lock().await.rag_options.clone();
    let start_memory = metric::get_current_memory().unwrap().physical_mem;
    let timer = Timer::new();
    let answer = memd_rag::method::naive_rag::query(
        &query,
        &mut bs_state.lock().await.local_comps,
        &method,
    )?;
    let elapsed = timer.read()?;
    let store_memory = bs_state
        .lock()
        .await
        .local_comps
        .index
        .evaluate_memory_usage()?;
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
    let answer = memd_rag::method::naive_rag::chat(
        &question,
        &mut bs_state.lock().await.local_comps,
        &method,
    )
    .await?;
    let elapsed = timer.read()?;
    let store_memory = bs_state
        .lock()
        .await
        .local_comps
        .index
        .evaluate_memory_usage()?;
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
        .route("/prepare", axum::routing::post(prepare_api))
        .route("/bulk_build", axum::routing::post(bulk_build_api))
        .route("/close", axum::routing::post(close_benchmark_api))
        .with_state(app_state)
}
