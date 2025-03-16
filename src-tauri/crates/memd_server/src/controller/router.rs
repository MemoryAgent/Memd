use super::{AppState, MetricData, Result, ServerMetadata, StorePayload};
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
    bs_state.lock().await.metrics.start_embedding();
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
    bs_state.lock().await.metrics.end_embedding();
    Ok("added")
}

/// query is a intermediate step of RAG. It gives the relating document with confidence score.
async fn query_api(State(bs_state): State<AppState>, query: String) -> Result<Json<QueryResults>> {
    bs_state.lock().await.metrics.start_query();
    let method = bs_state.lock().await.rag_options.clone();
    let answer =
        memd_rag::method::query(&query, &mut bs_state.lock().await.local_comps, method).await?;
    bs_state.lock().await.metrics.end_query();
    Ok(Json(answer))
}

async fn chat_api(State(bs_state): State<AppState>, question: String) -> Result<String> {
    let method = bs_state.lock().await.rag_options.clone();
    let answer =
        memd_rag::method::chat(&question, &mut bs_state.lock().await.local_comps, method).await?;

    Ok(answer.to_string())
}

async fn close_benchmark_api(State(bs_state): State<AppState>) -> Json<MetricData> {
    Json(bs_state.lock().await.metrics.report())
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
