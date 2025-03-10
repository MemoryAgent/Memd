use super::{AppState, MetricData, Result, StorePayload};
use crate::{component::operation::Document, method::QueryResults};
use axum::{extract::State, Json};

async fn open_benchmark_api(State(mut bs_state): State<AppState>) -> &'static str {
    bs_state.metrics.reset();
    "happy for challenge."
}

async fn store_api(
    State(mut bs_state): State<AppState>,
    Json(text): Json<StorePayload>,
) -> Result<&'static str> {
    bs_state.metrics.start_embedding();
    crate::method::insert(
        &Document {
            name: match text.title {
                Some(title) => title,
                None => "".to_string(),
            },
            content: text.content,
        },
        &mut bs_state.local_comps,
        crate::method::RAGMethods::NoRAG,
    )
    .await?;
    bs_state.metrics.end_embedding();
    Ok("added")
}

/// query is a intermediate step of RAG. It gives the relating document with confidence score.
async fn query_api(
    State(mut bs_state): State<AppState>,
    query: String,
) -> Result<Json<QueryResults>> {
    bs_state.metrics.start_query();
    let answer = crate::method::query(
        &query,
        &mut bs_state.local_comps,
        crate::method::RAGMethods::NoRAG,
    )
    .await?;
    bs_state.metrics.end_query();
    Ok(Json(answer))
}

async fn chat_api(State(mut bs_state): State<AppState>, question: String) -> Result<String> {
    let answer = crate::method::chat(
        &question,
        &mut bs_state.local_comps,
        crate::method::RAGMethods::NoRAG,
    )
    .await?;

    Ok(answer.to_string())
}

async fn close_benchmark_api(State(bs_state): State<AppState>) -> Json<MetricData> {
    Json(bs_state.metrics.report())
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
