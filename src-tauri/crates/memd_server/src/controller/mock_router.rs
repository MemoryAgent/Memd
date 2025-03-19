use crate::metric::MetricData;

use super::{Result, StorePayload};
use axum::Json;
use memd_rag::method::QueryResults;

async fn open_benchmark_api() -> &'static str {
    "happy for challenge."
}

async fn store_api(Json(text): Json<StorePayload>) -> Result<&'static str> {
    println!("added text: {:?}", text);
    Ok("added")
}

async fn query_api(query: String) -> Result<Json<QueryResults>> {
    println!("query: {:?}", query);
    Ok(Json(QueryResults(vec![])))
}

async fn chat_api(question: String) -> Result<String> {
    println!("question: {:?}", question);
    Ok(question.to_string())
}

async fn close_benchmark_api() -> Json<MetricData> {
    Json(MetricData {
        request_metrics: vec![],
        start_memory_usage: 0,
        end_memory_usage: 0,
    })
}

pub fn make_router() -> axum::Router {
    axum::Router::new()
        .route("/open", axum::routing::post(open_benchmark_api))
        .route("/store", axum::routing::post(store_api))
        .route("/query", axum::routing::post(query_api))
        .route("/chat", axum::routing::post(chat_api))
        .route("/close", axum::routing::post(close_benchmark_api))
}
