use std::sync::Arc;

use memd_rag::component::LocalComponent;
use memd_server::controller::{App, AppState};
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let app = App::new(LocalComponent::default());

    let app_state = AppState(Arc::new(app));

    let router = memd_server::controller::router::make_router(app_state);

    let listener = tokio::net::TcpListener::bind("localhost:3000")
        .await
        .unwrap();

    info!("listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, router)
        .with_graceful_shutdown(memd_server::controller::shutdown_signal())
        .await
        .unwrap();
}
