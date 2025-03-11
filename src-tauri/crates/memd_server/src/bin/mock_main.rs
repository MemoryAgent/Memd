use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let router = memd_server::controller::mock_router::make_router();

    let listener = tokio::net::TcpListener::bind("localhost:3000")
        .await
        .unwrap();

    info!("mock listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, router)
        .with_graceful_shutdown(memd_server::controller::shutdown_signal())
        .await
        .unwrap();
}
