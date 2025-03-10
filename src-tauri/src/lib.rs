use std::fmt::Debug;

use memd_rag::component::LocalComponent;
use online::{build_remote_app, RemoteState};
use tauri_plugin_dialog::{DialogExt, FilePath};
use tokio::sync::RwLock;

use anyhow::Result;
use tauri::Manager;
use tracing::info;

pub mod online;

#[derive(Clone, Debug)]
pub struct Config {
    use_online: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { use_online: false }
    }
}

#[derive(Debug)]
pub enum ServeMode {
    LOCAL(LocalComponent),
    REMOTE(RemoteState),
}

fn build_app(app: &mut tauri::App, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let serve_mode = match config.use_online {
        true => build_remote_app()
            .map_err(|err| {
                info!(
                    "connect to online host failed, fallback to local:\n {}",
                    err
                )
            })
            .map(|x| ServeMode::REMOTE(x))
            .unwrap_or(ServeMode::LOCAL(LocalComponent::default())),
        false => ServeMode::LOCAL(LocalComponent::default()),
    };

    info!("serve mode is {:?}", serve_mode);
    app.manage(config.to_owned());
    app.manage(RwLock::new(serve_mode));
    Ok(())
}

#[tauri::command]
async fn chat(question: &str, state: tauri::State<'_, RwLock<ServeMode>>) -> Result<String, ()> {
    let res = match &mut *state.write().await {
        ServeMode::LOCAL(local_state) => {
            memd_rag::method::chat(question, local_state, memd_rag::method::RAGMethods::NoRAG).await
        }
        ServeMode::REMOTE(remote_state) => online::chat_remote(question, remote_state).await,
    };
    Ok(res.unwrap_or_else(|err| format!("An error occurred during our conversation:\n{}", err)))
}

async fn pick_file_async(app_handle: tauri::AppHandle) -> Option<FilePath> {
    tokio::task::spawn_blocking(move || app_handle.dialog().file().blocking_pick_file())
        .await
        .ok()
        .unwrap_or_default()
}

#[tauri::command]
async fn pick_file(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, RwLock<ServeMode>>,
) -> Result<(), ()> {
    let file_path = pick_file_async(app_handle).await.ok_or(())?;
    match &*state.read().await {
        ServeMode::LOCAL { .. } => memd_rag::method::upload_local().await,
        ServeMode::REMOTE(remote_state) => crate::online::upload_file(file_path, remote_state)
            .await
            .unwrap(),
    };
    Ok(())
}

#[tauri::command]
fn set_serve_mode() {
    todo!("")
}

#[tauri::command]
fn refresh_session() {
    todo!("")
}

#[tauri::command]
fn clear_history() {
    todo!("")
}

#[cfg(target_os = "android")]
mod port {
    fn port_options() {}
}

#[cfg(not(target_os = "android"))]
mod port {
    fn port_options() {
        // std::env::set_var("HF_HOME", "");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| build_app(app, &Config::default()))
        .invoke_handler(tauri::generate_handler![
            chat,
            pick_file,
            set_serve_mode,
            refresh_session,
            clear_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
