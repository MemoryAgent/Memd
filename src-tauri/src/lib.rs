use std::sync::LazyLock;

use tokio::sync::RwLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use tauri_plugin_http::reqwest;

pub enum ServeMode {
    LOCAL,
    REMOTE {
        http_client: reqwest::Client,
        session: usize,
    },
}

#[tauri::command]
async fn chat(question: &str, state: tauri::State<'_, RwLock<ServeMode>>) -> Result<String, ()> {
    let res = match &*state.read().await {
        ServeMode::LOCAL => chat_local(question),
        ServeMode::REMOTE {
            http_client,
            session,
        } => chat_remote(question, &http_client, *session).await,
    };
    Ok(res.unwrap_or_else(|err| format!("An error occurred during our conversation:\n{}", err)))
}

fn chat_local(question: &str) -> Result<String> {
    Ok(format!("you said {}!", question))
}

const REMOTE_ADDR: &str = "http://localhost:8762";

const TALK_ENDPOINT: &str = "/talk";

const SESSION_ENDPOINT: &str = "/session";

macro_rules! build_endpoint {
    ($endpoint: ident) => {
        LazyLock::new(|| format!("{}{}", REMOTE_ADDR, $endpoint))
    };
}

fn create_http_client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[derive(Serialize)]
struct TalkRequest {
    talk: String,
    session: usize,
}

#[derive(Deserialize)]
struct QueryResponse {
    response: String,
}

async fn chat_remote(question: &str, client: &reqwest::Client, session: usize) -> Result<String> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(TALK_ENDPOINT);
    let request = TalkRequest {
        talk: question.to_string(),
        session,
    };
    let res: QueryResponse = client
        .post(&*ENDPOINT)
        .json(&request)
        .send()
        .await?
        .json()
        .await?;
    Ok(res.response)
}

#[derive(Deserialize)]
struct Session {
    session: usize,
}

fn open_session() -> usize {
    static ENDPOINT: LazyLock<String> = build_endpoint!(SESSION_ENDPOINT);
    let json: Session = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(&*ENDPOINT)
        .send()
        .unwrap()
        .json()
        .unwrap();
    json.session
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let http_client = create_http_client();
            let session = open_session();
            app.manage(RwLock::new(ServeMode::REMOTE {
                http_client,
                session,
            }));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![chat])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
