use std::sync::LazyLock;

use tauri_plugin_dialog::{DialogExt, FilePath};
use tokio::{fs::File, sync::RwLock};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use tauri_plugin_http::reqwest::{self, Body};
use tokio_util::codec::{BytesCodec, FramedRead};

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

const UPLOAD_ENDPOINT: &str = "/upload";

macro_rules! build_endpoint {
    ($endpoint: ident) => {
        LazyLock::new(|| format!("{}{}", REMOTE_ADDR, $endpoint))
    };
}

fn create_http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder().no_proxy().build()?)
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

fn open_session() -> Result<usize> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(SESSION_ENDPOINT);
    let json: Session = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()?
        .get(&*ENDPOINT)
        .send()?
        .json()?;
    Ok(json.session)
}

#[tauri::command]
async fn pick_file(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, RwLock<ServeMode>>,
) -> Result<(), ()> {
    let file_path = pick_file_async(app_handle).await.ok_or(())?;
    match &*state.read().await {
        ServeMode::LOCAL => upload_local(),
        ServeMode::REMOTE { http_client, .. } => {
            upload_file(file_path, &http_client).await.unwrap()
        }
    };
    Ok(())
}

fn upload_local() {
    println!("upload later");
}

async fn pick_file_async(app_handle: tauri::AppHandle) -> Option<FilePath> {
    tokio::task::spawn_blocking(move || app_handle.dialog().file().blocking_pick_file())
        .await
        .ok()
        .unwrap_or_default()
}

fn file_to_body(file: File) -> Body {
    let stream = FramedRead::new(file, BytesCodec::new());
    let body = Body::wrap_stream(stream);
    body
}

async fn upload_file(file_path: FilePath, client: &reqwest::Client) -> Result<()> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(UPLOAD_ENDPOINT);
    let file = File::open(file_path.into_path().unwrap()).await?;
    let res = client
        .post(&*ENDPOINT)
        .body(file_to_body(file))
        .send()
        .await?;
    println!("{:?}", res);
    Ok(())
}

fn build_remote_app() -> Result<ServeMode> {
    let http_client = create_http_client()?;
    let session = open_session()?;
    Ok(ServeMode::REMOTE {
        http_client,
        session,
    })
}

fn build_app(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let serve_mode = build_remote_app()
        .map_err(|err| {
            println!(
                "connect to online host failed, fallback to local:\n {}",
                err
            )
        })
        .unwrap_or(ServeMode::LOCAL);
    app.manage(RwLock::new(serve_mode));
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .setup(build_app)
        .invoke_handler(tauri::generate_handler![chat, pick_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
