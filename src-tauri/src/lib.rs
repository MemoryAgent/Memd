use std::sync::LazyLock;

use bert::build_model_and_tokenizer;
use candle_transformers::models::bert::BertModel;
use db::{InMemDB, VecStore};
use local::chat_local;
use tauri_plugin_dialog::{DialogExt, FilePath};
use tauri_plugin_http::reqwest;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::Manager;

mod api;
mod bert;
mod db;
mod llm;
mod local;

pub enum ServeMode {
    LOCAL {
        tokenizer: Tokenizer,
        bert: BertModel,
        db: Box<dyn VecStore + Sync + Send>,
    },
    REMOTE {
        http_client: reqwest::Client,
        session: usize,
    },
}

#[tauri::command]
async fn chat(question: &str, state: tauri::State<'_, RwLock<ServeMode>>) -> Result<String, ()> {
    let res = match &mut *state.write().await {
        local_state @ ServeMode::LOCAL { .. } => chat_local(question, local_state),
        ServeMode::REMOTE {
            http_client,
            session,
        } => chat_remote(question, &http_client, *session).await,
    };
    Ok(res.unwrap_or_else(|err| format!("An error occurred during our conversation:\n{}", err)))
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

// TODO: display retreived pictures in the front end.
#[derive(Deserialize)]
struct RelatingAssets {
    asset_dir: String,
}

impl RelatingAssets {
    // TODO: save to a local storage system, giving display layer the address.
    pub fn save(&self) {
        todo!()
    }
}

// TODO: add source in returned responses.
#[derive(Deserialize)]
struct QueryResponse {
    response: String,
    relating_assets: Vec<RelatingAssets>,
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
        ServeMode::LOCAL { .. } => upload_local().await,
        ServeMode::REMOTE { http_client, .. } => {
            upload_file(file_path, &http_client).await.unwrap()
        }
    };
    Ok(())
}

async fn pick_file_async(app_handle: tauri::AppHandle) -> Option<FilePath> {
    tokio::task::spawn_blocking(move || app_handle.dialog().file().blocking_pick_file())
        .await
        .ok()
        .unwrap_or_default()
}

async fn upload_local() {
    println!("upload later");
}

async fn upload_file(file_path: FilePath, client: &reqwest::Client) -> Result<()> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(UPLOAD_ENDPOINT);
    let form = reqwest::multipart::Form::new()
        .file("file", file_path.into_path().unwrap())
        .await?;
    let res = client.post(&*ENDPOINT).multipart(form).send().await?;
    println!("{:?}", res);
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
        .unwrap_or({
            let (bert, tokenizer) = build_model_and_tokenizer(None, None).unwrap();
            let db = Box::new(InMemDB::new());
            ServeMode::LOCAL {
                tokenizer,
                bert,
                db,
            }
        });
    println!(
        "serve mode is {:?}",
        match serve_mode {
            ServeMode::LOCAL { .. } => "local",
            ServeMode::REMOTE { .. } => "remote",
        }
    );
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
        .invoke_handler(tauri::generate_handler![
            chat,
            pick_file,
            set_serve_mode,
            refresh_session,
            clear_history
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
