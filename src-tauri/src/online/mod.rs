use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tauri_plugin_dialog::FilePath;
use tauri_plugin_http::reqwest;

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

#[derive(Debug)]
pub struct RemoteState {
    http_client: reqwest::Client,
    session: usize,
}

pub fn build_remote_app() -> Result<RemoteState> {
    let http_client = create_http_client()?;
    let session = open_session()?;
    Ok(RemoteState {
        http_client,
        session,
    })
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

pub async fn chat_remote(
    question: &str,
    RemoteState {
        http_client,
        session,
    }: &mut RemoteState,
) -> Result<String> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(TALK_ENDPOINT);
    let request = TalkRequest {
        talk: question.to_string(),
        session: *session,
    };
    let res: QueryResponse = http_client
        .post(&*ENDPOINT)
        .json(&request)
        .send()
        .await?
        .json()
        .await?;
    Ok(res.response)
}

pub async fn upload_file(
    file_path: FilePath,
    RemoteState { http_client, .. }: &RemoteState,
) -> Result<()> {
    static ENDPOINT: LazyLock<String> = build_endpoint!(UPLOAD_ENDPOINT);
    let form = reqwest::multipart::Form::new()
        .file("file", file_path.into_path().unwrap())
        .await?;
    let res = http_client.post(&*ENDPOINT).multipart(form).send().await?;
    println!("{:?}", res);
    Ok(())
}
