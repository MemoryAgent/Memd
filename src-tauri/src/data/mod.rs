//! The data structure design roughly follows [MiniRAG](https://github.com/HKUDS/MiniRAG/blob/main/minirag/kg/postgres_impl.py).
//!

use std::sync::Mutex;

use anyhow::{Context, Result};
use rusqlite::Connection;

type DocumentId = i64;
#[derive(Debug, Clone)]
pub struct Document {
    id: DocumentId,
    doc_name: String,
}

type ChunkId = i64;
#[derive(Debug, Clone)]
pub struct Chunk {
    id: ChunkId,
    full_doc_id: DocumentId,
    chunk_index: i64,
    tokens: usize,
    content: String,
    content_vector: Vec<f32>,
}

type EntityId = i64;
#[derive(Debug, Clone)]
pub struct Entity {
    id: EntityId,
    name: String,
    embedding: Vec<f32>,
}

type RelationId = i64;
#[derive(Debug, Clone)]
pub struct Relation {
    id: RelationId,
    source_id: EntityId,
    target_id: EntityId,
    relationship: String,
}

impl Relation {
    pub fn parse(s: &str) -> Self {
        let xs: Vec<String> = s.split(',').map(|x| x.to_string()).collect();
        Relation {
            id: 0,
            source_id: 0,
            target_id: 0,
            relationship: xs[1].clone(),
        }
    }
}

mod sqlite;

pub struct Store {
    conn: Mutex<Connection>,
}

impl Default for Store {
    fn default() -> Self {
        Self {
            conn: Mutex::new(rusqlite::Connection::open_in_memory().unwrap()),
        }
    }
}

impl Store {
    pub fn new(path: &str) -> Self {
        Self {
            conn: Mutex::new(rusqlite::Connection::open(path).unwrap()),
        }
    }

    pub fn add_document(&self, doc_name: &str) -> Result<Document> {
        sqlite::insert_document(&mut self.conn.lock().unwrap(), doc_name)
    }

    pub fn add_chunk(
        &self,
        full_doc_id: DocumentId,
        chuck_index: i64,
        tokens: usize,
        content: &str,
        content_vector: &Vec<f32>,
    ) -> Result<Chunk> {
        sqlite::insert_chunk(
            &mut self.conn.lock().unwrap(),
            full_doc_id,
            chuck_index,
            tokens,
            content,
            content_vector,
        )
    }
}
