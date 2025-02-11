//! The data structure design roughly follows [MiniRAG](https://github.com/HKUDS/MiniRAG/blob/main/minirag/kg/postgres_impl.py).
//!

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
    chuck_idx: i64,
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
    conn: Connection,
}
