//! The data structure design roughly follows [MiniRAG](https://github.com/HKUDS/MiniRAG/blob/main/minirag/kg/postgres_impl.py).
//!

use std::{collections::HashMap, sync::Mutex};

use anyhow::Result;
use rusqlite::Connection;

use crate::{operation, sqlite::run_migrations};

use super::sqlite;

type DocumentId = i64;
#[derive(Debug, Clone)]
pub struct Document {
    pub id: DocumentId,
    pub doc_name: String,
}

type ChunkId = i64;
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: ChunkId,
    pub full_doc_id: DocumentId,
    pub chunk_index: i64,
    pub tokens: usize,
    pub content: String,
    pub content_vector: Vec<f32>,
}

type EntityId = i64;
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: EntityId,
    pub name: String,
    pub embedding: Vec<f32>,
}

type RelationId = i64;
#[derive(Debug, Clone)]
pub struct Relation {
    pub id: RelationId,
    pub source_id: EntityId,
    pub target_id: EntityId,
    pub relationship: String,
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Default for Store {
    fn default() -> Self {
        let mut conn = Connection::open_in_memory().unwrap();
        run_migrations(&mut conn).unwrap();
        Self { conn: conn.into() }
    }
}

impl Store {
    pub fn new(path: &str) -> Self {
        let mut conn = Connection::open(path).unwrap();
        run_migrations(&mut conn).unwrap();
        Self { conn: conn.into() }
    }

    pub fn add_document(&self, doc_name: &str) -> Result<Document> {
        sqlite::insert_document(&mut self.conn.lock().unwrap(), doc_name)
    }

    pub fn add_chunk(
        &self,
        operation::Chunk {
            full_doc_id,
            tokens,
            content,
            chunk_index,
            embedding,
        }: &operation::Chunk,
    ) -> Result<Chunk> {
        sqlite::insert_chunk(
            &mut self.conn.lock().unwrap(),
            *full_doc_id,
            *chunk_index,
            *tokens,
            content,
            &embedding.to_vec1()?,
        )
    }

    pub fn add_entity(&self, entity: &operation::Entity, chunk: &Chunk) -> Result<Entity> {
        let entity = sqlite::insert_entity(
            &mut self.conn.lock().unwrap(),
            &entity.name,
            &entity.embedding.to_vec1()?,
        )?;
        sqlite::insert_entity_chunk(&mut self.conn.lock().unwrap(), entity.id, chunk.id)?;
        Ok(entity)
    }

    pub fn add_relation(
        &self,
        relation: &operation::Relation,
        mapping: &HashMap<String, EntityId>,
    ) -> Result<Relation> {
        let source_id = mapping.get(&relation.source_name).unwrap();
        let target_id = mapping.get(&relation.target_name).unwrap();
        sqlite::insert_relation(
            &mut self.conn.lock().unwrap(),
            *source_id,
            *target_id,
            &relation.relationship,
        )
    }
}
