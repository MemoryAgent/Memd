use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::data::{Chunk, Document, Entity, Relation};

refinery::embed_migrations!("migration");

pub(crate) fn run_migrations(conn: &mut Connection) -> Result<()> {
    migrations::runner().run(conn)?;
    Ok(())
}

#[test]
fn test_in_mem_migration() {
    let mut conn = Connection::open_in_memory().unwrap();
    run_migrations(&mut conn).unwrap();
}

pub(crate) fn insert_document(conn: &mut Connection, doc_name: &str) -> Result<Document> {
    conn.query_row(
        "INSERT INTO document (doc_name) VALUES (?) RETURNING *",
        &[doc_name],
        |row| {
            Ok(Document {
                id: row.get(0)?,
                doc_name: row.get(1)?,
            })
        },
    )
    .with_context(|| format!("Failed to insert document {}", doc_name))
}

pub(crate) fn query_document_by_id(conn: &mut Connection, doc_id: i64) -> Result<Document> {
    conn.query_row(
        "SELECT id, doc_name FROM document WHERE id = ?",
        [doc_id],
        |row| {
            Ok(Document {
                id: row.get(0)?,
                doc_name: row.get(1)?,
            })
        },
    )
    .with_context(|| format!("Failed to query document by id {}", doc_id))
}

#[test]
fn test_insert_query_document() {
    let mut conn = Connection::open_in_memory().unwrap();
    run_migrations(&mut conn).unwrap();
    let doc = insert_document(&mut conn, "test").unwrap();
    assert_eq!(doc.doc_name, "test");
}

fn to_binary_string(v: &Vec<f32>) -> Vec<u8> {
    let mut res = Vec::with_capacity(v.len() * 4);
    for x in v {
        res.extend_from_slice(&x.to_le_bytes());
    }
    res
}

fn to_f32(bytes: &Vec<u8>) -> Vec<f32> {
    let mut res = Vec::with_capacity(bytes.len() / 4);
    for i in 0..bytes.len() / 4 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
        res.push(f32::from_le_bytes(buf));
    }
    res
}

pub(crate) fn insert_chunk(
    conn: &mut Connection,
    full_doc_id: i64,
    chunk_index: i64,
    tokens: usize,
    content: &str,
    content_vector: &Vec<f32>,
) -> Result<Chunk> {
    conn.query_row(
        "INSERT INTO chunk (full_doc_id, chunk_index, tokens, content, content_vector) VALUES (?, ?, ?, ?, ?) RETURNING *",
        (full_doc_id, chunk_index, tokens, content, to_binary_string(content_vector)),
        |row| {
            Ok(Chunk {
                id: row.get(0)?,
                full_doc_id: row.get(1)?,
                chunk_index: row.get(2)?,
                tokens: row.get(3)?,
                content: row.get(4)?,
                content_vector: to_f32(&row.get(5)?),
            })
        },
    ).with_context(|| "insert chunk failed.")
}

#[test]
fn test_insert_query_chunk() {
    let mut conn = Connection::open_in_memory().unwrap();
    run_migrations(&mut conn).unwrap();
    let doc = insert_document(&mut conn, "test").unwrap();
    let chunk = insert_chunk(&mut conn, doc.id, 2, 3, "test", &vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(chunk.content, "test");
    assert_eq!(chunk.content_vector, vec![1.0, 2.0, 3.0]);
}

pub(crate) fn insert_entity(
    conn: &mut Connection,
    name: &str,
    embedding: &Vec<f32>,
) -> Result<Entity> {
    conn.query_row(
        "INSERT INTO entity (name, embedding) VALUES (?, ?)",
        (name, to_binary_string(embedding)),
        |row| {
            Ok(Entity {
                id: row.get(0)?,
                name: row.get(1)?,
                embedding: to_f32(&row.get(2)?),
            })
        },
    )
    .with_context(|| "insert entity failed.")
}

pub(crate) fn insert_relation(
    conn: &mut Connection,
    source_id: i64,
    target_id: i64,
    relationship: &str,
) -> Result<Relation> {
    conn.query_row(
        "INSERT INTO relation (source_id, target_id, relationship) VALUES (?, ?, ?)",
        (source_id, target_id, relationship),
        |f| {
            Ok(Relation {
                id: f.get(0)?,
                source_id: f.get(1)?,
                target_id: f.get(2)?,
                relationship: f.get(3)?,
            })
        },
    )
    .with_context(|| "insert relation failed.")
}

pub(crate) fn insert_entity_chunk(
    conn: &mut Connection,
    entity_id: i64,
    chunk_id: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO entity_chunk (entity_id, chunk_id) VALUES (?, ?)",
        (entity_id, chunk_id),
    )?;
    Ok(())
}

pub(crate) fn query_all_documents(conn: &mut Connection) -> Result<Vec<Document>> {
    let mut stmt = conn.prepare("SELECT id, doc_name FROM document")?;
    let doc_iter = stmt.query_map([], |row| {
        Ok(Document {
            id: row.get(0)?,
            doc_name: row.get(1)?,
        })
    })?;
    let mut res = Vec::new();
    for doc in doc_iter {
        res.push(doc.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_all_chunks(conn: &mut Connection) -> Result<Vec<Chunk>> {
    let mut stmt = conn.prepare(
        "SELECT id, full_doc_id, chunk_index, tokens, content, content_vector FROM chunk",
    )?;
    let chunk_iter = stmt.query_map([], |row| {
        Ok(Chunk {
            id: row.get(0)?,
            full_doc_id: row.get(1)?,
            chunk_index: row.get(2)?,
            tokens: row.get(3)?,
            content: row.get(4)?,
            content_vector: to_f32(&row.get(5)?),
        })
    })?;
    let mut res = Vec::new();
    for chunk in chunk_iter {
        res.push(chunk.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_all_entities(conn: &mut Connection) -> Result<Vec<Entity>> {
    let mut stmt = conn.prepare("SELECT id, name, embedding FROM entity")?;
    let entity_iter = stmt.query_map([], |row| {
        Ok(Entity {
            id: row.get(0)?,
            name: row.get(1)?,
            embedding: to_f32(&row.get(2)?),
        })
    })?;
    let mut res = Vec::new();
    for entity in entity_iter {
        res.push(entity.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_all_relations(conn: &mut Connection) -> Result<Vec<Relation>> {
    let mut stmt = conn.prepare("SELECT id, source_id, target_id, relationship FROM relation")?;
    let relation_iter = stmt.query_map([], |row| {
        Ok(Relation {
            id: row.get(0)?,
            source_id: row.get(1)?,
            target_id: row.get(2)?,
            relationship: row.get(3)?,
        })
    })?;
    let mut res = Vec::new();
    for relation in relation_iter {
        res.push(relation.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_chunks_by_doc_id(
    conn: &mut Connection,
    doc_id: i64,
) -> Result<Vec<super::Chunk>> {
    let mut stmt = conn.prepare(
        "SELECT id, full_doc_id, chunk_index, tokens, content, content_vector FROM chunk WHERE full_doc_id = ?",
    )?;
    let chunk_iter = stmt.query_map([doc_id], |row| {
        Ok(super::Chunk {
            id: row.get(0)?,
            full_doc_id: row.get(1)?,
            chunk_index: row.get(2)?,
            tokens: row.get(3)?,
            content: row.get(4)?,
            content_vector: to_f32(&row.get(5)?),
        })
    })?;
    let mut res = Vec::new();
    for chunk in chunk_iter {
        res.push(chunk.unwrap());
    }
    Ok(res)
}
