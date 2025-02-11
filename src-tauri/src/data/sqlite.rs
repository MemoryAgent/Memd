use anyhow::Result;
use rusqlite::Connection;

use super::Document;

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

pub(crate) fn insert_document(conn: &mut Connection, doc_name: &str) -> Result<()> {
    conn.execute("INSERT INTO document (doc_name) VALUES (?)", &[doc_name])?;
    Ok(())
}

fn to_binary_string(v: &Vec<f32>) -> Vec<u8> {
    let mut res = Vec::with_capacity(v.len() * 4);
    for x in v {
        res.extend_from_slice(&x.to_le_bytes());
    }
    res
}

pub(crate) fn insert_chunk(
    conn: &mut Connection,
    full_doc_id: i64,
    chuck_idx: i64,
    tokens: usize,
    content: &str,
    content_vector: &Vec<f32>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO chunk (full_doc_id, chuck_idx, tokens, content, content_vector) VALUES (?, ?, ?, ?, ?)",
        (full_doc_id, chuck_idx, tokens, content, to_binary_string(content_vector)),
    )?;
    Ok(())
}

pub(crate) fn insert_entity(
    conn: &mut Connection,
    name: &str,
    embedding: &Vec<f32>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO entity (name, embedding) VALUES (?, ?)",
        (name, to_binary_string(embedding)),
    )?;
    Ok(())
}

pub(crate) fn insert_relation(
    conn: &mut Connection,
    source_id: i64,
    target_id: i64,
    relationship: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO relation (source_id, target_id, relationship) VALUES (?, ?, ?)",
        (source_id, target_id, relationship),
    )?;
    Ok(())
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

pub(crate) fn query_all_chunks(conn: &mut Connection) -> Result<Vec<super::Chunk>> {
    let mut stmt = conn.prepare(
        "SELECT id, full_doc_id, chuck_idx, tokens, content, content_vector FROM chunk",
    )?;
    let chunk_iter = stmt.query_map([], |row| {
        Ok(super::Chunk {
            id: row.get(0)?,
            full_doc_id: row.get(1)?,
            chuck_idx: row.get(2)?,
            tokens: row.get(3)?,
            content: row.get(4)?,
            content_vector: Vec::new(),
        })
    })?;
    let mut res = Vec::new();
    for chunk in chunk_iter {
        res.push(chunk.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_all_entities(conn: &mut Connection) -> Result<Vec<super::Entity>> {
    let mut stmt = conn.prepare("SELECT id, name, embedding FROM entity")?;
    let entity_iter = stmt.query_map([], |row| {
        Ok(super::Entity {
            id: row.get(0)?,
            name: row.get(1)?,
            embedding: Vec::new(),
        })
    })?;
    let mut res = Vec::new();
    for entity in entity_iter {
        res.push(entity.unwrap());
    }
    Ok(res)
}

pub(crate) fn query_all_relations(conn: &mut Connection) -> Result<Vec<super::Relation>> {
    let mut stmt =
        conn.prepare("SELECT id, source_id, target_id, relationship FROM relation")?;
    let relation_iter = stmt.query_map([], |row| {
        Ok(super::Relation {
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
        "SELECT id, full_doc_id, chuck_idx, tokens, content, content_vector FROM chunk WHERE full_doc_id = ?",
    )?;
    let chunk_iter = stmt.query_map([doc_id], |row| {
        Ok(super::Chunk {
            id: row.get(0)?,
            full_doc_id: row.get(1)?,
            chuck_idx: row.get(2)?,
            tokens: row.get(3)?,
            content: row.get(4)?,
            content_vector: Vec::new(),
        })
    })?;
    let mut res = Vec::new();
    for chunk in chunk_iter {
        res.push(chunk.unwrap());
    }
    Ok(res)
}
