use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::{
    component::{
        bert::encode_single_sentence,
        database::{Chunk, Store},
        deepseek::extract_answer,
        llm::Llm,
        LocalComponent,
    },
    method::cluster::cluster_by_kmeans,
};

use super::{
    buffer_pool::BufferPool,
    index_file::IndexFile,
    page::{self, get_internal_reader_from_buffer, get_leaf_reader_from_buffer, RecordID},
};

#[derive(Clone, Debug)]
pub struct TopKBuffer {
    /// vector ID -> conf_score
    buffer: Vec<(usize, f32)>,
    /// top k
    k: usize,
}

impl TopKBuffer {
    pub fn new(k: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(k),
            k,
        }
    }

    pub fn push(&mut self, id: usize, score: f32) {
        if self.buffer.len() < self.k {
            self.buffer.push((id, score));
            // Total order comparison of floats is supported since Rust 1.62
            self.buffer.sort_by(|a, b| a.1.total_cmp(&b.1));
        } else if score > self.buffer[0].1 {
            self.buffer[0] = (id, score);
            self.buffer.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
    }

    pub fn get_topk(&self) -> &[(usize, f32)] {
        &self.buffer
    }
}

/// TODO: this is to be replaced by FAISS because I don't want to write SIMD optimizations ...
fn calculate_similarity(vec0: &[f32], vec1: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..vec0.len() {
        sum += vec0[i] * vec1[i];
    }
    sum / (vec0.len() as f32)
}

/// reinterpret &[u8] as &[f32]
fn reinterpret_as_f32<'a>(slice: &'a [u8]) -> &'a [f32] {
    assert!(slice.len() % std::mem::size_of::<f32>() == 0);
    let len = slice.len() / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, len) }
}

fn reinterpret_as_u8(slice: &[f32]) -> &[u8] {
    assert!(slice.len() % std::mem::size_of::<f32>() == 0);
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

impl Llm {
    fn get_summary_prompt(text: &[String]) -> String {
        let mut prompt =
            "Please summarize the following text, include as many details as possible:\n"
                .to_string();
        for (i, line) in text.iter().enumerate() {
            prompt.push_str(&format!("{}: {}\n", i + 1, line));
        }
        prompt.push_str("Summary:");
        prompt
    }
}

#[derive(Clone, Debug)]
pub struct InternalIndexEntry {
    pub summary: String,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct InternalIndexRow {
    pub vec_id: usize,
    pub summary: String,
    pub embedding: Vec<f32>,
}

// sqlite functions
fn insert_internal_index(
    conn: &mut rusqlite::Connection,
    index_entry: &InternalIndexEntry,
) -> Result<InternalIndexRow> {
    use crate::component::sqlite;
    conn.query_row(
        "INSERT INTO internal_vector_index (summary, embedding) VALUES (?, ?) RETURNING id",
        (
            &index_entry.summary,
            &sqlite::to_binary_string(&index_entry.embedding),
        ),
        |row| {
            let id: usize = row.get(0)?;
            Ok(InternalIndexRow {
                vec_id: id,
                summary: index_entry.summary.clone(),
                embedding: index_entry.embedding.clone(),
            })
        },
    )
    .with_context(|| {
        format!(
            "Failed to insert internal index entry: {}",
            index_entry.summary
        )
    })
}

fn insert_internal_relationship(
    conn: &mut rusqlite::Connection,
    parent_id: usize,
    child_id: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO internal_vector_children (parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )
    .with_context(|| {
        format!(
            "Failed to insert internal index relationship: {} -> {}",
            parent_id, child_id
        )
    })?;
    Ok(())
}

// db operations

impl Store {
    pub fn insert_internal_index(
        &mut self,
        index_entry: &InternalIndexEntry,
    ) -> Result<InternalIndexRow> {
        let mut conn = self.conn.lock().unwrap();
        insert_internal_index(&mut conn, index_entry)
    }

    pub fn insert_internal_relationship(&mut self, parent_id: usize, child_id: i64) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        insert_internal_relationship(&mut conn, parent_id, child_id)
    }
}

pub struct ShmIndex {
    buffer_pool: BufferPool,
    max_page_vectors: usize,
}

#[derive(Clone, Debug)]
pub struct ShmIndexOptions {
    /// The path to the file that will be used for backing store.
    pub backed_file: PathBuf,
    /// The size of the buffer pool.
    pub pool_size: usize,
    /// The size of each page.
    pub page_size: usize,
    /// The size of each vector unit.
    pub vector_unit_size: usize,
}

#[derive(Clone, Debug)]
pub struct InternalChunk {
    summary: String,
    embedding: Vec<f32>,
    child_page_id: usize,
    vec_id: usize,
}

impl ShmIndex {
    pub fn new(options: ShmIndexOptions) -> Result<Self> {
        let backed_file = IndexFile::create(
            options.backed_file,
            options.page_size,
            options.vector_unit_size,
        )?;
        let buffer_pool = BufferPool::new(
            backed_file,
            options.pool_size,
            options.page_size,
            options.vector_unit_size,
        );

        let max_page_vectors = page::calculate_max_vectors_in_internal_page(
            options.page_size,
            options.vector_unit_size,
        );

        Ok(Self {
            buffer_pool,
            max_page_vectors,
        })
    }

    pub fn load(file: PathBuf, pool_size: usize) -> Result<Self> {
        let backed_file = IndexFile::open(file)?;
        let page_size = backed_file.page_size;
        let vector_unit_size = backed_file.vector_unit_size;

        let buffer_pool = BufferPool::new(backed_file, pool_size, page_size, vector_unit_size);

        let max_page_vectors =
            page::calculate_max_vectors_in_internal_page(page_size, vector_unit_size);

        Ok(Self {
            buffer_pool,
            max_page_vectors,
        })
    }

    pub fn persist(&mut self) -> Result<()> {
        self.buffer_pool.flush_all();
        Ok(())
    }

    // assume chunks is needed to split
    // every call into this function will build upper one level node
    // returns current level nodes
    fn bulk_build_recursive(
        &mut self,
        chunks: &[InternalChunk],
        local_comps: &mut LocalComponent,
    ) -> Vec<InternalChunk> {
        if chunks.len() <= self.max_page_vectors {
            return chunks.to_vec();
        }

        let embeddings: Vec<Vec<f32>> =
            chunks.iter().map(|chunk| chunk.embedding.clone()).collect();

        let num_clusters = chunks.len() / 2;

        let cluster_labels = cluster_by_kmeans(&embeddings, num_clusters);

        let mut clusters: Vec<Vec<InternalChunk>> = vec![Vec::new(); num_clusters];

        for (i, label) in cluster_labels.iter().enumerate() {
            clusters[*label].push(chunks[i].clone());
        }

        let higher_level_internal_chunks: Vec<InternalChunk> = clusters
            .iter()
            .map(|cluster| {
                let texts: Vec<String> =
                    cluster.iter().map(|chunk| chunk.summary.clone()).collect();
                let prompt = Llm::get_summary_prompt(&texts);
                let summary_whole = local_comps.llm.complete(&prompt).unwrap();
                let (_, summary) = extract_answer(&summary_whole);
                let embedding =
                    encode_single_sentence(&summary, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                let entry = InternalIndexEntry {
                    summary: summary.to_string(),
                    embedding: embedding.to_vec1().unwrap(),
                };
                let row = local_comps.store.insert_internal_index(&entry).unwrap();

                // TODO: maintain parent information
                let internal_page_id = self.buffer_pool.create_internal_page(0);
                let internal_page = self.buffer_pool.fetch_page(internal_page_id);
                let mut internal_page_accessor =
                    get_internal_reader_from_buffer(internal_page.get_page());

                for chunk in cluster.iter() {
                    local_comps
                        .store
                        .insert_internal_relationship(row.vec_id, chunk.vec_id as i64)
                        .unwrap();
                    internal_page_accessor
                        .append_record(
                            chunk.child_page_id,
                            reinterpret_as_u8(&chunk.embedding),
                            chunk.child_page_id,
                        )
                        .unwrap();
                }

                InternalChunk {
                    summary: entry.summary.clone(),
                    embedding: entry.embedding.clone(),
                    child_page_id: internal_page_id,
                    vec_id: row.vec_id,
                }
            })
            .collect();

        // tail recursion
        self.bulk_build_recursive(&higher_level_internal_chunks, local_comps)
    }

    pub fn bulk_build(&mut self, chunks: &[Chunk], local_comps: &mut LocalComponent) {
        let embeddings: Vec<Vec<f32>> = chunks
            .iter()
            // TODO: optimiza this by slicing
            .map(|chunk| chunk.content_vector.clone())
            .collect();

        // TODO: optimiza this data transfer?
        // TODO: how to decide k? BIC
        let cluster_labels = cluster_by_kmeans(&embeddings, chunks.len() / 2);

        // group chunks by cluster labels
        let mut clusters: Vec<Vec<Chunk>> = vec![Vec::new(); chunks.len() / 2];
        for (i, label) in cluster_labels.iter().enumerate() {
            clusters[*label].push(chunks[i].clone());
        }

        let first_internal_layer: Vec<InternalChunk> = clusters
            .iter()
            .map(|cluster| {
                let texts: Vec<String> =
                    cluster.iter().map(|chunk| chunk.content.clone()).collect();
                let prompt = Llm::get_summary_prompt(&texts);
                let summary_whole = local_comps.llm.complete(&prompt).unwrap();
                let (_, summary) = extract_answer(&summary_whole);
                let embedding =
                    encode_single_sentence(&summary, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                let index_entry = InternalIndexEntry {
                    summary: summary.to_string(),
                    embedding: embedding.to_vec1().unwrap(),
                };
                let index_row = local_comps
                    .store
                    .insert_internal_index(&index_entry)
                    .unwrap();

                let leaf_page_id = self.buffer_pool.create_leaf_page();
                let leaf_page = self.buffer_pool.fetch_page(leaf_page_id);
                let mut leaf_page_accessor = get_leaf_reader_from_buffer(leaf_page.get_page());

                for chunk in cluster.iter() {
                    local_comps
                        .store
                        .insert_internal_relationship(index_row.vec_id, chunk.id)
                        .unwrap();
                    leaf_page_accessor
                        .append_record(
                            chunk.id.try_into().unwrap(),
                            reinterpret_as_u8(&chunk.content_vector),
                        )
                        .unwrap();
                }

                InternalChunk {
                    summary: index_entry.summary.clone(),
                    embedding: index_entry.embedding.clone(),
                    child_page_id: leaf_page_id,
                    vec_id: index_row.vec_id,
                }
            })
            .collect();

        // top level
        let internal_page_id = self.buffer_pool.create_internal_page(0);
        let internal_page = self.buffer_pool.fetch_page(internal_page_id);
        let mut internal_page_accessor = get_internal_reader_from_buffer(internal_page.get_page());

        let top_level_internal_chunks =
            self.bulk_build_recursive(&first_internal_layer, local_comps);

        for chunk in top_level_internal_chunks.iter() {
            internal_page_accessor
                .append_record(
                    chunk.child_page_id,
                    reinterpret_as_u8(&chunk.embedding),
                    chunk.child_page_id,
                )
                .unwrap();
        }

        self.buffer_pool.flush_all();
    }

    /// insert one vector into the index
    pub fn insert(&mut self, vector_data: &[u8], vector_id: usize) -> Result<RecordID> {
        todo!()
    }

    pub fn query(&mut self, target_vector: &[u8]) -> Result<TopKBuffer> {
        todo!()
    }
}

impl Drop for ShmIndex {
    fn drop(&mut self) {
        self.persist().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::component::operation;

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_shm_index() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4,
        })
        .unwrap();
        let vector = 4.0_f32.to_le_bytes();
        let id = index.insert(&vector, 0).unwrap();
        println!("inserted rid: {:?}", id);
        assert_eq!(id.page_id, 0);
        assert_eq!(id.slot_id, 0);
        let target_vector = 4.0_f32.to_le_bytes();
        let topk = index.query(&target_vector).unwrap();
        println!("topk: {:?}", topk.get_topk());
    }

    #[test]
    fn test_bulk_build() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test_bulk_build.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4 * 384,
        })
        .unwrap();
        let mut local_comps = LocalComponent::default();

        let full_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        let inserted_full_text = local_comps
            .store
            .add_document(&operation::Document {
                name: "hi".to_string(),
                content: full_text.to_string(),
            })
            .unwrap();

        let text = vec![
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];

        let chunks: Vec<Chunk> = text
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let embedding =
                    encode_single_sentence(text, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                operation::Chunk {
                    full_doc_id: inserted_full_text.id,
                    chunk_index: 0,
                    tokens: 20,
                    content: text.clone(),
                    embedding,
                }
            })
            .map(|chunk| local_comps.store.add_chunk(&chunk).unwrap())
            .collect();

        index.bulk_build(&chunks, &mut local_comps);

        index.persist().unwrap();
    }

    #[test]
    fn test_bulk_build_two_layer() {
        let mut index = ShmIndex::new(ShmIndexOptions {
            backed_file: PathBuf::from("test_bulk_build_two_layer.bin"),
            pool_size: 10,
            page_size: 4096,
            vector_unit_size: 4 * 384,
        })
        .unwrap();
        let mut local_comps = LocalComponent::default();

        let full_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

        let inserted_full_text = local_comps
            .store
            .add_document(&operation::Document {
                name: "hi".to_string(),
                content: full_text.to_string(),
            })
            .unwrap();

        let text = vec![
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "ccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "ccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "ddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "ddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
        ];

        let chunks: Vec<Chunk> = text
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let embedding =
                    encode_single_sentence(text, &mut local_comps.tokenizer, &local_comps.bert)
                        .unwrap();
                operation::Chunk {
                    full_doc_id: inserted_full_text.id,
                    chunk_index: 0,
                    tokens: 20,
                    content: text.clone(),
                    embedding,
                }
            })
            .map(|chunk| local_comps.store.add_chunk(&chunk).unwrap())
            .collect();

        index.bulk_build(&chunks, &mut local_comps);

        index.persist().unwrap();
    }
}
